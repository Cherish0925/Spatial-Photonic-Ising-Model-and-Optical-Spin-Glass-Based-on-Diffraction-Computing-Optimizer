# -*- coding: utf-8 -*-
"""
Single-SLM Double-Phase Hologram (DPH) encoder for real-valued maps in [-1, 1].
Implements the on-axis, single-pixel DPH with complementary checkerboards:

Target real field: U = ξ, ξ ∈ [-1, 1]
Map to amplitude/phase: A = |ξ|, φ = 0 (ξ>=0) or π (ξ<0)
DPH (Amax = 1  ->  B = 1/2):
    θ± = φ ± arccos(A/Amax) = φ ± arccos(A)        (since Amax = 1)
Single SLM phase to load:
    α = M1·θ+ + M2·θ-             (checkerboard multiplexing)
    phase01 = (α mod 2π) / (2π)    in [0,1]  →  load on LCOS (0..1 → 0..2π)

Additionally:
- pack_vectors_to_square(vectors, pixel_repeat=...) packs (S,r) -> [r,N,N] and then
  repeats the last two dims by 'pixel_repeat' (giant-pixel upsampling by nearest).

Author: you
"""

import math
import numpy as np
import torch
from Bandlimited_ASM import _center_embed, _center_crop, BandlimitedASM
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体支持中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ----------------------------- packing (S, r) -> [r, N, N] (+ giant pixels) -----------------------------
def pack_vectors_to_square(vectors,
                           return_torch=True,
                           device=None,
                           dtype=None,
                           pixel_repeat: int = 1):
    """
    Column-major packing: (S, r) -> [r, N, N], N = ceil(sqrt(S)), pad with zeros.
    Then repeat the last two dims by 'pixel_repeat' (giant pixels).

    Parameters
    ----------
    vectors : np.ndarray or torch.Tensor of shape (S, r)
    return_torch : bool, return torch.Tensor when True; else np.ndarray
    device, dtype : used only when return_torch=True
    pixel_repeat : int >= 1. If >1, output becomes [r, N*pixel_repeat, N*pixel_repeat]

    Returns
    -------
    out : torch.Tensor or np.ndarray, shape [r, N', N'] where N' = N * pixel_repeat
    """
    if isinstance(vectors, torch.Tensor):
        vec_np = vectors.detach().cpu().numpy()
    else:
        vec_np = np.asarray(vectors)

    if vec_np.ndim != 2:
        raise ValueError("vectors must be 2D (S, r)")

    S, r = vec_np.shape
    if S <= 0 or r <= 0:
        raise ValueError("vectors shape must be (S>0, r>0)")

    N = int(math.ceil(math.sqrt(S)))
    total = N * N
    pad = total - S

    out_np = np.zeros((r, N, N), dtype=vec_np.dtype)
    for k in range(r):
        col = vec_np[:, k]
        if pad > 0:
            col = np.pad(col, (0, pad), mode="constant", constant_values=0.0)
        out_np[k] = np.reshape(col, (N, N), order="F")  # column-major fill

    if not return_torch:
        if pixel_repeat > 1:
            out_np = out_np.repeat(pixel_repeat, axis=1).repeat(pixel_repeat, axis=2)
        return out_np

    # torch branch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32
    out = torch.from_numpy(out_np).to(device=device, dtype=dtype)
    if pixel_repeat > 1:
        out = out.repeat_interleave(pixel_repeat, dim=-2).repeat_interleave(pixel_repeat, dim=-1)
    return out


# ----------------------------- checkerboard (superpixel W×W) -----------------------------
def build_checkerboard(N, tile=1, device=None, dtype=torch.float32):
    """
    M1 is 1 on even tiles, 0 otherwise; M2 = 1 - M1.
    If tile=W>1, checkerboard period is W (superpixel).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    i = torch.arange(N, device=device).view(-1, 1)
    j = torch.arange(N, device=device).view(1, -1)
    if tile <= 1:
        M1 = (((i + j) & 1) == 0).to(dtype)
    else:
        M1 = (((torch.div(i, tile, rounding_mode="floor") +
                torch.div(j, tile, rounding_mode="floor")) & 1) == 0).to(dtype)
    M2 = 1.0 - M1
    return M1, M2


# ----------------------------- DPH for real-valued ξ ∈ [-1, 1] (Amax=1 → B=1/2) -----------------------------
def dph_encode_real_single_slm(grids_real,
                               tile: int = 1,
                               to01: bool = True,
                               clamp: bool = True):
    r"""
    Encode real-valued target U = ξ (ξ ∈ [-1,1]) into a single phase-only map using DPH:

        A = |ξ|,  φ = 0 (ξ>=0) or π (ξ<0)
        θ± = φ ± arccos(A)            (Amax = 1)
        α  = M1·θ+ + M2·θ-            (checkerboard multiplexing)

    Output:
        phase01 = (α mod 2π) / (2π) ∈ [0,1], same shape as input grids_real

    Parameters
    ----------
    grids_real : torch.Tensor [B,N,N] or [N,N] with values in [-1,1]
    tile       : checkerboard superpixel size (1 = Nyquist alternation)
    to01       : normalize to [0,1] (otherwise return α/(2π) possibly outside [0,1])
    clamp      : clamp input to [-1,1] before acos

    Returns
    -------
    phase01 : torch.Tensor [B,N,N] or [N,N], in [0,1]
    B_const : torch.Tensor scalar or [B]-shaped, equals 1/2 (since Amax=1)
    """
    if grids_real.ndim == 2:
        grids_real = grids_real.unsqueeze(0)
    Bsz, N, _ = grids_real.shape
    device, dtype = grids_real.device, grids_real.dtype

    X = grids_real
    if clamp:
        X = torch.clamp(X, -1.0, 1.0)

    # A=|ξ|, φ=0/π
    A = torch.abs(X)
    phi = torch.where(X >= 0, torch.zeros_like(X), torch.full_like(X, math.pi))

    # Amax = 1 → θ± = φ ± arccos(A)
    alpha_mag = torch.acos(A)  # in [0, π/2] when A∈[0,1]
    theta_plus  = phi + alpha_mag
    theta_minus = phi - alpha_mag

    # checkerboard multiplex to single phase
    M1, M2 = build_checkerboard(N, tile=tile, device=device, dtype=dtype)
    M1 = M1.expand(Bsz, -1, -1)
    M2 = M2.expand(Bsz, -1, -1)
    alpha = M1 * theta_plus + M2 * theta_minus  # radians

    if to01:
        phase01 = torch.remainder(alpha, 2*math.pi) / (2*math.pi)
    else:
        phase01 = alpha / (2*math.pi)

    # B = Amax/2 = 1/2
    B_const = torch.full((Bsz,), 0.5, device=device, dtype=dtype)
    return phase01.squeeze(0) if phase01.shape[0] == 1 else phase01, B_const.squeeze(0) if Bsz == 1 else B_const

# ====== 简单画一个 160×160 的“6”灰度图，归一化到 [0,1] ======
def make_digit_six(N=160):
    img = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(img)
    # 尝试自适应字号与居中
    txt = "6"
    # 粗略选一个字号填满画布
    fontsize = int(N*0.9)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    w, h = draw.textbbox((0,0), txt, font=font)[2:]
    scale = min((N*0.85)/max(1,w), (N*0.85)/max(1,h))
    fontsize = max(1, int(fontsize*scale))
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    w, h = draw.textbbox((0,0), txt, font=font)[2:]
    x = (N - w)//2; y = (N - h)//2
    draw.text((x,y), txt, fill=255, font=font)
    arr = np.asarray(img, dtype=np.float32)/255.0
    return torch.from_numpy(arr)

# =================================== 主流程 ===================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N       = 160
    lam     = 634e-9
    dx      = 8e-6
    f       = 100e-3
    dtype_c = torch.complex64
    dtype   = torch.float32

    # 生成一个“6”的强度图，作为实数 ξ
    six = make_digit_six(int(N/4)).to(device=device, dtype=dtype)
    six=(six.repeat_interleave(4,dim=0)).repeat_interleave(4,dim=1)

    xi  = six.unsqueeze(0)
    print(rf"xi.shape={xi.shape},xi.dtype={xi.dtype}")
    plt.figure(1)
    plt.imshow(xi[0].cpu())
    plt.title("Pure amplitude images that need to be achieved through biphasic encoding")

    # DPH 相位板（0..1 → ×2π）
    phase01, B = dph_encode_real_single_slm(xi, tile=1, to01=True, clamp=True)
    phase01 = phase01.to(device)
    print(rf"phase01.shape={phase01.shape},phase01.dtype={phase01.dtype}")


    asm_1mm = BandlimitedASM(wavelength=lam, N_size=N, pixel_size=dx,
                         zero_fill=1, is_lens=False, distance=1e-3,
                         sampling="midpoint", dtype=dtype_c, device=device, verbose=True)

    asm = BandlimitedASM(wavelength=lam, N_size=N, pixel_size=dx,
                         zero_fill=1, is_lens=True, distance=f,
                         sampling="midpoint", dtype=dtype_c, device=device, verbose=True)

    Ein=1.0*torch.exp(-1j*2.0*torch.pi*phase01.unsqueeze(0))
    print(rf"Ein.shape={Ein.shape},Ein.dtype={Ein.dtype}")
    E_1mm_out=asm_1mm(Ein)
    print(rf"E_1mm_out.shape={E_1mm_out.shape},E_1mm_out.dtype={E_1mm_out.dtype}")

    amp_out = torch.abs(E_1mm_out)[0].detach().cpu().numpy()
    phase_out = torch.angle(E_1mm_out)[0].detach().cpu().numpy()
    phase_out = np.mod(phase_out, 2*np.pi)
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle("The complex amplitude after the propagation of the biphasic hologram for 1mm")
    cmap_amp = 'gray'
    cmap_phase = 'twilight'

    im0 = axs[0].imshow(amp_out, cmap=cmap_amp)
    axs[0].set_title("After 1mm Amplitude with Input")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(phase_out, cmap=cmap_phase)
    axs[1].set_title("After 1mm Phase with Input")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    for ax in axs.ravel():
        ax.axis("on")

    plt.tight_layout()


    # 4F传播
    # Eimg = fourf_run_with_lp_using_asm(asm, phase01.unsqueeze(0), lp_cut=None, tile=1)  # [1,N,N] complex
    Emiddle = asm(Ein)

    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle("The amplitude and phase maps on the frequency domain surface during the propagation of the biphasic hologram")
    amp_out = torch.abs(Emiddle)[0].detach().cpu().numpy()
    phase_out = torch.angle(Emiddle)[0].detach().cpu().numpy()
    phase_out = np.mod(phase_out, 2 * np.pi)
    im0 = axs[0].imshow(amp_out, cmap=cmap_amp)
    axs[0].set_title("the frequency domain surface Amplitude")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(phase_out, cmap=cmap_phase)
    axs[1].set_title("the frequency domain surface Phase")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    # 在 SLM 面上是空间分离的子像素网格，其频谱包含许多高频衍射级次只有0阶低频部分包含你想要的目标复振幅信息
    # 其他高频部分是由于棋盘格调制引入的混叠项，会在空间叠加成背景噪声。
    # Low-pass filtering in the frequency domain surface

    Eimg=asm(Emiddle)
    print(rf"Eimg.shape={Eimg.shape},Eimg.dtype={Eimg.dtype}")
    amp_out = torch.abs(Eimg)[0].detach().cpu().numpy()
    phase_out = torch.angle(Eimg)[0].detach().cpu().numpy()
    phase_out = np.mod(phase_out, 2*np.pi)

    # 入射场（单位振幅 + 相位 = DPH 相位板）
    amp_in = torch.ones_like(phase01).detach().cpu().numpy()
    phase_in = phase01.detach().cpu().numpy() * 2*np.pi  # [0,2π]

    # 倒像检查
    six_flip = torch.flip(xi, dims=[1,2])[0].detach().cpu().numpy()
    a = amp_out - amp_out.mean()
    b = six_flip - six_flip.mean()
    corr = (a*b).sum() / (np.sqrt((a*a).sum())*np.sqrt((b*b).sum()) + 1e-12)
    print(f"[Check] correlation(output amplitude vs. flipped target) = {corr:.4f}")
    print("amp_out stats:", float(amp_out.min()), float(amp_out.max()))

    # ----------------------- 可视化部分 -----------------------
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle("The complex amplitude images of the biphasic hologram before and after passing through the 4F system")


    im0 = axs[0,0].imshow(amp_in, cmap=cmap_amp)
    axs[0,0].set_title("Input Amplitude (Uniform)")
    plt.colorbar(im0, ax=axs[0,0], fraction=0.046)

    im1 = axs[0,1].imshow(phase_in, cmap=cmap_phase)
    axs[0,1].set_title("Input Phase (DPH encoded)")
    plt.colorbar(im1, ax=axs[0,1], fraction=0.046)

    im2 = axs[1,0].imshow(amp_out, cmap=cmap_amp)
    axs[1,0].set_title("Output Amplitude (after 4F)")
    plt.colorbar(im2, ax=axs[1,0], fraction=0.046)

    im3 = axs[1,1].imshow(phase_out, cmap=cmap_phase)
    axs[1,1].set_title("Output Phase (after 4F)")
    plt.colorbar(im3, ax=axs[1,1], fraction=0.046)

    for ax in axs.ravel():
        ax.axis("on")

    plt.tight_layout()
    plt.show()

# ---------- 小工具 ----------
def to_np2d(x):
    """torch/numpy → 2D numpy（squeeze 到 [H,W]）"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    else:
        x = np.asarray(x).squeeze()
    assert x.ndim == 2, f"Expect 2D for imshow, got {x.shape}"
    return x

def make_lowpass_mask_spatial(N, cutoff_cyc_per_pix, device, dtype):
    """
    在“Emiddle”的采样网格上生成圆形低通掩膜（简洁做法：以像素为单位的频率半径）。
    Emiddle 与输入同尺寸 N×N，因此直接在 N×N 网格上做中心圆形 mask。
    cutoff_cyc_per_pix 建议 0.35~0.5（tile=1）
    """
    yy, xx = torch.meshgrid(
        torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij'
    )
    cy = (N - 1) / 2.0
    cx = (N - 1) / 2.0
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    # 把像素半径线性映射为“伪”频率半径：最大半径对应 0.5 cycles/pixel
    # 因此 r = (N/2) -> 0.5 cyc/pix，得到像素半径阈值：
    r_cut = cutoff_cyc_per_pix / 0.5 * (N / 2.0)
    mask = (r <= r_cut).to(dtype)
    return mask

def imshow2x2(figtitle, images, titles, cmaps, vmins=None, vmaxs=None):
    """一个 figure 内绘制 2×2 子图 + colorbar"""
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(figtitle, fontsize=14)
    for k, ax in enumerate(axs.ravel()):
        vmin = None if vmins is None else vmins[k]
        vmax = None if vmaxs is None else vmaxs[k]
        im = ax.imshow(images[k], cmap=cmaps[k], vmin=vmin, vmax=vmax)
        ax.set_title(titles[k], fontsize=11)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.ax.tick_params(labelsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

import math
import torch

def dph_encode_complex_single_slm(
    A,                      # [B,N,N] or [N,N],  目标振幅 ∈ [0, 2B]
    phi,                    # [B,N,N] or [N,N],  目标相位（弧度）任意实数
    tile: int = 1,          # 棋盘/超像素的周期
    B: float = 0.5,         # 两个子相位分量的等幅系数（经典取 B=1/2 → 2B=1）
    to01: bool = True,      # 输出是否归一化到 [0,1] (×2π 上 SLM)
    clamp_amp: bool = True  # 是否把振幅 clamp 到 [0, 2B]
):
    r"""
    单片 SLM 的双相位（DPH）编码，把目标复场 U = A·exp(iφ) 映射为单张相位片：
        令 θ± = φ ± arccos(A / (2B))，并用棋盘 M1/M2 交错复用：
            α = M1·θ+ + M2·θ−
        最终上片相位 phase01 = (α mod 2π)/(2π) ∈ [0,1]

    要点：
    - 选择 B=1/2 则要求 A ∈ [0,1]（即 A ≤ 2B），这样 acos(A/(2B)) 有定义；
    - 若 A 超界，且 clamp_amp=True，会自动截断到 [0,2B]。
    """
    # 统一 batch 维
    if A.ndim == 2: A = A.unsqueeze(0)
    if phi.ndim == 2: phi = phi.unsqueeze(0)
    assert A.shape == phi.shape, f"A and phi must have same shape, got {A.shape} vs {phi.shape}"

    Bsz, N, _ = A.shape
    device, dtype = A.device, A.dtype

    # 幅度规范：A ∈ [0, 2B]
    if clamp_amp:
        A = torch.clamp(A, 0.0, 2.0 * B)
    # 数值安全：传入 acos 的量
    x = torch.clamp(A / (2.0 * B), 0.0, 1.0)

    # θ± = φ ± arccos(A/(2B))
    alpha_mag = torch.acos(x)  # ∈ [0, π/2]
    theta_plus  = phi + alpha_mag
    theta_minus = phi - alpha_mag

    # 棋盘交错复用
    M1, M2 = build_checkerboard(N, tile=tile, device=device, dtype=dtype)
    M1 = M1.expand(Bsz, -1, -1)
    M2 = M2.expand(Bsz, -1, -1)
    alpha = M1 * theta_plus + M2 * theta_minus  # radians

    # 归一化到 [0,1]（上片时 ×2π）
    if to01:
        phase01 = torch.remainder(alpha, 2*math.pi) / (2*math.pi)
    else:
        phase01 = alpha / (2*math.pi)

    # 按 DPH 理论，等幅常数 B（返回便于你做能量标定）
    B_const = torch.full((Bsz,), float(B), device=device, dtype=dtype)
    return (phase01.squeeze(0) if Bsz==1 else phase01,
            B_const.squeeze(0) if Bsz==1 else B_const)


# ===== 新增：把任意文本渲染为 NxN 的 [0,1] 灰度 mask，用作目标相位图案 =====

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math

def _pick_solid_font(preferred=None, size: int = 64):
    """
    尝试挑一个常见的“实心” TrueType 字体；找不到就退化到默认位图字体。
    """
    tried = []
    candidates = preferred or [
        # 常见跨平台字体（尽量实心）
        "NotoSansCJK-Regular.ttc",  # Google Noto CJK
        "NotoSans-Regular.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
        "SimHei.ttf",               # 黑体（中文环境）
        "msyh.ttc",                 # 微软雅黑
        "Helvetica.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception as e:
            tried.append((name, str(e)))
    # 全失败就默认
    return ImageFont.load_default()

def make_text_mask(
    N: int,
    text: str = "P",
    fontname: str =None,
    antialias: int = 4,         # 抗锯齿倍率：4 表示在 (4N×4N) 渲染后再下采样到 N×N
    bold: int = 0,              # 人为“加粗”像素（下采样前的粗糙膨胀像素数，0 表示不加粗）
    fill_holes: bool = True,    # 是否尝试“闭运算/填洞”，进一步实心
    threshold: float = 0.5      # 下采样后再阈值化到 [0,1]，0.5 一般合适
) -> torch.Tensor:
    """
    生成 NxN 的 **实心** 字符蒙版（[0,1]，1 为“笔画内部”，0 为背景）。
    - 保证实心：使用大分辨率渲染 + 可选加粗 + 可选闭运算（cv2） + 阈值化
    - 如果系统字体不可用，会自动回退

    返回：torch.FloatTensor，[N,N] in [0,1]
    """
    # 1) 超采样画布
    s = max(1, int(antialias))
    H = W = N * s
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    # 2) 选字体（或强制指定）
    fontsize = int(0.9 * min(W, H))
    if fontname:
        try:
            font = ImageFont.truetype(fontname, fontsize)
        except:
            font = _pick_solid_font(size=fontsize)
    else:
        font = _pick_solid_font(size=fontsize)

    # 3) 自适应缩放到 ~85% 画布
    bbox = draw.textbbox((0, 0), text, font=font)
    w0, h0 = bbox[2] - bbox[0], bbox[3] - bbox[1]
    scale = min((W * 0.85) / max(1, w0), (H * 0.85) / max(1, h0))
    fontsize2 = max(1, int(fontsize * scale))
    try:
        font = ImageFont.truetype(font.path if hasattr(font, "path") else fontname or "Arial.ttf", fontsize2)
    except:
        font = _pick_solid_font(size=fontsize2)

    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - w) // 2 - bbox[0]
    y = (H - h) // 2 - bbox[1]

    # 4) 关键点：**填充**文本（fill=255），不要只描边
    #    不设置 stroke_width/stroke_fill，避免“空心字”
    draw.text((x, y), text, fill=255, font=font)

    # 5) 可选“加粗”：简单方式——对高分辨率蒙版做边缘扩张
    if bold > 0:
        from PIL import ImageFilter
        # 用最大值滤波近似膨胀（kernel 大小 ≈ 2*bold+1）
        img = img.filter(ImageFilter.MaxFilter(size=2 * int(bold) + 1))

    # 6) 可选“闭运算/填洞”：OpenCV 更干净；没有 cv2 就跳过
    if fill_holes:
        try:
            import cv2
            arr_hr = np.array(img, dtype=np.uint8)
            # 先二值化（高分辨率）
            _, bw = cv2.threshold(arr_hr, 127, 255, cv2.THRESH_BINARY)
            # 形态学闭运算，填小孔洞
            k = max(1, int(0.01 * min(H, W)))   # 核大小~1% 画幅
            kernel = np.ones((k, k), np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
            img = Image.fromarray(bw, mode="L")
        except Exception:
            # 无 OpenCV：保持原状
            pass

    # 7) 下采样回 N×N（抗锯齿）
    img_small = img.resize((N, N), resample=Image.LANCZOS)

    # 8) 归一化 + 阈值（确保边缘干净、“实心”）
    arr = np.asarray(img_small, dtype=np.float32) / 255.0
    mask = (arr >= float(threshold)).astype(np.float32)

    return torch.from_numpy(mask)  # [N,N] in [0,1]



# ===== 新增：对“目标复场 U(x,y)”做单片 DPH（双相位互补棋盘）编码 =====
def dph_encode_from_complex(
    U_target: torch.Tensor,         # [B,N,N] 或 [1,N,N] 或 [N,N] 复数目标场
    tile: int = 1,                  # 棋盘超像素
    B: float = 0.5,                 # DPH 的 B 常数，Amax=1 时 B=1/2 最优
    to01: bool = True,              # 输出映射到 [0,1]
    clamp_amp: bool = True          # 幅度>1 时是否裁剪
):
    """
    经典 DPH：将目标复场 U = A·exp(iφ) 编码到单片相位 SLM。
    公式：
        A = |U|/Amax，φ = arg(U)，θ± = φ ± arccos(A)
        α = M1·θ+ + M2·θ-，phase01 = (α mod 2π)/(2π)

    返回：
        phase01: [B,N,N] or [1,N,N] in [0,1]
        B_used : torch.Tensor 常数 B
    """
    import math
    if U_target.ndim == 2:
        U_target = U_target.unsqueeze(0)
    if not U_target.is_complex():
        raise ValueError("U_target must be a complex tensor, e.g., dtype=torch.complex64.")

    Bsz, N, _ = U_target.shape
    device = U_target.device

    # --- 振幅与相位（实数）---
    A   = torch.abs(U_target)                              # [B,N,N] real
    Amax = torch.amax(A).clamp_min(1e-12)
    A_n = (A / Amax).clamp(0.0, 1.0) if clamp_amp else (A / Amax)
    phi = torch.angle(U_target)                            # [B,N,N] real in [-π, π]

    # θ±（全是实数）
    alpha_mag  = torch.acos(A_n)                           # [B,N,N] real in [0, π/2]
    theta_plus = phi + alpha_mag
    theta_minus= phi - alpha_mag

    # --- 棋盘互补复用（保持实数 dtype！）---
    # 用 theta_plus 的 dtype 作为“实数 dtype”
    dtype_real = theta_plus.dtype
    M1, M2 = build_checkerboard(N, tile=tile, device=device, dtype=dtype_real)  # [N,N] real
    M1 = M1.expand(Bsz, -1, -1)                           # [B,N,N]
    M2 = M2.expand(Bsz, -1, -1)                           # [B,N,N]

    # α 仍为实数
    alpha = M1 * theta_plus + M2 * theta_minus            # [B,N,N] real

    # --- 相位归一化 ---
    twopi = 2.0 * math.pi
    if to01:
        # remainder 需要实数 dtype；此时 alpha 是实数，安全
        phase01 = torch.remainder(alpha, twopi) / twopi   # [0,1]
    else:
        phase01 = alpha / twopi                           # 可能超出 [0,1]

    B_used = torch.full((Bsz,), float(B), device=device, dtype=dtype_real)
    return phase01, B_used


# ========== 完整替换版：dpm_verify（能指定目标“幅度图案+相位图案”） ==========
def dpm_verify():
    import matplotlib.pyplot as plt
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    N       = 160
    lam     = 634e-9
    dx      = 8e-6
    f       = 100e-3
    dtype_c = torch.complex64
    dtype   = torch.float32

    # 傅里叶面低通（以 cycles/pixel 给出）；tile=1 时 0.35~0.5 合理
    lp_cut  = 0.350

    # 1) 幅度 A01（“6”）
    six = make_digit_six(N).to(device=device, dtype=dtype)  # [N,N], 0..1
    A01 = six.unsqueeze(0)  # [1,N,N]

    # 2) 相位 φ（例如“W”）
    phase_mask01 = make_text_mask(N, text="W").to(device)  # [N,N] in [0,1]
    phi = (2 * np.pi) * phase_mask01.unsqueeze(0).to(dtype)  # [1,N,N] radians

    # ---- 并排显示 ----
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Target amplitude and phase patterns", fontsize=13, fontweight='bold')

    im0 = axs[0].imshow(A01[0].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Amplitude (pattern '6')")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)

    im1 = axs[1].imshow(phi[0].detach().cpu(), cmap='twilight', vmin=0, vmax=2 * np.pi)
    axs[1].set_title("Phase (pattern 'W')")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)

    for ax in axs:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()

    # 3) 组成目标复场 U = A·exp(iφ)
    U_target = (A01 * torch.exp(1j * phi)).to(torch.complex64)  # [1,N,N] complex

    U_target = (A01 * torch.exp(1j * phi)).to(torch.complex64)  # [1,N,N] complex

    # 4) DPH 单片编码（互补棋盘）
    phase01, _ = dph_encode_from_complex(U_target, tile=1, B=0.5, to01=True, clamp_amp=True)
    phase01 = phase01.to(device)  # [1,N,N] in [0,1]

    # 5) 构建 ASM：1 mm 自由空间 + f 透镜（4F = 两次 asm_f）
    asm_1mm = BandlimitedASM(wavelength=lam, N_size=N, pixel_size=dx,
                             zero_fill=1, is_lens=False, distance=1e-3,
                             sampling="midpoint", dtype=dtype_c, device=device, verbose=False)
    asm_f   = BandlimitedASM(wavelength=lam, N_size=N, pixel_size=dx,
                             zero_fill=1, is_lens=True,  distance=f,
                             sampling="midpoint", dtype=dtype_c, device=device, verbose=False)

    # 6) 输入复场：均匀振幅 + DPH 相位（注意：phase01 已是 [1,N,N]，不要再 unsqueeze）
    Ein = torch.exp(-1j * (2*math.pi) * phase01).to(dtype_c)     # [1,N,N]

    # ---------------- Fig.1：输入与 1mm 传播 ----------------
    E_1mm   = asm_1mm(Ein)  # [1,N,N]
    amp_in  = to_np2d(torch.abs(Ein))
    pha_in  = to_np2d(torch.remainder(torch.angle(Ein), 2*np.pi))
    amp_1mm = to_np2d(torch.abs(E_1mm))
    pha_1mm = to_np2d(torch.remainder(torch.angle(E_1mm), 2*np.pi))

    imshow2x2(
        figtitle="(Fig.1) Input & free-space (1 mm) propagation",
        images=[amp_in, pha_in, amp_1mm, pha_1mm],
        titles=["Input Amplitude (Uniform)",
                "Input Phase (DPH encoded)",
                "After 1 mm: Amplitude",
                "After 1 mm: Phase"],
        cmaps =['gray', 'twilight', 'gray', 'twilight']
    )

    # ---------------- Fig.2：傅里叶面（Emiddle）低通前/后 ----------------
    # 按你的约定，第一次 asm_f(E) 的输出就是“傅里叶面”
    Emiddle = asm_f(Ein)  # [1,N,N]
    # 低通掩膜先用 float 生成，再匹配类型，避免复数 mask 造成可视化困惑
    LPmask_f32 = make_lowpass_mask_spatial(N, lp_cut, device=Emiddle.device, dtype=torch.float32)  # [N,N] float
    LPmask     = LPmask_f32.to(Emiddle.dtype)  # complex64 for multiplication
    print(f"LPmask.shape={LPmask.shape}, LPmask.dtype={LPmask.dtype}")

    # 可视化低通孔径（用实部）
    plt.figure()
    plt.imshow(LPmask_f32.detach().cpu(), cmap="gray", vmin=0, vmax=1)
    plt.title("Fourier-plane low-pass aperture (LPmask)")
    plt.tight_layout()

    Emiddle_lp = Emiddle * LPmask  # [1,N,N]

    amp_mid    = to_np2d(torch.abs(Emiddle))
    pha_mid    = to_np2d(torch.remainder(torch.angle(Emiddle), 2*np.pi))
    amp_mid_lp = to_np2d(torch.abs(Emiddle_lp))
    pha_mid_lp = to_np2d(torch.remainder(torch.angle(Emiddle_lp), 2*np.pi))

    imshow2x2(
        figtitle=f"(Fig.2) Fourier plane LP filtering (cutoff={lp_cut:.3f} cyc/pix)",
        images=[amp_mid, pha_mid, amp_mid_lp, pha_mid_lp],
        titles=["Fourier plane: Amplitude (before LP)",
                "Fourier plane: Phase (before LP)",
                "Fourier plane: Amplitude (after LP)",
                "Fourier plane: Phase (after LP)"],
        cmaps=['viridis', 'twilight', 'viridis', 'twilight']
    )

    # ---------------- Fig.3：像面重建（第二次 asm_f） ----------------
    Eimg    = asm_f(Emiddle)       # [1,N,N]
    Eimg_lp = asm_f(Emiddle_lp)    # [1,N,N]

    amp_img    = to_np2d(torch.abs(Eimg))
    pha_img    = to_np2d(torch.remainder(torch.angle(Eimg), 2*np.pi))
    amp_img_lp = to_np2d(torch.abs(Eimg_lp))
    pha_img_lp = to_np2d(torch.remainder(torch.angle(Eimg_lp), 2*np.pi))

    imshow2x2(
        figtitle="(Fig.3) Image plane reconstruction (4F)",
        images=[amp_img, pha_img, amp_img_lp, pha_img_lp],
        titles=["Image plane: Amplitude (no LP)",
                "Image plane: Phase (no LP)",
                "Image plane: Amplitude (with LP)",
                "Image plane: Phase (with LP)"],
        cmaps=['gray', 'twilight', 'gray', 'twilight']
    )

    # ---------------- 定量核验：与目标对比（注意：4F 正像 or 倒像） ----------------
    # 你的 4F 设置若是“正像”，直接与 A01/phi 比；若为“倒像”，对最后两个维度翻转比较。
    # 这里给出“倒像”比较：
    A_ref_flip  = torch.flip(A01,  dims=[-2, -1])[0].detach().cpu().numpy()        # [N,N], amplitude
    P_ref_flip  = torch.flip(phi,   dims=[-2, -1])[0].detach().cpu().numpy()       # [N,N], radians

    def corr2(a, b, eps=1e-12):
        a = a - a.mean(); b = b - b.mean()
        return float((a*b).sum() / (np.sqrt((a*a).sum())*np.sqrt((b*b).sum()) + eps))

    # 幅度相关（with / without LP）
    corr_amp_lp = corr2(amp_img_lp, A_ref_flip)
    corr_amp    = corr2(amp_img,    A_ref_flip)

    # 相位相关（wrap 到 [-π,π) 再比较；或换成 circular correlation）
    def wrap_pm_pi(x):
        return (x + np.pi) % (2*np.pi) - np.pi
    pha_img_lp_wrapped = wrap_pm_pi(pha_img_lp)
    P_ref_flip_wrapped = wrap_pm_pi(P_ref_flip)
    corr_phase_lp = corr2(pha_img_lp_wrapped, P_ref_flip_wrapped)
    corr_phase    = corr2(wrap_pm_pi(pha_img), P_ref_flip_wrapped)

    print(f"[Check] amplitude corr (LP)    vs flipped target = {corr_amp_lp:.4f}")
    print(f"[Check] amplitude corr (no LP) vs flipped target = {corr_amp:.4f}")
    print(f"[Check] phase     corr (LP)    vs flipped target = {corr_phase_lp:.4f}")
    print(f"[Check] phase     corr (no LP) vs flipped target = {corr_phase:.4f}")

    plt.show()



def build_input_amplitude_columnwise(spins_total: int,
                                     pixel_repeat: int,
                                     *,
                                     device=None,
                                     dtype=torch.float32,
                                     return_batch: bool = False):
    """
    生成自适应的入射振幅张量 input_amp（列优先填充：先填满第一列，再第二列）。

    参数
    ----
    spins_total   : 自旋总数（>=1）
    pixel_repeat  : 每个自旋对应的“巨像素”边长（>=1）
    device        : torch 设备（默认自动）
    dtype         : torch dtype（默认 float32）
    return_batch  : True 则返回 [1, H, W]；False 则返回 [H, W]

    规则
    ----
    - 计算块网格边长：G = ceil(sqrt(spins_total))
    - 输出尺寸：H = W = G * pixel_repeat
    - 前 spins_total 个块置为 1，每块 pixel_repeat×pixel_repeat
    - 布局顺序：列优先（column-major）
        即第 k 个自旋块的位置为：
            (row_block, col_block) = (k % G, k // G)
    """
    if spins_total <= 0:
        raise ValueError("spins_total must be >= 1")
    if pixel_repeat <= 0:
        raise ValueError("pixel_repeat must be >= 1")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = int(math.ceil(math.sqrt(spins_total)))
    H = W = G * pixel_repeat

    amp = torch.zeros((H, W), dtype=dtype, device=device)

    for k in range(spins_total):
        row_block = k % G         # 列优先
        col_block = k // G
        r0 = row_block * pixel_repeat
        c0 = col_block * pixel_repeat
        amp[r0:r0 + pixel_repeat, c0:c0 + pixel_repeat] = 1.0

    if return_batch:
        return amp.unsqueeze(0)
    return amp


def spins_vector_to_block_image(
    spins_state: torch.Tensor,
    pixel_repeat: int,
    *,
    pad_value: float = 0.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_batch: bool = False,
    column_major: bool = True,
) -> torch.Tensor:
    """
    把一维自旋状态 [S] 映射为由巨像素 (pixel_repeat×pixel_repeat) 组成的正方形阵列。
    - 列优先 (column_major=True)：先填满第0列再第1列...（你要求的逻辑）
      第 k 个自旋块位置 = (row_block, col_block) = (k % G, k // G)
    - 多余块用 pad_value 填充（默认 0）
    - 输出尺寸为 (G*pixel_repeat) × (G*pixel_repeat)，其中 G = ceil(sqrt(S))

    参数
    ----
    spins_state : torch.Tensor, 形状 [S]，元素可为 {0,1} 或 {-1,+1} 或实数
    pixel_repeat: int >= 1，每个自旋对应的巨像素块边长
    pad_value   : float，多余空块填充值（默认 0.0）
    dtype       : 输出 dtype（默认保持与 spins_state 相同的浮点类型；若为整型则用 float32）
    device      : 输出张量设备（默认与 spins_state 一致）
    return_batch: True 返回 [1,H,W]，False 返回 [H,W]
    column_major: True 列优先；False 行优先

    返回
    ----
    img : torch.Tensor，形状 [H,W] 或 [1,H,W]，H=W=G*pixel_repeat
    """
    # ---------- 基本检查 ----------
    if spins_state.ndim != 1:
        raise ValueError(f"spins_state must be 1D, got shape {tuple(spins_state.shape)}")
    S = int(spins_state.numel())
    if S <= 0:
        raise ValueError("spins_state must have at least one element")
    if not isinstance(pixel_repeat, int) or pixel_repeat <= 0:
        raise ValueError("pixel_repeat must be a positive integer")

    # ---------- 设备 / 类型 ----------
    if device is None:
        device = spins_state.device
    else:
        device = torch.device(device)

    # 确定输出 dtype：若未指定且源为非浮点，则使用 float32
    if dtype is None:
        dtype = spins_state.dtype if spins_state.dtype.is_floating_point else torch.float32

    # 源向量搬到目标设备
    x = spins_state.to(device=device, dtype=dtype)  # [S] on device

    # ---------- 计算网格边长与输出尺寸 ----------
    G = int(math.ceil(math.sqrt(S)))        # 块网格边长
    H = W = G * pixel_repeat                # 输出像素尺寸

    # ---------- 先构建块级网格 G×G ----------
    grid_blocks = torch.full((G, G), pad_value, dtype=dtype, device=device)  # [G,G]

    # 生成 0..S-1 的索引，并计算每个自旋块应放置的 (row_block, col_block)
    k = torch.arange(S, device=device)      # [S]
    if column_major:
        row_block = torch.remainder(k, G)   # k % G
        col_block = torch.div(k, G, rounding_mode="floor")  # k // G
    else:
        row_block = torch.div(k, G, rounding_mode="floor")  # k // G
        col_block = torch.remainder(k, G)   # k % G

    # 将前 S 个块位置写入（索引均为 int64/long）
    grid_blocks[row_block.long(), col_block.long()] = x

    # ---------- 把每个块扩展为 pixel_repeat×pixel_repeat 巨像素 ----------
    img = grid_blocks.repeat_interleave(pixel_repeat, dim=0).repeat_interleave(pixel_repeat, dim=1)  # [H,W]

    return img.unsqueeze(0) if return_batch else img

# ---------------------------------------- minimal self-test ----------------------------------------
if __name__ == "__main__":
    # demo 1: pack + giant pixels
    S, r = 197, 4
    vec = 2.0 * (np.random.rand(S, r) - 0.5)  # [-1,1]
    grids = pack_vectors_to_square(vec, return_torch=True, device="cpu", dtype=torch.float32, pixel_repeat=16)
    print("packed grids:", grids.shape)  # e.g. [4, 240, 240] because N=15 and repeat=16 → 15*16=240

    # demo 2: DPH on real-valued grids (Amax=1 → B=1/2)
    phase01, Bc = dph_encode_real_single_slm(grids, tile=1, to01=True, clamp=True)
    print("phase01:", phase01.shape, phase01.min().item(), phase01.max().item(), "B =", Bc)

    # demo 3: constant checks (your two points)
    for const in [1.0, 0.0, -1.0]:
        N = 15
        xi = torch.full((N, N), const, dtype=torch.float32)
        ph01, B = dph_encode_real_single_slm(xi, tile=1)
        uvals = torch.unique(ph01)
        print(f"ξ={const:+.1f}: unique phase01 values ~", uvals.cpu().numpy()[:8], "; B=", B.item())
        # Expect: ξ=+1 → two values near 1/6 and 5/6; ξ=0 → near 1/4 and 3/4; ξ=-1 → near 1/6 and 5/6 but with φ=π already handled

    phase01, Bc = dph_encode_real_single_slm(torch.ones((1,2,2)), tile=1, to01=True, clamp=True)
    print(phase01)
    phase01, Bc = dph_encode_real_single_slm(0.5*torch.ones((1,2, 2)), tile=1, to01=True, clamp=True)
    print(phase01)

    asm = BandlimitedASM(wavelength=634e-9, N_size=160, pixel_size=8e-6,
                         zero_fill=1, is_lens=True, distance=100e-3,
                         sampling="midpoint", dtype=torch.complex64, device='cuda', verbose=True)

    # main()
    dpm_verify()

