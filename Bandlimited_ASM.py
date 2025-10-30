# -*- coding: utf-8 -*-
"""
Bandlimited Angular Spectrum Method (ASM) with:
- Evanescent-wave removal (propagation disk)
- Distance & aperture dependent band-limit window (∩ Nyquist ∩ 1/λ)
- Mid-cell (midpoint) frequency sampling
- Optional zero-fill padding
- Two modes: free-space (is_lens=False) and 2F lens system (is_lens=True)

Validation in main(): four scenarios with 2x2 subplots per figure.
"""
# 基于 torch.nn.Module 的一个衍射层，角谱传播，去除了消失波/倏逝波的影响，以及考虑了带限角谱（带限的本质是两个平行平面之间，太过于倾斜/光角谱分量过大的光是传递不到第二个平面的）
# 同时，倏逝波由于指数衰减，很容易排除其影响，但是带限角谱较难计算，在下面的类初始化中，需要考虑补零后的self.kz矩阵在经过两次掩膜后的非零的占比
# 即计算和print出经过两次符合物理原理的掩膜作用后self.kz中非0值元素占总元素的占比，让我们直观看到物理限制对self.kz矩阵的影响
# batch = rank(J)；height*width 其实不直接等于spins的个数，因为双相位编码 DPM 就扩到了 (2*2)*N 个像素，
# DMP-Spin 是 2*2 个像素,对角线的两个像素相等，类似[[A,B],[B,A]],干涉形成真正的Spin的复振幅
# 因然后每个 DMP-Spin 像素再在两个维度上扩充 4 倍到 8*8 的大小；因此每个 macropixel 中有16份 DMP-Spin,
# 每个 spin-macropixel 的大小的两个维度是 16*16；假如 SLM 面利用 400*400 个像素，则一共是可以加载625个自旋，已经很多了
# 但张量运算，应该也用不到那么多
# 形状：[batch, height, width], 现在是伊辛耦合矩阵降维分解，因此可以把多个张量进行分解，将 rank(J) 个耦合矩阵双相位编码到 SLM 的自旋上
# 然后写入 batch，经过 2F 系统后，将 batch 个张量一起传递到后焦面上
# 即计算思路为：从空域到频率域的傅里叶变换——频率域角谱传播——从频率域到空间域的逆傅里叶变换
# 下面这个类初始化时需要传递进来的参数是：波长 λ 正方形区域的x或者y方向的尺寸N_size 像素尺寸pixel_size
# 补零zero_fill(一般至少为1倍，例如当补零为1倍的时候，创建一个2倍N_size尺寸的正方形区域将原来N_size*N_size大小的原复振幅放在最中间！补零的意义是防止频谱混叠)
# 判断参数：is_lens=True or False 当 is_lens=False的时候，进入的逻辑循环是传递进来的参数 distance 的单位是毫米mm，即只衍射传播一段距离就ok
# 当is_lens=True的时候，进入的逻辑循环是传递进来的参数 distance 被当作是透镜的焦距，即发生的过程是：从起始平面的复振幅进行傅里叶变换后，角谱传播distance距离后，逆傅里叶变换，叠加上透镜的焦距（相位板的参数矩阵很容易算出来），然后再傅里叶变换，然后角谱传播distance距离后进行逆傅里叶变换回到空间域
# 即is_lens=True的时候，进行的是一个典型的 2F 透镜系统的傅里叶变换过程。
# 综上，该类初始化的时候，需要传递进来的参数有 (λ=634.0e-9,N_size=400,pixel_size=8e-6,zero_fill=1,is_lens=True,distance=100e-3)
# 在初始化好该类后，每次调用，只要传递进来复振幅图像就行，输入数据的三个维度分别是 批次*图像张量矩阵的上下高度方向*图像张量矩阵的左右宽度方向

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ---------------------------- utility: center embed / crop ----------------------------
def _center_embed(E: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
    """Center-embed E[..., H, W] into a larger zero array E[..., H_out, W_out]."""
    *head, H, W = E.shape
    out = E.new_zeros(*head, H_out, W_out)
    y0 = (H_out - H) // 2
    x0 = (W_out - W) // 2
    out[..., y0:y0 + H, x0:x0 + W] = E
    return out

def _center_crop(X: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Center-crop X[..., H_big, W_big] to X[..., H, W]."""
    *head, H_big, W_big = X.shape
    y0 = (H_big - H) // 2
    x0 = (W_big - W) // 2
    return X[..., y0:y0 + H, x0:x0 + W]


# ------------------------------------- ASM core --------------------------------------
class BandlimitedASM(torch.nn.Module):
    """
    Angular Spectrum Method with:
      - evanescent cut (propagating region disk)
      - band-limit window depending on z and zero-filled aperture size
      - midpoint frequency sampling (bin centers)
      - optional zero-fill (total grid size P = (1 + zero_fill) * N)
      - two modes: free-space (is_lens=False) and 2F lens system (is_lens=True)

    Parameters
    ----------
    wavelength : float
        Wavelength λ [m]
    N_size : int
        Original square grid size N (input must be [B, N, N])
    pixel_size : float
        Pixel pitch Δx = Δy [m]
    zero_fill : int
        k >= 0. Total working grid becomes P = (1 + k) * N
    is_lens : bool
        False: free-space propagation; True: 2F lens chain
    distance : float
        If is_lens=False: propagation distance z [m]
        If is_lens=True : focal length f=z and each propagation segment uses z
    sampling : 'midpoint' | 'fft'
        'midpoint' uses mid-cell sampling (recommended); 'fft' follows fftfreq
    dtype : torch dtype
        Complex dtype (e.g., torch.complex64)
    device : str or torch.device
    verbose : bool
        Print nonzero ratios after masks for sanity-checking
    """

    def __init__(
        self,
        wavelength: float = 634e-9,
        N_size: int = 256,
        pixel_size: float = 8e-6,
        zero_fill: int = 1,
        is_lens: bool = False,
        distance: float = 100e-3,
        sampling: str = "midpoint",
        dtype: torch.dtype = torch.complex64,
        device=None,
        verbose: bool = True,
    ):
        super().__init__()
        assert zero_fill >= 0 and int(zero_fill) == zero_fill
        self.lam = float(wavelength)
        self.N = int(N_size)
        self.dx = float(pixel_size)
        self.dy = float(pixel_size)
        self.zero_fill = int(zero_fill)
        self.is_lens = bool(is_lens)
        self.z = float(distance)
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # working grid size and physical aperture after zero-fill
        self.P = (1 + self.zero_fill) * self.N
        self.Lx = self.P * self.dx
        self.Ly = self.P * self.dy

        # --------- frequency sampling (cycles/m) ---------
        if sampling not in ("midpoint", "fft"):
            raise ValueError("sampling must be 'midpoint' or 'fft'")
        self.sampling = sampling

        if self.sampling == "fft":
            fx = torch.fft.fftshift(torch.fft.fftfreq(self.P, d=self.dx), dim=0)
            fy = torch.fft.fftshift(torch.fft.fftfreq(self.P, d=self.dy), dim=0)
        else:
            # midpoint rule: f[k] = (k - P/2 + 0.5) * (1/L), already centered
            dfx = 1.0 / self.Lx
            dfy = 1.0 / self.Ly
            kx = torch.arange(self.P, device=self.device)
            ky = torch.arange(self.P, device=self.device)
            fx = (kx - self.P / 2 + 0.5) * dfx
            fy = (ky - self.P / 2 + 0.5) * dfy

        FX, FY = torch.meshgrid(fx.to(self.device), fy.to(self.device), indexing="ij")

        # --------- evanescent cut (propagating region) ---------
        k = 2 * math.pi / self.lam
        rho2 = (self.lam * FX) ** 2 + (self.lam * FY) ** 2       # (λ fx)^2 + (λ fy)^2
        prop_mask = (rho2 <= 1.0)
        kz = k * torch.sqrt(torch.clamp(1.0 - rho2, min=0.0))
        H = torch.exp(1j * kz * self.z)
        H = torch.where(prop_mask, H, torch.zeros_like(H))

        # --------- band-limit window (z & L dependent) ---------
        fx_max = 1.0 / (self.lam * math.sqrt(1.0 + (2.0 * self.z / self.Lx) ** 2))
        fy_max = 1.0 / (self.lam * math.sqrt(1.0 + (2.0 * self.z / self.Ly) ** 2))
        f_nx = 0.5 / self.dx
        f_ny = 0.5 / self.dy
        f_cut = min(fx_max, fy_max, f_nx, f_ny, 1.0 / self.lam)  # conservative
        band_mask = (FX ** 2 + FY ** 2) <= (f_cut ** 2)
        H = torch.where(band_mask, H, torch.zeros_like(H))

        # --------- print non-zero ratios (sanity) ---------
        if self.verbose:
            nnz_all = self.P * self.P
            nnz_prop = int(prop_mask.sum().item())
            nnz_band = int((prop_mask & band_mask).sum().item())
            print(f"[ASM] grid P={self.P}×{self.P}, L=({self.Lx:.3e},{self.Ly:.3e}) m, z={self.z:.3e} m")
            print(f"[ASM] propagating ratio   : {nnz_prop}/{nnz_all} = {nnz_prop/nnz_all:.4f}")
            print(f"[ASM] after band-limit    : {nnz_band}/{nnz_all} = {nnz_band/nnz_all:.4f}")
            print(f"[ASM] f_cut={f_cut:.3e}")

        self.register_buffer("H", H.to(self.dtype))

        # space coordinates for lens phase & plotting scales
        x = (torch.arange(self.P, device=self.device) - self.P // 2) * self.dx
        y = (torch.arange(self.P, device=self.device) - self.P // 2) * self.dy
        XX, YY = torch.meshgrid(x, y, indexing="ij")
        self.register_buffer("XX", XX)
        self.register_buffer("YY", YY)

        # thin-lens phase (used only when is_lens=True)
        if self.is_lens:
            f = self.z  # in this API, distance also serves as focal length for 2F mode
            lens_phase = torch.exp(-1j * (k / (2.0 * f)) * (self.XX ** 2 + self.YY ** 2))
            self.register_buffer("lens_phase", lens_phase.to(self.dtype))

    # ----------------------------- forward propagation -----------------------------
    def forward(self, Ein: torch.Tensor) -> torch.Tensor:
        """
        Ein: [B, N, N] complex tensor
        Returns: [B, N, N] complex tensor
        """
        assert Ein.dtype.is_complex, "Ein must be complex (e.g., complex64)."
        B, H, W = Ein.shape
        assert H == self.N and W == self.N, "Ein must be [B, N, N]"

        # zero-fill by center-embedding to the working grid P×P
        E = _center_embed(Ein, self.P, self.P)

        if not self.is_lens:
            # free-space: FFT -> multiply H -> IFFT -> crop
            Fk = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
            Fk = Fk * self.H
            out = torch.fft.ifft2(torch.fft.ifftshift(Fk, dim=(-2, -1)))
            out = _center_crop(out, self.N, self.N)
            return out.to(self.dtype)

        else:
            # 2F chain (as requested):
            # 1) FFT
            Fk1 = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
            # 2) propagate z
            Fk1 = Fk1 * self.H
            # 3) back to space
            E1 = torch.fft.ifft2(torch.fft.ifftshift(Fk1, dim=(-2, -1)))
            # 4) thin lens phase
            E2 = E1 * self.lens_phase
            # 5) FFT
            Fk2 = torch.fft.fftshift(torch.fft.fft2(E2), dim=(-2, -1))
            # 6) propagate z
            Fk2 = Fk2 * self.H
            # 7) back to space
            Eout = torch.fft.ifft2(torch.fft.ifftshift(Fk2, dim=(-2, -1)))
            Eout = _center_crop(Eout, self.N, self.N)
            return Eout.to(self.dtype)


# -------------------------------- helper: lens phase --------------------------------
def make_lens_phase(N: int, dx: float, wavelength: float, f: float,
                    device, dtype=torch.complex64) -> torch.Tensor:
    """Generate thin-lens quadratic phase exp{-i k (x^2+y^2)/(2f)} on an N×N grid."""
    k = 2 * math.pi / wavelength
    x = (torch.arange(N, device=device) - N // 2) * dx
    y = (torch.arange(N, device=device) - N // 2) * dx
    XX, YY = torch.meshgrid(x, y, indexing="ij")
    phase = torch.exp(-1j * (k / (2.0 * f)) * (XX ** 2 + YY ** 2))
    return phase.to(dtype)


# ------------------------------ plotting: 2x2 subplots ------------------------------
def visualize_io_2x2(title: str, Ein: torch.Tensor, Eout: torch.Tensor):
    """Draw a single figure with 2x2 subplots: input amp/phase, output amp/phase."""
    assert Ein.ndim == 3 and Eout.ndim == 3
    amp_in = Ein.abs()[0].cpu().numpy()
    pha_in = torch.angle(Ein)[0].cpu().numpy()
    amp_out = Eout.abs()[0].cpu().numpy()
    pha_out = torch.angle(Eout)[0].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    im0 = axes[0, 0].imshow(amp_in);  axes[0, 0].set_title("Input Amplitude");  fig.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].imshow(pha_in);  axes[0, 1].set_title("Input Phase (rad)");fig.colorbar(im1, ax=axes[0, 1])
    im2 = axes[1, 0].imshow(amp_out); axes[1, 0].set_title("Output Amplitude"); fig.colorbar(im2, ax=axes[1, 0])
    im3 = axes[1, 1].imshow(pha_out); axes[1, 1].set_title("Output Phase (rad)");fig.colorbar(im3, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.grid(color="w", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")

    fig.suptitle(title, fontsize=13)
    plt.show()


# ----------------------------------------- main -------------------------------------
def main():

    # Common defaults (you可按需改大 N 或 zero_fill 以获得更细致的图样)
    lam = 634e-9      # wavelength [m]
    N   = 400         # input grid size
    dx  = 8e-6        # pixel size [m]
    z1  = 100e-3      # 100 mm
    z2  = 200e-3      # 200 mm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.complex64
    torch.manual_seed(0)


    
    # ========== Case 1: free-space, z=100 mm, input = unit amplitude + lens phase f=100mm ==========
    Ein1 = torch.ones(1, N, N, dtype=dtype, device=device) * make_lens_phase(N, dx, lam, z1, device, dtype)
    asm1 = BandlimitedASM(wavelength=lam, N_size=N, pixel_size=dx,
                          zero_fill=1, is_lens=False, distance=z1,
                          sampling="midpoint", dtype=dtype, device=device, verbose=True)
    Eout1 = asm1(Ein1)
    visualize_io_2x2(f"Case 1: Free-space, z={z1*1e3:.0f} mm; Input has lens phase f=100 mm",
                     Ein1, Eout1)

    # ========== Case 2: free-space, z=200 mm, input = unit amplitude + lens phase f=200mm ==========
    Ein2 = torch.ones(1, N, N, dtype=dtype, device=device) * make_lens_phase(N, dx, lam, z1, device, dtype)
    asm2 = BandlimitedASM(wavelength=lam, N_size=N, pixel_size=dx,
                          zero_fill=1, is_lens=False, distance=z2,
                          sampling="midpoint", dtype=dtype, device=device, verbose=True)
    Eout2 = asm2(Ein2)
    visualize_io_2x2(f"Case 2: Free-space, z={z2*1e3:.0f} mm; Input has lens phase f=100 mm",
                     Ein2, Eout2)

    # ========== Case 3: 2F lens, f=z=100 mm, input = delta-like point near center ==========
    Ein3 = torch.zeros(1, N, N, dtype=dtype, device=device)
    c = N // 2
    Ein3[0, c-1:c+2, c-1:c+2] = 1.0 + 0j   # small dot (3×3)
    asm3 = BandlimitedASM(wavelength=lam, N_size=N, pixel_size=dx,
                          zero_fill=1, is_lens=True, distance=z1,
                          sampling="midpoint", dtype=dtype, device=device, verbose=True)
    Eout3 = asm3(Ein3)
    visualize_io_2x2(f"Case 3: 2F lens, f={z1*1e3:.0f} mm; Input is a small point source",
                     Ein3, Eout3)

    # ========== Case 4: 2F lens, f=z=100 mm, input = plane wave (unit amplitude, zero phase) ==========
    Ein4 = torch.ones(1, N, N, dtype=dtype, device=device)  # plane wave
    asm4 = BandlimitedASM(wavelength=lam, N_size=N, pixel_size=dx,
                          zero_fill=1, is_lens=True, distance=z1,
                          sampling="midpoint", dtype=dtype, device=device, verbose=True)
    Eout4 = asm4(Ein4)
    visualize_io_2x2(f"Case 4: 2F lens, f={z1*1e3:.0f} mm; Input is a plane wave (Airy-like focus expected)",
                     Ein4, Eout4)
    


    # 模拟 100 个 spin 的情况，张量只占 约 0.02 GB（20 MB） 显存，对现代 GPU 几乎可以忽略.只做推理（with torch.no_grad()）时约 1.1 GB
    real = torch.randn(100, 160, 160, device=device)
    imag = torch.randn(100, 160, 160, device=device)
    Ein5 = torch.complex(real, imag).to("cuda")  # 自动 dtype=torch.complex64
    print(Ein5.dtype,Ein5.shape)
    asm5 = BandlimitedASM(wavelength=lam, N_size=160, pixel_size=dx,
                          zero_fill=1, is_lens=True, distance=z1,
                          sampling="midpoint", dtype=dtype, device=device, verbose=True)
    with torch.no_grad():
        Eout5 = asm5(Ein5)
        print(Eout5.dtype, Eout5.shape)

# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Verify whether this module is correct?
    main()
