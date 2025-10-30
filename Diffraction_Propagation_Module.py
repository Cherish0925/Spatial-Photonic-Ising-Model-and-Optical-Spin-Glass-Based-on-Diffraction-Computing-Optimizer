# 这个矩阵的作用是，比如输入一个 [batch,d=2,height,width] 一个 batch 的图像，首先将d=2这个维度认为是复振幅的实部与虚部
# 因此，将 [batch,d=2,height,width] 重新合成为一个复振幅张量矩阵？这个复振幅张量矩阵，直接在空间域中加上 Zernike 多项式的像差。
# 各阶像差的权重（肯定是大于等于0的）是可以优化的，因此称为 neural_diffraction
# 合成了像差（像差矩阵，相位弥补矩阵）的复振幅(在相位的指数部分加上相应的相位延迟)再次基于 fft 进行衍射传播，得到出射面的复振幅
# 模型具备补0的功能，二倍补0
import numpy as np
import torch
import torch.nn.functional as F
import math
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import random

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



def _center_embed(E, H_out, W_out):
    """把 E（[..., H, W]）居中镶嵌到更大的 [..., H_out, W_out]（其余补零）"""
    *head, H, W = E.shape
    out = E.new_zeros(*head, H_out, W_out)
    y0 = (H_out - H)//2
    x0 = (W_out - W)//2
    out[..., y0:y0+H, x0:x0+W] = E
    return out

def _center_crop(X, H, W):
    """从 [..., H_big, W_big] 的中心裁出 [..., H, W]"""
    *head, H_big, W_big = X.shape
    y0 = (H_big - H)//2
    x0 = (W_big - W)//2
    return X[..., y0:y0+H, x0:x0+W]

class BandlimitedASM(torch.nn.Module):
    """
    角谱法（ASM）传播层，集成：
      - 倏逝波屏蔽（传播圆盘）
      - 带限掩膜（与 z 和补零后 L 有关）∩ 奈奎斯特 ∩ 1/λ
      - 半格偏移（midpoint rule）或 'fft' 采样
      - 可选 k×补零（zero_fill=k => 总尺寸 = (1+k)*N）
      - is_lens=False: 单段传播
      - is_lens=True : 2F 透镜系统（按你的流程）

    参数
    ----
    wavelength: λ (m)
    N_size    : 原始方阵边长 N（输入 H=W=N）
    pixel_size: Δx=Δy (m)
    zero_fill : >=0 的整数；1 表示总尺寸变为 2N
    is_lens   : 是否走 2F 链路
    distance  : 当 is_lens=False 时是传播距离 z；当 is_lens=True 时作为焦距 f（每段传播距离仍用 z=distance）
    sampling  : 'midpoint'（半格偏移，默认）| 'fft'
    dtype     : 复数 dtype
    device    : 设备
    verbose   : 是否打印 kz 非零比例统计
    """
    def __init__(self,
                 wavelength=634e-9,
                 N_size=400,
                 pixel_size=8e-6,
                 zero_fill=1,
                 is_lens=True,
                 distance=100e-3,
                 sampling='midpoint',
                 dtype=torch.complex64,
                 device=None,
                 verbose=True):
        super().__init__()
        assert zero_fill >= 0 and int(zero_fill) == zero_fill
        self.lam = float(wavelength)
        self.N = int(N_size)
        self.dx = float(pixel_size)
        self.dy = float(pixel_size)
        self.zero_fill = int(zero_fill)
        self.is_lens = bool(is_lens)
        self.z = float(distance)  # 单段传播距离；is_lens=True 时也用该 z（每段）
        self.dtype = dtype
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # 几何：补零后总尺寸 P=(1+zero_fill)*N；物理尺寸 Lx=Px*dx, Ly=Py*dy
        self.P = (1 + self.zero_fill) * self.N
        self.Lx = self.P * self.dx
        self.Ly = self.P * self.dy

        # ---------- 频率采样（cycles/m） ----------
        if sampling not in ('midpoint', 'fft'):
            raise ValueError("sampling must be 'midpoint' or 'fft'")
        self.sampling = sampling

        # ----- frequency sampling (cycles/m), midpoint & already centered -----
        if sampling == 'midpoint':
            # total length after zero-fill
            # midpoint rule：bin 中心；Δf = 1/L
            # 把采样点放到每个频率小方块的中心（midpoint rule），更贴近把连续谱积分用矩形中点求积近似、同时避免在 Nyquist 边界取样造成的奇异/对称伪影
            Lx = self.Lx  # = P * dx
            Ly = self.Ly
            dfx = 1.0 / Lx
            dfy = 1.0 / Ly

            # k = 0..P-1  ->  f[k] = (k - P/2 + 0.5) * df  〈—— 半格偏移 + 居中
            kx = torch.arange(self.P, device=self.device)
            ky = torch.arange(self.P, device=self.device)
            fx = (kx - self.P / 2 + 0.5) * dfx
            fy = (ky - self.P / 2 + 0.5) * dfy
        elif sampling == 'fft':
            fx = torch.fft.fftfreq(self.P, d=self.dx)
            fy = torch.fft.fftfreq(self.P, d=self.dy)
            # 对应我们后面对频谱做 fftshift，这里把坐标也 shift 到中心，便于直观一致
            fx = torch.fft.fftshift(fx, dim=0)
            fy = torch.fft.fftshift(fy, dim=0)

        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        FX = FX.to(self.device)
        FY = FY.to(self.device)

        # ---------- 传播圆盘（倏逝波屏蔽） ----------
        k = 2*math.pi/self.lam
        rho2 = (self.lam*FX)**2 + (self.lam*FY)**2    # (λ fx)^2 + (λ fy)^2
        prop_mask = (rho2 <= 1.0)                      # 传播区域
        kz = k * torch.sqrt(torch.clamp(1.0 - rho2, min=0.0))
        H = torch.exp(1j * kz * self.z)
        H = torch.where(prop_mask, H, torch.zeros_like(H))  # 屏蔽倏逝波（硬切）

        # ---------- 带限掩膜（与 z 与 L 相关） ----------
        # 与你脚本一致的保守形式（z 越大，f_cut 越小）：
        # fx_max = 1 / ( λ * sqrt(1 + (2z/Lx)^2) ), fy_max 同理
        fx_max = 1.0 / ( self.lam * math.sqrt(1.0 + (2.0*self.z/self.Lx)**2) )
        fy_max = 1.0 / ( self.lam * math.sqrt(1.0 + (2.0*self.z/self.Ly)**2) )
        # 还要与奈奎斯特和 1/λ 共同取交集（保守的圆窗半径）
        f_nx = 0.5/self.dx
        f_ny = 0.5/self.dy
        f_cut = min(fx_max, fy_max, f_nx, f_ny, 1.0/self.lam)
        band_mask = (FX**2 + FY**2) <= (f_cut**2)
        H = torch.where(band_mask, H, torch.zeros_like(H))

        # ---------- 统计非零占比 ----------
        with torch.no_grad():
            nnz_all = self.P*self.P
            nnz_prop = int(prop_mask.sum().item())
            nnz_band = int((prop_mask & band_mask).sum().item())
            if self.verbose:
                print(f"[ASM] grid P={self.P}×{self.P}, L=({self.Lx:.3e},{self.Ly:.3e}) m, z={self.z:.3e} m")
                print(f"[ASM] propagating ratio   : {nnz_prop}/{nnz_all} = {nnz_prop/nnz_all:.4f}")
                print(f"[ASM] after band-limit    : {nnz_band}/{nnz_all} = {nnz_band/nnz_all:.4f}")
                print(f"[ASM] f_cut={f_cut:.3e} (min of fx_max, fy_max, Nyquist, 1/λ)")

        # 保存传递函数
        self.register_buffer('H', H.to(self.dtype))

        # ---------- 预备空间坐标（用于薄透镜相位 & IFT/FT 尺寸） ----------
        # 注意：透镜相位按补零后的物理坐标计算（和真实口径一致）
        x = (torch.arange(self.P, device=self.device) - self.P//2) * self.dx
        y = (torch.arange(self.P, device=self.device) - self.P//2) * self.dy
        XX, YY = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('XX', XX)
        self.register_buffer('YY', YY)

        # 薄透镜相位（当 is_lens=True 才会使用）：exp(- i k (x^2+y^2) / (2f))
        if self.is_lens:
            f = self.z  # 你的描述里 distance 当作焦距
            lens_phase = torch.exp(-1j * (k/(2.0*f)) * (self.XX**2 + self.YY**2))
            self.register_buffer('lens_phase', lens_phase.to(self.dtype))

    # ---------------------- 前向传播 ----------------------
    def forward(self, Ein):
        """
        Ein: [B, N, N] 复振幅（complex tensor）
        输出: 与输入同大小 [B, N, N]
        """
        assert Ein.dtype.is_complex, "Ein must be complex tensor (e.g., complex64)."
        B, H, W = Ein.shape
        assert H == self.N and W == self.N, "Ein must be [B, N, N]"

        # 居中镶嵌（补零后再做 FFT）
        E = _center_embed(Ein, self.P, self.P)

        if not self.is_lens:
            # 单段传播：FFT -> 乘 H -> IFFT -> 裁回
            Fk = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2,-1))
            Fk = Fk * self.H
            out = torch.fft.ifft2(torch.fft.ifftshift(Fk, dim=(-2,-1)))
            out = _center_crop(out, self.N, self.N)
            return out.to(self.dtype)

        else:
            # 2F 链路（按你的流程）：
            # 1) FFT
            Fk1 = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2,-1))
            # 2) 乘 H（传播 z）
            Fk1 = Fk1 * self.H
            # 3) IFFT -> 回到空间
            E1 = torch.fft.ifft2(torch.fft.ifftshift(Fk1, dim=(-2,-1)))
            # 4) 乘薄透镜相位
            E2 = E1 * self.lens_phase
            # 5) 再 FFT
            Fk2 = torch.fft.fftshift(torch.fft.fft2(E2), dim=(-2,-1))
            # 6) 再传播 z
            Fk2 = Fk2 * self.H
            # 7) IFFT -> 空间域
            Eout = torch.fft.ifft2(torch.fft.ifftshift(Fk2, dim=(-2,-1)))
            # 8) 裁回原尺寸
            Eout = _center_crop(Eout, self.N, self.N)
            return Eout.to(self.dtype)




class diffraction_module(torch.nn.Module):
    # N_pixels 是 3 倍的 IMG_SIZE ,即 441*3 ，中间441*441是有效的，其他都是补0、填充0
    def __init__(self, λ=532.109e-9, N_pixels=800, pixel_size=8e-6, distance=torch.tensor([0.05])):
        super(diffraction_module, self).__init__()  # 初始化父类
        # 以1/d为单位频率，得到一系列频率分量[0, 1, 2, ···, N_pixels/2-1,-N_pixels/2, ···, -1]/(N_pixels*d)。
        # 设置设备为GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.zernike_basis=torch.tensor( torch.exp(-1j*math.pi*torch.tensor(zernike_polynomials(mode=15, select='all', size=int(N_pixels/2), show=False).astype(np.float32))), device=self.device)
        self.PADDING = int(N_pixels / 4)
        self.IMG_SIZE = N_pixels
        self.distance = torch.tensor(distance).to(self.device)
        self.pixel_size = pixel_size

        # fx = (torch.fft.fftshift(torch.fft.fftfreq(N_pixels, d=pixel_size))).to(self.device)
        # fy = (torch.fft.fftshift(torch.fft.fftfreq(N_pixels, d=pixel_size))).to(self.device)
        # fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')  # 拉网格，每个网格坐标点为空间频率各分量
        # print(fxx.shape, fxx)
        # print(fyy.shape, fyy)

        dx, dy = 8e-6, 8e-6
        num_x, num_y = N_pixels, N_pixels
        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))
        # 这里的 y 和 x 是频率分辨率
        # frequency coordinates sampling
        # 指定间隔起始点、终止端，以及指定分隔值总数（包括起始点和终止点）；最终函数返回间隔类均匀分布的数值序列。
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)
        # 偏移 0.5个频率分辨率的原因是：做角谱法时，希望正好是每个动量空间格子的中心!代表整个格子，而非正方形格子的边缘代表整个格子的kx,ky
        # momentum/reciprocal space 动量空间
        FX, FY = np.meshgrid(fx, fy)
        # print(f"FX={FX},FY={FY}")
        print(FX.shape, FX)
        print(FY.shape, FY)
        fxx, fyy = torch.from_numpy(FX).cuda(), torch.from_numpy(FY).cuda()

        fy_max = 1 / torch.sqrt((2 * self.distance * (1 / y)) ** 2 + 1) / λ
        fx_max = 1 / torch.sqrt((2 * self.distance * (1 / x)) ** 2 + 1) / λ
        print(f"fx_max={fx_max},fy_max={fy_max}")
        print(f"torch.max(fxx)={torch.max(fxx)},torch.max(fyy)={torch.max(fyy)}")
        # f_max=(((1.0 / λ) ** 2) * (
        #             1.0 / (1 + (self.distance ** 2) / (2 * self.IMG_SIZE * self.pixel_size ** 2))))
        # print(f"f_max={f_max}")
        argument = (((2 * torch.pi) ** 2 * ((1.0 / λ) ** 2 - fxx ** 2 - fyy ** 2))).to(self.device)

        # 计算传播场或倏逝场的模式kz，传播场kz为实数，倏逝场kz为复数 有掩膜
        tmp = torch.sqrt(torch.abs(argument)).to(self.device)
        self.kz = torch.tensor(torch.where(argument >= 0, tmp, 0)).to(self.device)  # 大于0是tmp，否则是j*一个虚数或者0
        print(f"torch.max(self.kz)={torch.max(self.kz)},self.kz.dtype={self.kz.dtype}")
        self.kz=self.kz.type_as(torch.tensor([0.0]))
        # 统计零值的数量和位置
        self.count_zeros()
        self.print_zero_indices()

        # self.limited_filter = (fxx ** 2 + fyy ** 2) - (((1.0 / λ) ** 2) * (
        #             1.0 / (1 + (self.distance ** 2) / (2 * self.IMG_SIZE * self.pixel_size ** 2)))).to(self.device)

        # H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

        self.kz=self.kz.cuda()
        fx_max=fx_max.cuda()
        fy_max=fy_max.cuda()
        # c=(torch.abs(fxx)**2+torch.abs(fyy)**2) < fx_max**2
        # print("---\r\n",torch.max(c),'----\r\n')
        self.kz = torch.tensor(torch.where((torch.abs(fxx)**2+torch.abs(fyy)**2) < fx_max**2 , self.kz, 0)).to(self.device)  # 带限衍射
        # 统计零值的数量和位置
        self.count_zeros()
        self.print_zero_indices()
        self.phase = torch.exp(1j * self.kz * self.distance).to(
            self.device)  # 角谱传播，每个频率分量的平面波的相位变化只与distance有关，各个角谱的相位积累



    def forward(self, E):
        # 定义单个衍射层内的前向传播
        # 图像输入进来的是 IMG_SIZE*IMG_SIZE 的，需要在图像周围 padding, 填充一圈0 变成 （3*IMG_SIZE） 边长大小的 与self.phase维度相同
        # 按理说应该要padding的 把 440*440 的图像，上下左右各自padding 填充0，到 1320*1320 大小 如果传输进来的就已经 PADDING 了呢？那就算了
        # 转换为 batch_size*1320*1320 大小的张量矩阵
        # 输入进来的是 200*440*440 维度的张量矩阵，将其上下左右 padding 8个440*440大小的全为0的矩阵，然后傅里叶变换
        # 对输入张量 E 进行零填充
        E = F.pad(E, (self.PADDING, self.PADDING, self.PADDING, self.PADDING), "constant", 0).to(self.device)
        fft_c = torch.fft.fft2(E).to(
            self.device)  # 对电场E进行二维傅里叶变换 不用填充吗？如果填充，那怎么将BATCH_SIZE*IMG_SIZE*IMG_SIZE周围填充0到BATCH_SIZE*N_pixels*N_pixels
        c = torch.fft.fftshift(fft_c).to(self.device)  # 将零频移至张量中心
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * self.phase)).to(
            self.device)  # 不同角谱平面波传输积累不同的self.phase 卷积后逆变换得到响应的角谱
        # 提取中间的 IMG_SIZE x IMG_SIZE 区域
        # print(angular_spectrum.shape)
        # e_output = torch.sqrt(torch.abs(angular_spectrum))
        # print(e_output.shape, e_output.dtype)
        # plt.figure()
        # plt.imshow(e_output.squeeze(0).cpu().numpy())
        # plt.show()
        angular_spectrum_center = angular_spectrum[:, self.PADDING:self.PADDING + int(self.IMG_SIZE / 2),
                                  self.PADDING:self.PADDING + int(self.IMG_SIZE / 2)].to(self.device)
        return angular_spectrum_center

    def count_zeros(self):
        # 统计零值的数量
        print(self.kz.dtype,torch.tensor(0.0).dtype)
        self.num_zeros = torch.sum(torch.isclose(self.kz, torch.tensor(0.0)))

    def print_zero_indices(self):
        # 打印零值的索引
        zero_indices = torch.where(torch.isclose(self.kz, torch.tensor(0.0)))
        print(f"Number of zeros in kz: {self.num_zeros}/{self.kz.size(0)*self.kz.size(1)}")
        print(f"Indices of zeros in kz: {zero_indices}")


def visualize_output(input_tensor, output):
    # 将张量移动到 CPU 并移除 batch 维度
    input_tensor = input_tensor.squeeze().cpu().numpy()  # 输入图像振幅
    output = output.squeeze().cpu()  # 输出复振幅张量

    # 计算振幅和相位
    amplitude = torch.abs(output).numpy()  # 振幅
    phase = torch.angle(output).numpy()  # 相位

    # 创建 1x3 的网格布局，确保每个子图大小一致
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.1)

    # 绘制输入图像的振幅
    ax0 = plt.subplot(gs[0])
    im0 = ax0.imshow(input_tensor, cmap='hot')
    ax0.set_title('Input Amplitude')
    ax0.axis('off')
    # 添加 colorbar
    cbar0 = fig.colorbar(im0, ax=ax0, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar0.set_label('Amplitude')

    # 绘制输出复振幅的振幅
    ax1 = plt.subplot(gs[1])
    im1 = ax1.imshow(amplitude, cmap='hot')
    ax1.set_title('Output Amplitude')
    ax1.axis('off')
    # 添加 colorbar
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar1.set_label('Amplitude')

    # 绘制输出复振幅的相位
    ax2 = plt.subplot(gs[2])
    im2 = ax2.imshow(phase, cmap='hsv')
    ax2.set_title('Output Phase')
    ax2.axis('off')
    # 添加 colorbar
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar2.set_label('Phase (radians)')

    plt.tight_layout()
    plt.show()

    # 可选：保存图像
    # plt.savefig('output_visualization.png', dpi=300)

def num_visual():
    # 设置字体路径（确保系统中有中文字体）
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows字体路径

    # 创建数字和中文数字的列表
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    chinese_digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']

    # 创建图像列表
    images = []
    image_size = 400

    # 创建数字图像
    for digit in digits:
        # 创建一个黑色背景的图像
        img = Image.new('L', (image_size, image_size), color=0)
        draw = ImageDraw.Draw(img)

        # 计算合适的字体大小
        font_size = 1
        while True:
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), digit, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            if text_width >= image_size - 20 or text_height >= image_size - 20:
                break
            font_size += 1

        # 计算绘制位置以居中显示文字，并向上移动30个像素
        bbox = draw.textbbox((0, 0), digit, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((image_size - text_width) // 2, (image_size - text_height) // 2 - 84)

        # 绘制文字
        draw.text(position, digit, fill=255, font=font)

        # 转换为 numpy 数组并归一化到 [0, 1]
        img_array = np.array(img) / 255.0
        images.append(img_array)

    # 创建中文数字图像
    for digit in chinese_digits:
        print(digit)
        img = Image.new('L', (image_size, image_size), color=0)
        draw = ImageDraw.Draw(img)

        # 计算合适的字体大小
        font_size = 1
        while True:
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), digit, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            if text_width >= image_size - 20 or text_height >= image_size - 20:
                break
            font_size += 1

        # 计算绘制位置以居中显示文字
        bbox = draw.textbbox((0, 0), digit, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if digit =='一':
            position = ((image_size - text_width) // 2, (image_size - text_height) // 2 - 85)
        elif digit =='二':
            position = ((image_size - text_width) // 2, (image_size - text_height) // 2 - 65)
        else:
            position = ((image_size - text_width) // 2, (image_size - text_height) // 2 - 30)
        # 绘制文字
        draw.text(position, digit, fill=255, font=font)

        # 转换为 numpy 数组并归一化到 [0, 1]
        img_array = np.array(img) / 255.0
        images.append(img_array)

    # 将图像转换为 PyTorch 张量
    tensor_images = [torch.tensor(img, dtype=torch.float32) for img in images]

    # # 可视化图像
    # plt.figure(figsize=(20, 10))
    #
    # for i in range(20):
    #     plt.subplot(4, 5, i + 1)
    #     plt.imshow(tensor_images[i].numpy(), cmap='gray')
    #     plt.title(f'Image {i}')
    #     plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()
    return tensor_images

def detect_tensor():
    targets = []
    for i in range(0, 10):
        k=40
        roi_space = torch.zeros(10*k, 10*k)
        x=k*i
        y=x
        roi_space[x:(x+k), y:(y+k)] = 1.0  # 白色正方形
        targets.append(roi_space)
    return targets

def generate_unique_random_numbers(num_groups, nums_per_group, min_val, max_val):
    result = []
    while len(result) < num_groups:
        # 生成一个随机行
        row = [random.randint(min_val, max_val) for _ in range(nums_per_group)]
        # 检查该行是否已经存在
        if row not in result:
            result.append(row)
    return result

