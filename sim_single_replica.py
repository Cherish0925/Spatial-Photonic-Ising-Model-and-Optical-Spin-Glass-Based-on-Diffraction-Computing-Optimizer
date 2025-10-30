# 仿真空间光伊辛机器，2D 自旋格点系统；通过 2F 光学傅里叶变换模块，高效计算自旋相互作用以及外加磁场下的哈密顿量；
# 研究相变过程发生的条件，以及相变过程带来的序参量的变化

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
from Diffraction_Propagation_Module import diffraction_module

# ==========================================================
# 1. 系统常数
# ==========================================================
λ      = 634.0e-9
f      = 100e-3
px     = 8e-6
N_spin = 100                     # 伊辛格点
N_total  = 200                     # diffraction 内部总网格(算力足够),上下左右补0各补1/4
device = torch.device('cuda')
N_coarse = 50           # 巨像素格点
scale    = N_spin // N_coarse   # 4（100/25）
# 每个巨像素 0-1 正态分布
amp_coarse = torch.empty(N_coarse, N_coarse, device=device).normal_(mean=0.5, std=0.1)
amp_coarse.clamp_(0, 1)          # 裁到 [0,1]

beta=1.0*1.0
delta_x_y=8 # 2-6，7-11可以，7：0.19、8：0.14、9：0.169、10：0.1、11：0.25；所以以后一直选择8，经验值，就这样吧

# 放大 4×4 → 100×100（重复块）
amp_map = amp_coarse.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)

# 形状 [100,100]，与 phase 对齐
# 2F 衍射层（两次 f 距离）
prop_f = diffraction_module(λ=λ,
                                N_pixels=N_total,
                                pixel_size=px,
                                distance=torch.tensor([f])).to(device)


# ----------------------------------------------------------
# 1. 先算每个像素对应的物理坐标（中心对齐）
# ----------------------------------------------------------
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"decice={device}")
# ------------------ 1. NumPy 端计算 ------------------
L         = N_total * px
x_center  = np.linspace(-L/2 + px/2, L/2 - px/2, N_total)
X_np, Y_np = np.meshgrid(x_center, x_center, indexing='ij')

# 2×2 局部打印验证精度
print('X_np[0:2,0:2]:\n', X_np[0:2,0:2])
print('Y_np[0:2,0:2]:\n', Y_np[0:2,0:2])
print('X_np[(N_total-2):N_total,(N_total-2):N_total]:\n', X_np[(N_total-2):N_total,(N_total-2):N_total])
print('Y_np[(N_total-2):N_total,(N_total-2):N_total]:\n', Y_np[(N_total-2):N_total,(N_total-2):N_total])
# 透镜相位（复数，在 NumPy 中计算）
lens_phase_np = np.exp(-1j * np.pi * (X_np**2 + Y_np**2) / (λ * f))

# ------------------ 2. 搬到 torch / cuda ------------------
lens_phase = torch.from_numpy(lens_phase_np).to(device, dtype=torch.complex64)

# 可选：打印验证
print('lens_phase shape:', lens_phase.shape)
print('dtype on device :', lens_phase.dtype)

# ==========================================================
# 哈密顿量（取中心 3×3 像素强度）
# ==========================================================
def hamiltonian(spin):
    """
    spin : [N_spin, N_spin] 取值 ±1
    """
    # 1. 把自旋映射到 0/π 相位
    phase_small = (1 - torch.tensor(spin, device=device, dtype=torch.float32)) * np.pi / 2
    # print(f"phase_small.shape={phase_small.shape}")
    # # 2. 网格中央放 100×100 有效相位，其余补零
    # phase_big = torch.zeros(N_total, N_total, device=device)
    # print(f"phase_big.shape={phase_big.shape}")
    c = N_total // 2 - N_spin // 2
    # phase_big[c:c+N_spin, c:c+N_spin] = phase_small

    # 3. 构造复振幅
    field = amp_map*torch.exp(1j * phase_small)

    # 4. 2F 系统：自由空间→透镜相位→自由空间
    field = prop_f(field.unsqueeze(0))
    field *= lens_phase[c:c+N_spin, c:c+N_spin]
    field = prop_f(field).squeeze()

    # 5. 中心 3×3 强度
    cnt = N_total // 2
    intensity = torch.abs(field)**2

    # print(f"intensity={intensity.shape}")
    # plt.imshow(intensity.cpu().numpy(), cmap='hot')
    # plt.colorbar(label='Intensity (a.u.)')
    # plt.title('2F 后焦面强度')
    # plt.show()

    return intensity[(50-delta_x_y):(50+delta_x_y), (50-delta_x_y):(50+delta_x_y)].sum().item()


def fft_hamiltonian(spin, device='cuda'):
    """
    spin : numpy.ndarray, 末尾两维为 [..., N, N]
    返回
        base_intensity : np.ndarray, shape 与 spin[..., 0, 0] 相同
        amp_center     : np.ndarray, 中心 N×N 频谱振幅
    """
    # 1) numpy → torch → cuda
    spin_t = torch.as_tensor(spin, dtype=torch.float32, device=device)

    # 2) 相位映射
    phase = (1 - spin_t) * np.pi

    # 3) 复振幅
    field = amp_map*torch.exp(1j * phase)                       # [..., N, N]

    # 4) 零填充 3N×3N：上下左右各补 N 个 0
    *lead, N, _ = field.shape
    field_pad = F.pad(field, (N, N, N, N), mode='constant', value=0)   # [..., 3N, 3N]
    # 5) FFT 与 fftshift
    fft_pad = torch.fft.fft2(field_pad, dim=(-2, -1))
    spectrum_shift = torch.fft.fftshift(fft_pad, dim=(-2, -1))

    # 6) 零频强度（3N×3N 的左上角 (0,0)）
    base_complex = fft_pad[..., 0, 0]
    print(f"base_complex={base_complex}")
    base_intensity = torch.abs(base_complex)**2

    # 7) 裁剪中心 N×N 频谱振幅
    *_, H3, W3 = spectrum_shift.shape
    c = H3 // 2
    amp_center = spectrum_shift[..., c-N//2:c+N//2, c-N//2:c+N//2]

    return base_intensity.cpu().numpy(), amp_center.cpu().numpy()

# ==========================================================
# MCMC（Metropolis，100 轮）
# beta 是温控，需要精心设计与摸索
# spin 是 numpy 的二维矩阵变量，可能是复数
# ==========================================================
def mcmc_ising(steps=100, beta=1.0):
    # spin = 2*np.random.randint(0, 2, (N_spin, N_spin)) - 1   # ±1
    spin_coarse = 2 * np.random.randint(0, 2, (N_coarse, N_coarse)) - 1
    # spin_coarse = 2 * np.ones((N_coarse, N_coarse))  # ±1
    # print(f'spin.shape={spin.shape},np.mean(spin)={np.mean(spin)},spin={spin}')
    energies, mags = [], []                     # 记录能量和一阶磁化
    spin = np.repeat(np.repeat(spin_coarse, scale, axis=0), scale, axis=1)
    old_hamiltonian=-1*hamiltonian(spin)

    # base_I, amp_N = fft_hamiltonian(spin)
    # print(f"base_I={base_I}")
    # print(f"amp_N.shape={amp_N.shape}")
    # plt.imshow(np.abs(amp_N)**2, cmap='hot')
    # plt.colorbar(label='amp_N (a.u.)')
    # plt.title('2F 后焦面强度')
    # plt.show()
    # old_hamiltonian = -1 * base_I
    mags.append(np.abs(np.mean(spin_coarse)))  # 巨像素上的 |m|
    energies.append(old_hamiltonian)

    for _ in range(steps):
        # print(f"old_hamiltonian={old_hamiltonian}")
        i, j = np.random.randint(0, N_coarse, 2)
        spin_coarse_old_ij = spin_coarse[i, j]
        spin_coarse[i, j] *= -1 # 翻转单个自旋格点
        spin = np.repeat(np.repeat(spin_coarse, scale, axis=0), scale, axis=1)
        new_hamiltonian=-1*hamiltonian(spin)

        # base_I, amp_N = fft_hamiltonian(spin)
        # print(f"base_I={base_I}")

        # new_hamiltonian = -1 * base_I
        delta_E = new_hamiltonian - old_hamiltonian
        # 如果温度下降了，即 delta_E < 0，则直接接受翻转
        # print(f"delta_E={delta_E}")
        if delta_E > 0:
            markov_random_number = np.random.rand()
            transition_probability = np.exp(-beta * delta_E)
            # print(f"transition_probability={transition_probability}")
            if not transition_probability > markov_random_number:
                spin_coarse[i, j] = spin_coarse_old_ij      # 拒绝
        old_hamiltonian=new_hamiltonian
        energies.append(old_hamiltonian)
        mags.append(np.abs(np.mean(spin_coarse)))
    spin = np.repeat(np.repeat(spin_coarse, scale, axis=0), scale, axis=1)
    return spin_coarse, energies, mags

# ==========================================================
# 运行 & 绘图
# ==========================================================
if __name__ == "__main__":
    final_spin, energy, magnetization = mcmc_ising(steps=10000, beta=beta)

    plt.figure(figsize=(10, 4))

    # 1) 自旋构型
    plt.subplot(1, 2, 1)
    plt.imshow(np.repeat(np.repeat(final_spin, scale, axis=0), scale, axis=1),
               cmap='gray', vmin=-1, vmax=1)
    plt.title("Final Spin Configuration")
    plt.axis('off')

    # 2) 双轴：能量 + 磁化强度
    ax1 = plt.subplot(1, 2, 2)
    # 能量曲线（蓝色，左侧 y 轴）
    line1 = ax1.plot(energy, color='tab:blue', label='Energy')
    ax1.set_xlabel("Markov step")
    ax1.set_ylabel("Ising Energy H", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 磁化强度 |m|（红色，右侧 y 轴）
    ax2 = ax1.twinx()
    line2 = ax2.plot(magnetization, color='tab:red', label='|m|')
    ax2.set_ylabel("Magnetization |m|", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper right')
    ax1.set_title("Energy & |m| vs. Step")
    plt.tight_layout()
    plt.show()