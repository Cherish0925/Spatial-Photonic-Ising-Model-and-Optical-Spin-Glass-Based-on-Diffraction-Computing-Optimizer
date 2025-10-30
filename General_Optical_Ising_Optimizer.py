# First, define the solution scale of this Ising model, which is the number of optimized spin states.
# Then, the coupling interaction matrix between spins in the Ising problem is introduced.
# Then, using the general matrix decomposition method,
# the high-rank coupling matrix is decomposed into a series of rank-1 vector outer products,
# and normalization is performed to obtain the weight of each rank-1 vector.
# Using the tensor operations provided by Pytorch,
# the spatial optical Ising machine process corresponding to each rank-1 vector is rapidly solved.
# The Hamiltonian of each sub-vector is weighted and summed.
# Combining this process with the annealing algorithm,
# a general solution method for optimizing the QUBO problem using the spatial optical Ising machine is formed.

"""
    # The simple operations are accomplished based on the CPU and Numpy,
    # while the complex parallel computations are handled by the GPU and Pytorch.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from decompose_J_coupling_matrix import decompose_ising_coupling_with_rank, ising_2d_coupling, ising_coupling_rect, \
    decompose_ising_coupling_with_rank_visualize
from Bandlimited_ASM import _center_embed, _center_crop, BandlimitedASM
from vec2dpm import pack_vectors_to_square, dph_encode_real_single_slm, build_input_amplitude_columnwise, \
    spins_vector_to_block_image
import math
from typing import Union, Tuple, Optional
from get_hamiltonian import hamiltonian_from_Eout
from datas_process import spins01_to_pm1, mean_magnetization_pm1, spin_autocorr, SixPanelInputs, plot_six_panel, \
        spin_autocorr_general

#
width, height = 30, 30
# case 1 最简单的全连接形式的、全正数的连接矩阵
J_Coupling = np.ones((width*height, width*height))  # 秩为 +1 的耦合矩阵，全连接，铁磁；优化后应该是一个状态

# case 2 列优先的长方形伊辛耦合、耦合强度多元化
# J_Coupling = ising_coupling_rect(Lx=width, Ly=height, Jx=1.0, Jy=-0.5, pbc_x=True, pbc_y=True, antiperiodic_x=True,
#                                  antiperiodic_y=False, J2=-1.0, index_order="col", dtype=np.float32)

# case 3 反铁磁形式的、最终的最优解应该是棋盘格形式
# J_Coupling=-1*np.ones((width*height, width*height)) # 秩为 -1 的耦合矩阵，全连接，反铁磁；优化后应该是棋盘格

# case 4 模拟自旋玻璃、高斯分布的耦合强度矩阵
# a = 2*np.random.rand(width*height, width*height)-1
# J_Coupling = a + a.T


# ising_2d_coupling : 模拟二维周期性正方形格子的伊辛模型，返回耦合强度矩阵
# Input the number of spins, the strength of coupling interaction, and the boundary conditions.
# J_Coupling = ising_2d_coupling(10, J=1.0, pbc=True)  # (spins_dim0, spins_dim1) <class 'numpy.ndarray'> float32
print(rf"The spin-coupling interaction matrix has been obtained.")
print(rf"J_Coupling.shape={J_Coupling.shape}, J_Coupling.dtype={J_Coupling.dtype}")
# fig, ax = plt.subplots(figsize=(4, 3.5))
# im = ax.imshow(J_Coupling, cmap='coolwarm', vmin=np.min(J_Coupling), vmax=np.max(J_Coupling)) # 'coolwarm' 'bwr' 'seismic' 'RdBu'
# cb = fig.colorbar(im, ax=ax, ticks=[np.min(J_Coupling),np.mean(J_Coupling),np.max(J_Coupling)])
# cb.set_label(rf'Coupling Strength', fontsize=10)
# ax.set_title(rf'J Coupling Matrix of {spins_total} spins', fontsize=11)
# plt.tight_layout()
# plt.show()

spins_total = J_Coupling.shape[0]  # or J_Coupling.shape[1]
n = int(np.sqrt(spins_total))
if np.power(n, 2) < spins_total:
    n = n + 1
print(rf"Create a spin matrix of size {n}*{n}, but only {spins_total} spins are filled in it.")
T_0 = 1 * 1.0
alpha = 0.999 * 1.0
iterations = 10000
Method = "Simulated_Annealing"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(rf"Computing device is {device}")
wavelength = 634e-9  # wavelength [m]
pixel_size = 8e-6  # pixel size [m]
lens_focal_length = 100e-3
dtype = torch.complex64
# 单自旋耦合因子的扩增倍数，实际上是1*1个像素变成了16*16个;经验值肯定是越大越好；2和4都试过了没有8好；16最好，目前来看；12的效果也还可以
pixel_repeat_Coupling_Jij = 16  # The amplification factor of the single-spin coupling constant actually means that 1*1 pixel has been transformed into 16*16 pixels.
collector = 6  # 6 原本就可以了；最终输出光强面的聚焦的区域。

vectors, weights, r, order, J_rec, mse_map = decompose_ising_coupling_with_rank_visualize(
    J_Coupling,
    tol=1e-10,
    return_full=False,  # 只要前 r 列
    visualize=True,
    visualize_mode="live",  # 重点：非阻塞显示
    pause_s=0.2,  # 给 GUI 一点时间刷新
    fig_title=f"Decomposition for {J_Coupling.shape[0]} spins",
    extras_out=True,  # 需要 J_rec / mse_map
    fig_tag="J-decomp@case1"  # 窗口标题；多次调用时可换一个 tag
)
print(rf"J_rec.shape={J_rec.shape}, J_rec.dtype={J_rec.dtype}")
print(rf"mse_map.shape={mse_map.shape}, mse_map.dtype={mse_map.dtype}")
print(
    rf"The spin-coupling interaction matrix has been decomposed into {r} matrices of rank 1, all of which are of equal size.")
print(rf"The dimension of the decomposed vector matrix with a rank of 1 is {vectors.shape}")
print(rf"The corresponding number of weight vectors is {weights.shape}")

# Next, based on the dual-phase encoding method,
# r coupling coefficient vectors are encoded onto the complex amplitude of the spatial light.
# Each DMP-Spin pixel is 8*8()
real = torch.randn(100, 160, 160, device=device)
imag = torch.randn(100, 160, 160, device=device)
Ein = torch.complex(real, imag).to("cuda")  # 自动 dtype=torch.complex64

Square_tensor = pack_vectors_to_square(vectors, return_torch=True, device="cpu", dtype=torch.float32,
                                       pixel_repeat=pixel_repeat_Coupling_Jij)
print(rf"Square_tensor.shape={Square_tensor.shape},Square_tensor.dtype={Square_tensor.dtype}")
print(rf"torch.max(Square_tensor)={torch.max(Square_tensor)}, torch.min(Square_tensor)={torch.min(Square_tensor)}")
dpm_phase, B = dph_encode_real_single_slm(Square_tensor, tile=1, to01=True, clamp=True)
print(rf"dpm_phase.shape={dpm_phase.shape},dpm_phase.dtype={dpm_phase.dtype}")
print(rf"torch.max(dpm_phase)={torch.max(dpm_phase)}, torch.min(dpm_phase)={torch.min(dpm_phase)}")
asm = BandlimitedASM(wavelength=wavelength, N_size=int(n * pixel_repeat_Coupling_Jij), pixel_size=pixel_size,
                     zero_fill=1, is_lens=True, distance=lens_focal_length,
                     sampling="midpoint", dtype=dtype, device=device, verbose=True)
# 下面定义输入的振幅矩阵（某些理应空白的位置不应该有单位光场输入，否则相当于增加了等效的外加磁场）以及初始自旋格点（一列向量、或者拆分一列一列为矩阵
# 然后每轮优化迭代开始时都拼接为 torch.complex64
input_amp = build_input_amplitude_columnwise(spins_total=spins_total, pixel_repeat=pixel_repeat_Coupling_Jij,
                                             device=device, return_batch=False)
print(rf"input_amp.shape={input_amp.shape}, input_amp.dtype={input_amp.dtype}, input_amp.device={input_amp.device}")

initial_spins_state = torch.randint(0, 2, (spins_total,))
# initial_spins_state = torch.zeros((spins_total,))  # 反铁磁，更容易收敛到基态，如果初始化是均匀的 up or down
# initial_spins_state = torch.arange(spins_total) % 2      # 0,1,0,1…

# 下面是创建初始的棋盘格自旋 profile; 特殊的，最优解，对于反铁磁
# L = int(math.isqrt(spins_total))    # 最近整数边长（向下取整）
# if L * L != spins_total:
#     raise ValueError('spins_total 必须是完全平方数')
# board = (torch.arange(L).unsqueeze(0) + torch.arange(L).unsqueeze(1)) % 2   # 棋盘 0/1
# initial_spins_state = board.reshape(-1)                  # 列优先扁平化 → 一维


init_spin_phase = spins_vector_to_block_image(initial_spins_state,
                                              pixel_repeat=1,
                                              return_batch=False)  # [1,H,W]

print(
    rf"initial_spins_state.shape={initial_spins_state.shape}, initial_spins_state.dtype={initial_spins_state.dtype}, initial_spins_state.device={initial_spins_state.device}")
input_spin_phase = spins_vector_to_block_image(initial_spins_state,
                                               pixel_repeat=pixel_repeat_Coupling_Jij,
                                               return_batch=False).cuda()  # [1,H,W]
print(
    rf"input_spin_phase.shape={input_spin_phase.shape}, input_spin_phase.dtype={input_spin_phase.dtype}, input_spin_phase.device={input_spin_phase.device}")
# --- 确保 dpm_phase 有 batch 维 [r,H,W] ---
# 你这次分解只有一个秩 r=1，dph_encode_real_single_slm 返回了 [H,W] 被你直接用了
if dpm_phase.ndim == 2:
    dpm_phase = dpm_phase.unsqueeze(0)  # -> [1,H,W]
dpm_phase_exp = torch.exp(-1j * 2 * torch.pi * dpm_phase).cuda()  # coupling matrix
print(
    rf"dpm_phase_exp.shape={dpm_phase_exp.shape}, dpm_phase_exp.dtype={dpm_phase_exp.dtype}, dpm_phase_exp.device={dpm_phase_exp.device}")
input_amp_with_dpm_phase_exp = dpm_phase_exp * input_amp
print(
    rf"input_amp_with_dpm_phase_exp.shape={input_amp_with_dpm_phase_exp.shape}, input_amp_with_dpm_phase_exp.dtype={input_amp_with_dpm_phase_exp.dtype}, input_amp_with_dpm_phase_exp.device={input_amp_with_dpm_phase_exp.device}")

E_in = input_amp_with_dpm_phase_exp * torch.exp(-1j * torch.pi * input_spin_phase)
print(rf"E_in.shape={E_in.shape}, E_in.dtype={E_in.dtype}, E_in.device={E_in.device}")

with torch.no_grad():
    E_out = asm(E_in)
    print(rf"E_out.shape={E_out.shape}, E_out.dtype={E_out.dtype}, E_out.device={E_out.device}")


# ------------------------------ 模拟退火（Metropolis） ------------------------------
@torch.no_grad()
def simulated_annealing_optimize(
        initial_spins_state: torch.Tensor,  # [S] in {0,1}
        asm,  # BandlimitedASM 对象，调用 asm(E) → [r,H,W] complex
        input_amp_with_dpm_phase_exp,
        weights: Union[np.ndarray, torch.Tensor],  # [r]
        pixel_repeat,
        steps: int = 2000,
        T0: float = 1.0,
        alpha: float = 0.999,
        win: Union[int, Tuple[int, int]] = 10,
        device: Union[str, torch.device] = "cuda",
        verbose_every: int = 100
):
    """
    逐步单点翻转自旋，用 ASM 计算能量，Metropolis 接受/拒绝。
    返回：spins_best, H_best, trace_H(list), trace_T(list)
    """
    device = torch.device(device)
    # 状态放 CPU (int64) 即可
    spins = initial_spins_state.clone().to(torch.int64).cpu()
    S = int(spins.numel())  # numel() 函数用于返回数组或张量中的元素总数
    input_spin_phase = spins_vector_to_block_image(spins,
                                                   pixel_repeat=pixel_repeat,
                                                   return_batch=False).cuda()  # [1,H,W]
    E_in_0 = input_amp_with_dpm_phase_exp * torch.exp(-1j * torch.pi * input_spin_phase)
    E_Out_0 = asm(E_in_0)
    H_cur = float(hamiltonian_from_Eout(E_Out_0, weights, win=win, device=device).item())

    H_best = H_cur
    spins_best = spins.clone()
    T = float(T0)

    trace_H = [H_cur]
    trace_T = [T]

    for t in range(1, steps + 1):
        # 随机翻转
        k = np.random.randint(0, S)
        old_val = int(spins[k].item())
        spins[k] = 1 - old_val  # 0<->1

        input_spin_phase = spins_vector_to_block_image(spins,
                                                       pixel_repeat=pixel_repeat_Coupling_Jij,
                                                       return_batch=False).cuda()  # [1,H,W]
        E_in = input_amp_with_dpm_phase_exp * torch.exp(-1j * torch.pi * input_spin_phase)

        E_out = asm(E_in)
        H_new = float(hamiltonian_from_Eout(E_out, weights, win=win, device=device).item())

        dE = H_new - H_cur
        accept = (dE <= 0.0) or (np.random.rand() < math.exp(-dE / max(T, 1e-12)))

        if accept:
            H_cur = H_new
            if H_new < H_best:
                H_best = H_new
                spins_best = spins.clone()
        else:
            spins[k] = old_val  # 回滚

        T *= alpha

        if (t % verbose_every) == 0:
            print("[SA] step=%6d  T=%.4e  H_cur=%.6e  H_best=%.6e" % (t, T, H_cur, H_best))

        trace_H.append(H_cur)
        trace_T.append(T)

    return spins_best, H_best, trace_H, trace_T


# 带有记录的模拟退火，logs 意义是记录、记录优化迭代的过程中的各种参数，尤其是将 spins±1 构型同步记录，后续可以算自相关
# snapshots_pm1_flat：每隔几步保存一次整幅自旋（±1），形状 [M,S]，用于时序统计。
# 在时间维度上保存多少系统状态（快照）: 输入参数 snapshot_every —— 控制「记录频率」
# 若想要连续的自相关曲线（C(t) 很平滑），应当保存得更密集。
@torch.no_grad()
def simulated_annealing_optimize_with_logs(
        initial_spins_state: torch.Tensor,  # [S] in {0,1}
        asm,  # BandlimitedASM
        input_amp_with_dpm_phase_exp,  # [r,H,W] complex
        weights: Union[np.ndarray, torch.Tensor],  # [r]
        pixel_repeat: int,
        steps: int = 2000,
        T0: float = 1.0,
        alpha: float = 0.999,
        win: Union[int, Tuple[int, int]] = 10,
        device: Union[str, torch.device] = "cuda",
        verbose_every: int = 100,
        snapshot_every: int = 20  # store spin snapshots periodically snapshot_every —— 控制「记录频率」
):
    """
    Single-spin flips with Metropolis acceptance.
    Returns: spins_best, H_best, trace_H, trace_T, trace_m, snapshots_pm1_flat
    """
    device = torch.device(device)
    spins = initial_spins_state.clone().to(torch.int64).cpu()  # [S] 0/1
    S = int(spins.numel())

    # initial energy
    input_spin_phase = spins_vector_to_block_image(spins,
                                                   pixel_repeat=pixel_repeat,
                                                   return_batch=False).cuda()
    E_in_0 = input_amp_with_dpm_phase_exp * torch.exp(-1j * torch.pi * input_spin_phase)
    E_out_0 = asm(E_in_0)
    H_cur = float(hamiltonian_from_Eout(E_out_0, weights, win=win, device=device).item())

    H_best, spins_best = H_cur, spins.clone()
    T = float(T0)

    trace_H = [H_cur]
    trace_T = [T]
    trace_m = [mean_magnetization_pm1(spins01_to_pm1(spins))]

    snapshots = [spins01_to_pm1(spins).to(torch.float32)]  # [S]

    for t in range(1, steps + 1):
        k = np.random.randint(0, S)
        old_val = int(spins[k].item())
        spins[k] = 1 - old_val  # flip 0↔1

        input_spin_phase = spins_vector_to_block_image(spins,
                                                       pixel_repeat=pixel_repeat,
                                                       return_batch=False).cuda()
        E_in = input_amp_with_dpm_phase_exp * torch.exp(-1j * torch.pi * input_spin_phase)
        E_out = asm(E_in)
        H_new = float(hamiltonian_from_Eout(E_out, weights, win=win, device=device).item())

        dE = H_new - H_cur
        accept = (dE <= 0.0) or (np.random.rand() < np.exp(-dE / max(T, 1e-12)))
        if accept:
            H_cur = H_new
            if H_new < H_best:
                H_best = H_new
                spins_best = spins.clone()
        else:
            spins[k] = old_val  # rollback

        T *= alpha

        if (t % verbose_every) == 0:
            print(f"[SA] step={t:6d}  T={T:.4e}  H_cur={H_cur:.6e}  H_best={H_best:.6e}")

        trace_H.append(H_cur)
        trace_T.append(T)
        trace_m.append(mean_magnetization_pm1(spins01_to_pm1(spins)))

        if (t % snapshot_every) == 0:
            snapshots.append(spins01_to_pm1(spins).to(torch.float32))

    if len(snapshots) == 0:
        snapshots.append(spins01_to_pm1(spins).to(torch.float32))

    snapshots_pm1_flat = torch.stack(snapshots, dim=0).contiguous()  # [M, S]
    return spins_best, H_best, trace_H, trace_T, trace_m, snapshots_pm1_flat


with torch.no_grad():
    spins_best, H_best, trace_H, trace_T, trace_m, snapshots_pm1_flat = simulated_annealing_optimize_with_logs(
        initial_spins_state=initial_spins_state,
        asm=asm,
        input_amp_with_dpm_phase_exp=input_amp_with_dpm_phase_exp,
        weights=weights,
        pixel_repeat=pixel_repeat_Coupling_Jij,
        steps=iterations,
        T0=T_0,
        alpha=alpha,
        win=collector,
        device=device,
        verbose_every=200,
        snapshot_every=1
    )
print(f"[SA] Best H = {H_best:.6e}")

# build 0/1 phase images for panel (1) and (2)
init_spin_phase = spins_vector_to_block_image(initial_spins_state, pixel_repeat=1, return_batch=False)
best_spin_phase = spins_vector_to_block_image(spins_best, pixel_repeat=1, return_batch=False)
print(
    rf"snapshots_pm1_flat.shape={snapshots_pm1_flat.shape}, snapshots_pm1_flat.dtype={snapshots_pm1_flat.dtype}, snapshots_pm1_flat.device={snapshots_pm1_flat.device}")
print(rf"torch.max(snapshots_pm1_flat)={torch.max(snapshots_pm1_flat)}, torch.min(snapshots_pm1_flat)={torch.min(snapshots_pm1_flat)}")

# autocorrelation (keep a moderate max_lag)
# 经典自旋/自旋玻璃里常用的 时间自相关对「时间起点 τ」取平均（也可理解为时间平均）
# Temporal correlation decay / freezing signature
# C(0)=1;C(t) 下降越快，说明系统越快遗忘过去（混合好/探索快）;
# 下降很慢或在非零值qEA平台化，表示冻结/玻璃化或强有序（动力学很慢）。
# 可以写出 C(t)=q_EA+(1-q_EA)e^{-t/τ} 然后反向拟合得到松弛时间 τ 以及Edwards–Anderson 参量𝑞_𝐸𝐴
# 严格意义上自相关最好在固定温度阶段测;固定温度段或者晚期/早期（温度变化不剧烈的时候）
# snapshots_pm1_flat:期望输入 snapshots_pm1_flat 形状 [M, S]，M 是保存的快照数、S 是自旋数；数值须为 ±1。
# C_t = spin_autocorr(
#     snapshots_pm1_flat,
#     max_lag=min(300, snapshots_pm1_flat.shape[0] - 1),
#     stride_tau=1
# )

# 计算 C(t) = (1/N) Σ_i(i 指的是对所有自旋)⟨s_i(τ)*s_i(τ+t)⟩_τ
# τ 指的是选中所有间隔时间延迟（lag）为 t 的自旋对做点积
# max_lag 就是你打算算到的最大延迟，lag 指的是相对的时间延迟
# max_lag 不能设置的太大，如果设太大，后半段几乎没有有效平均（因为 τ+t 超出范围），数值会非常噪。
# 规律：小 max_lag → 快速计算，曲线短但平滑；
# 大 max_lag → 可以看到长时间衰减（例如是否有 Edwards–Anderson 平台），但曲线末尾会变噪。
# 一般取max_lag ≈ M / 10 或 M / 5（M 是快照总数） 这里 M = snapshots_pm1_flat.shape[0]
# stride_tau —— 控制「时间平均的下采样间隔」;stride_tau=1 → 每个快照都参与；提高统计独立性；不改变自相关形状（只是更稀疏采样）。
# 在 t 确定后，⟨s_i(τ)*s_i(τ+t)⟩_τ 其实是针对 M-t 个时间窗口来算，按理说会有 (M-t)*(M-t-1)/2 pairs, 但是对 τ 再次下采样，并非对所有 M-t 个时间窗口都两两计算
C_t = spin_autocorr_general(
    snapshots_pm1_flat,
    # max_lag=min(300, snapshots_pm1_flat.shape[0] - 1), # 如果 max_lag = 300，你只计算从 t=0 到 t=300；
    max_lag=int(snapshots_pm1_flat.shape[0]/5),
    stride_tau=1 # 对τ的下采样步长。>1 可降噪、降算量；=1 表示用满全部 𝜏
)

# final 2×3 figure
diag = SixPanelInputs(
    init_spin_phase=init_spin_phase,
    best_spin_phase=best_spin_phase,
    trace_H=trace_H,
    trace_T=trace_T,
    trace_m=trace_m,
    C_t=C_t,
    title="Simulated Annealing diagnostics (optical Ising)"
)
plot_six_panel(diag)
plt.show()

# with torch.no_grad():
#     spins_best, H_best, trace_H, trace_T = simulated_annealing_optimize(
#         initial_spins_state=initial_spins_state,
#         asm=asm,
#         input_amp_with_dpm_phase_exp=input_amp_with_dpm_phase_exp,
#         weights=weights,
#         pixel_repeat=pixel_repeat_Coupling_Jij,
#         steps=iterations,
#         T0=T_0,
#         alpha=alpha,
#         win=collector,
#         device=device,
#         verbose_every=100
#     )
# print(f"[SA] Best H = {H_best:.6e}")
# best_spin_phase = spins_vector_to_block_image(spins_best,
#                                               pixel_repeat=1,
#                                               return_batch=False)  # [1,H,W]
#
# fig, ax = plt.subplots(figsize=(4, 3.5))
# phase2spin = torch.exp(-1j * 1 * torch.pi * init_spin_phase.detach()).real
# im = ax.imshow(phase2spin, cmap='bwr', vmin=-1, vmax=1)  # 'coolwarm' 'bwr' 'seismic' 'RdBu'
# cb = fig.colorbar(im, ax=ax, ticks=[-1, 1])
# cb.set_label(rf'Spin State', fontsize=10)
# ax.set_title(rf'2D Spin Lattice Initialization', fontsize=11)
# plt.tight_layout()
# plt.show()
# fig, ax = plt.subplots(figsize=(4, 3.5))
# phase2spin = torch.exp(-1j * 1 * torch.pi * best_spin_phase.detach()).real
# im = ax.imshow(phase2spin, cmap='bwr', vmin=-1, vmax=1)  # 'coolwarm' 'bwr' 'seismic' 'RdBu'
# cb = fig.colorbar(im, ax=ax, ticks=[-1, 1])
# cb.set_label(rf'Spin State', fontsize=10)
# ax.set_title(rf'Best Spin After Optimization', fontsize=11)
# plt.tight_layout()
# plt.show()
