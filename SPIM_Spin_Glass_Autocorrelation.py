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
from scipy.optimize import curve_fit

width, height = 10, 10
# case 4 模拟自旋玻璃、高斯分布的耦合强度矩阵
a = 2 * np.random.rand(width * height, width * height) - 1
J_Coupling = a + a.T
print(rf"The spin-coupling interaction matrix has been obtained.")
print(rf"J_Coupling.shape={J_Coupling.shape}, J_Coupling.dtype={J_Coupling.dtype}")
spins_total = J_Coupling.shape[0]  # or J_Coupling.shape[1]
n = int(np.sqrt(spins_total))
if np.power(n, 2) < spins_total:
    n = n + 1
print(rf"Create a spin matrix of size {n}*{n}, but only {spins_total} spins are filled in it.")
alpha = 1.0
iterations = 10000
Method = "Simulated_Annealing"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(rf"Computing device is {device}")
wavelength = 634e-9  # wavelength [m]
pixel_size = 8e-6  # pixel size [m]
lens_focal_length = 100e-3
dtype = torch.complex64
# 单自旋耦合因子的扩增倍数，实际上是1*1个像素变成了16*16个;经验值肯定是越大越好；2和4都试过了没有8好；16最好，目前来看；12的效果也还可以
pixel_repeat_Coupling_Jij = 8  # The amplification factor of the single-spin coupling constant actually means that 1*1 pixel has been transformed into 16*16 pixels.
collector = 6  # 6 原本就可以了；最终输出光强面的聚焦的区域。收集中间 6*6 像素区域中的光强的和
# T_list = torch.logspace(np.log10(0.2), np.log10(5.0), 12)  # 0.2 … 5.0
T_list = torch.tensor([0.001, 0.010, 0.10, 0.3,0.6, 1.0,1.25,1.5, 3.0, 10.0])
C_dict = {}
qEA_arr = []
tau_arr = []

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


# build 0/1 phase images for panel (1) and (2)
init_spin_phase = spins_vector_to_block_image(initial_spins_state, pixel_repeat=1, return_batch=False)


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


def exp_decay(t, qEA, tau):
    return qEA + (1 - qEA) * np.exp(-t / tau)


def fit_autocorr(t, C):
    try:
        popt, _ = curve_fit(exp_decay, t, C, bounds=([0, 0], [1, np.inf]))
        return popt
    except:
        return np.nan, np.nan


with torch.no_grad():
    for T in T_list:
        spins_best, H_best, trace_H, trace_T, trace_m, snapshots_pm1_flat = simulated_annealing_optimize_with_logs(
            initial_spins_state=initial_spins_state,
            asm=asm,
            input_amp_with_dpm_phase_exp=input_amp_with_dpm_phase_exp,
            weights=weights,
            pixel_repeat=pixel_repeat_Coupling_Jij,
            steps=iterations,
            T0=T,
            alpha=alpha,
            win=collector,
            device=device,
            verbose_every=200,
            snapshot_every=1
        )
        snapshots = snapshots_pm1_flat
        # t = np.arange(snapshots.shape[0] // 5 + 1)
        # 1. 先确定 max_lag（想用满窗口就设 M-1）
        max_lag = int(snapshots.shape[0] // 5 - 1)  # 0...M-1 → 长度 M # 毕竟是从 C(0) 到 C(max_lag)
        # 因此 snapshots.shape[0] 个窗口，第一个和最后一个的间隔最多也就 （snapshots.shape[0] - 1）
        t = np.arange(max_lag + 1)  # t 只是刻度向量，与 C_t 一一对应，方便画图和拟合；它不影响计算，只影响横轴标签。

        C_t = spin_autocorr_general(
            snapshots,
            # max_lag=min(300, snapshots_pm1_flat.shape[0] - 1), # 如果 max_lag = 300，你只计算从 t=0 到 t=300；
            max_lag=max_lag,
            stride_tau=1  # 对τ的下采样步长。>1 可降噪、降算量；=1 表示用满全部 𝜏
        )

        C_dict[float(T)] = (t, C_t)
        qEA, tau = fit_autocorr(t, C_t)
        print(rf"T={T}, qEA={qEA}, tau={tau}")
        # qEA → 平台高度（Edwards–Anderson序参量）tau → 松弛时间（指数衰减特征时间）
        # 如果C(t)几乎不下降（低温玻璃），拟合会把衰减部分归因于很大的tau → 数值上tau ≫ 模拟窗口，表示系统极慢地遗忘初始构型。
        qEA_arr.append(qEA)
        tau_arr.append(tau)

# 1. 确保温度从小到大 → 图例自上而下 = 高温→低温
T_list_sorted, C_items = zip(*sorted(C_dict.items(), key=lambda x: x[0]))

plt.figure(figsize=(6, 4))
for T, (t, C) in zip(T_list_sorted, C_items):
    # ---- 散点 ----
    plt.semilogx(t, C, 'o', ms=3, label=f'T = {T:.2f}')

    # ---- 拟合线 ----
    qEA, tau = fit_autocorr(t, C)
    if not (np.isnan(qEA) or np.isnan(tau)):
        plt.semilogx(t, exp_decay(t, qEA, tau), '-', lw=1,
                     color=plt.gca().lines[-1].get_color(), alpha=0.8)

plt.xlabel('Monte-Carlo time t')
plt.ylabel('C(t)')
plt.ylim(-0.05, 1.05)
plt.grid(True, which='both', ls='--', alpha=.4)
plt.legend(ncol=2, fontsize=8, frameon=False)
plt.title('Autocorrelation vs time @ different T')
plt.tight_layout()
plt.show()
