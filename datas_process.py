import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---- map 0/1 → ±1 (consistent with π-phase convention) ----
def spins01_to_pm1(spins01: torch.Tensor) -> torch.Tensor:
    # 0 → +1, 1 → -1
    return 1 - 2 * spins01.to(torch.int64)


# ---- average magnetization |m| over ±1 spins ----
def mean_magnetization_pm1(spins_pm1: torch.Tensor) -> float:
    s = spins_pm1.to(torch.float32)
    return float(torch.mean(s).abs().item())


def spin_autocorr_general(
        snapshots_pm1: torch.Tensor,  # [M, N]，每行一个快照，值应为 ±1；若为 0/1 会自动映射
        max_lag: int,
        stride_tau: int = 1,  # 对 τ 下采样，减小连续快照的强相关性
        force_pm1: bool = True
) -> np.ndarray:
    """
    计算 C(t) = (1/N) ⟨ Σ_i s_i(τ) s_i(τ+t) ⟩_τ
    返回 numpy 数组，长度 (max_lag+1)，理论上 C(0)=1。
    """
    x = snapshots_pm1
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.to(torch.float32)
    M, N = x.shape
    max_lag = min(int(max_lag), M - 1)
    stride_tau = max(1, int(stride_tau))

    # 若不是 ±1（如 0/1），映射到 ±1，保证 C(0)=1
    if force_pm1:
        with torch.no_grad():
            # 近似判断 0/1
            uniq = torch.unique(torch.round(x, decimals=6))
            if set(uniq.cpu().numpy().tolist()) <= {0.0, 1.0}:
                # 0->+1, 1->-1（或反过来都不影响相关结构）
                x = torch.cos(np.pi * x)  # cos(0)=+1, cos(pi)=-1
            else:
                # 用符号映射：>=0 -> +1, <0 -> -1
                x = torch.where(x >= 0, torch.tensor(1.0, device=x.device), torch.tensor(-1.0, device=x.device))

    C = np.zeros(max_lag + 1, dtype=np.float64)
    taus = torch.arange(0, M, device=x.device, dtype=torch.long)[::stride_tau]
    for t in range(max_lag + 1):
        valid = taus[taus + t < M]
        if valid.numel() == 0:
            C[t] = np.nan
            continue
        s_tau = x.index_select(0, valid)  # [T,N]
        s_tau_t = x.index_select(0, valid + t)  # [T,N]
        # 先对自旋求均值（1/N ∑_i），再对 τ 求均值
        prod_mean_over_spins = torch.mean(s_tau * s_tau_t, dim=1)  # [T]
        C[t] = float(torch.mean(prod_mean_over_spins).item())
    return C


# N 个自旋；M是需要时间平均的窗口
# ---- time autocorrelation: C(t) = (1/N) < Σ_i s_i(τ) s_i(τ+t) >_τ ----
def spin_autocorr(
        snapshots_pm1: torch.Tensor,  # [M, N], history of ±1 (flattened spins)
        max_lag: int,
        stride_tau: int = 1  # thin τ to reduce correlation of consecutive snapshots
) -> np.ndarray:
    """
    snapshots_pm1: (M, N) torch.float32 or similar, values ±1
    returns: numpy array C(t), length (max_lag+1)
    """
    snapshots = snapshots_pm1.to(torch.float32)
    M, N = snapshots.shape
    max_lag = min(max_lag, M - 1)
    C = np.zeros(max_lag + 1, dtype=np.float64)
    invN = 1.0 / float(N)
    taus = torch.arange(0, M, device=snapshots.device, dtype=torch.long)[::max(1, stride_tau)]

    for t in range(max_lag + 1):
        valid = taus[taus + t < M]
        if valid.numel() == 0:
            C[t] = np.nan
            continue
        s_tau = snapshots.index_select(0, valid)  # [T,N]
        s_tau_t = snapshots.index_select(0, valid + t)  # [T,N]
        prod_mean = torch.mean(s_tau * s_tau_t, dim=1)  # [T]
        # average across τ and normalize by N
        # C[t] = float(invN * torch.sum(prod_mean).item())
        C[t] = float(torch.mean(prod_mean).item())
    return C


ArrayLike1D = Union[List[float], np.ndarray, torch.Tensor]


@dataclass
class SixPanelInputs:
    # [H,W]；可以是 0/π 相位 或 0/1（代表 0/π）
    init_spin_phase: Union[np.ndarray, torch.Tensor]
    best_spin_phase: Union[np.ndarray, torch.Tensor]
    # 迭代日志：能量 / 温度 / |m| / 自相关
    trace_H: ArrayLike1D
    trace_T: ArrayLike1D
    trace_m: ArrayLike1D
    C_t: ArrayLike1D
    title: str = "Simulated Annealing diagnostics"


def _to_np_2d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    assert x.ndim == 2, f"expect 2D array, got {x.shape}"
    return x


def _to_np_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).ravel()


def _phase_to_pm1(phase_2d: np.ndarray) -> np.ndarray:
    """
    支持两种输入：
      - 0/π 相位 → cos(phase) = +1/-1
      - 0/1 码（0 表示 0 相位，1 表示 π）→ cos(π*code) 仍为 ±1
    """
    p = np.asarray(phase_2d, dtype=np.float32)
    uniq = np.unique(np.round(p, 6))
    if set(uniq.tolist()) <= {0.0, 1.0}:
        return np.cos(np.pi * p)  # 0->+1, 1->-1
    else:
        return np.cos(p)  # 0/π -> ±1


def plot_six_panel(diag) -> None:
    """2×3 面板，等面积 + 舒展布局 + inset colorbar。"""
    init_pm1 = _phase_to_pm1(_to_np_2d(diag.init_spin_phase))
    best_pm1 = _phase_to_pm1(_to_np_2d(diag.best_spin_phase))
    H = _to_np_1d(diag.trace_H)
    TT = _to_np_1d(diag.trace_T)
    MM = _to_np_1d(diag.trace_m)
    CC = _to_np_1d(diag.C_t)

    fig = plt.figure(figsize=(14.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, wspace=0.18, hspace=0.22)
    fig.suptitle(diag.title, fontsize=15, fontweight='bold')

    # (1) initial spins
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(init_pm1, cmap='bwr', vmin=-1, vmax=1, interpolation='nearest', aspect='equal')
    ax1.set_title("Initial spin configuration (±1)", fontsize=12)
    ax1.set_xticks([]);
    ax1.set_yticks([])
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="4%", pad=0.04)
    cb1 = fig.colorbar(im1, cax=cax1, ticks=[-1, 1])
    cb1.set_label("spin", fontsize=9)

    # (2) best/final spins
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(best_pm1, cmap='bwr', vmin=-1, vmax=1, interpolation='nearest', aspect='equal')
    ax2.set_title("Best spin configuration (±1)", fontsize=12)
    ax2.set_xticks([]);
    ax2.set_yticks([])
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="4%", pad=0.04)
    cb2 = fig.colorbar(im2, cax=cax2, ticks=[-1, 1])
    cb2.set_label("spin", fontsize=9)

    # (3) Hamiltonian
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(np.arange(len(H)), H, lw=1.8)
    ax3.set_title("Hamiltonian vs. iteration", fontsize=12)
    ax3.set_xlabel("iteration");
    ax3.set_ylabel("H")
    ax3.grid(True, alpha=0.35)

    # (4) Temperature
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(np.arange(len(TT)), TT, lw=1.8, color='tab:orange')
    ax4.set_title("Temperature schedule", fontsize=12)
    ax4.set_xlabel("iteration");
    ax4.set_ylabel("T")
    ax4.grid(True, alpha=0.35)

    # (5) |m|
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(np.arange(len(MM)), MM, lw=1.8, color='tab:green')
    ax5.set_title("Average magnetization |m|", fontsize=12)
    ax5.set_xlabel("iteration");
    ax5.set_ylabel("|m|")
    ax5.set_ylim(0, 1.02)
    ax5.grid(True, alpha=0.35)

    # (6) Autocorrelation
    ax6 = fig.add_subplot(gs[1, 2])
    t = np.arange(len(CC))
    ax6.plot(t, CC, lw=1.9, label="C(t)")
    ax6.set_title("Spin autocorrelation $C(t)$", fontsize=12)
    ax6.set_xlabel("lag $t$");
    ax6.set_ylabel("C(t)")
    ax6.set_ylim(-1.05, 1.05)
    ax6.grid(True, alpha=0.35)
    ax6.legend(frameon=False, fontsize=10)
