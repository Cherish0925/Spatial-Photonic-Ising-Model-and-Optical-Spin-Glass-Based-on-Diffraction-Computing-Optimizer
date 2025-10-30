# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches

# —— 如需强制交互后端，请在导入 pyplot 之前启用（脚本最顶端做一次即可）——
# matplotlib.use("TkAgg")


def decompose_ising_coupling_with_rank_visualize(
    J: np.ndarray,
    tol: float = 1e-10,
    return_full: bool = True,
    visualize: bool = False,
    extras_out: bool = False,
    fig_title: str = "decompose_ising_coupling_with_rank",
    visualize_mode: str = "live",     # "live" 非阻塞; "block" 阻塞; "off" 不画
    pause_s: float = 0.15,
    fig_tag: str = "J-decomp"
):
    """
    对称实矩阵 J 的秩-1分解：J ≈ Σ w_k (v_k v_k^T)
    - eigh 得到 (λ_k, q_k)
    - 归一化 v_k = q_k / max|q_k|
    - 权重   w_k = λ_k * (max|q_k|)^2
    - 按 |w_k| 降序排序；第二子图的柱子是“按索引 0..r-1 排列的 w_k（正负同轴）”

    visualize_mode:
        "live"  -> 非阻塞显示（边跑边看）
        "block" -> 阻塞直到关窗
        "off"   -> 不画

    返回:
        vectors, weights, r, order  （与原版一致）
        若 extras_out=True 额外返回 J_rec, mse_map
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square matrix")
    if not np.allclose(J, J.T, atol=tol):
        raise ValueError("J must be a real symmetric matrix")

    N = J.shape[0]
    # eigh: λ 升序
    lam_all, U_all = np.linalg.eigh(J)
    mask = np.abs(lam_all) > tol
    lam = lam_all[mask]
    U   = U_all[:, mask]
    r   = lam.size

    if r == 0:
        vectors = np.zeros((N, N if return_full else 0), dtype=float)
        weights = np.zeros((N if return_full else 0,), dtype=float)
        order   = np.array([], dtype=int)
        if extras_out:
            return vectors, weights, 0, order, np.zeros_like(J), np.zeros_like(J)
        return vectors, weights, 0, order

    maxabs = np.max(np.abs(U), axis=0)
    safe   = maxabs > tol
    V      = np.zeros_like(U)
    w      = np.zeros((r,), dtype=float)
    V[:, safe] = U[:, safe] / maxabs[safe]
    w[safe]    = lam[safe] * (maxabs[safe] ** 2)

    # 按 |w| 降序排序
    order = np.argsort(-np.abs(w))
    V = V[:, order]
    w = w[order]

    # 重构与 MSE map
    J_rec = (V * w[np.newaxis, :]) @ V.T
    err   = J - J_rec
    mse_map = err**2
    mse_scalar = float(mse_map.mean())

    # ————————— 可视化 —————————
    if visualize and visualize_mode != "off":
        _plot_decomp_4panels_live(
            J=J, V=V, w=w, J_rec=J_rec, mse_map=mse_map,
            mse_scalar=mse_scalar, r=r, fig_title=fig_title, mode=visualize_mode,
            pause_s=pause_s, fig_tag=fig_tag
        )

    # ——— 返回
    if return_full:
        vectors = np.zeros((N, N), dtype=float)
        weights = np.zeros((N,), dtype=float)
        vectors[:, :r] = V
        weights[:r]    = w
    else:
        vectors = V
        weights = w

    if extras_out:
        return vectors, weights, r, order, J_rec, mse_map
    return vectors, weights, r, order


def _plot_decomp_4panels_live(
    J, V, w, J_rec, mse_map, mse_scalar, r,
    fig_title=None, mode="live", pause_s=0.15, fig_tag="J-decomp"
):
    # === 统一颜色与数值范围 ===
    cmap_J   = "coolwarm"
    cmap_mse = "magma"
    vmax = float(np.max(np.abs(J))) or 1.0
    vmin = -vmax

    # === 布局：上 3 图 + 下 1（长条权重） ===
    fig = plt.figure(figsize=(12.8, 8.2), num=fig_tag, clear=True)
    gs  = gridspec.GridSpec(
        nrows=2, ncols=3, height_ratios=[1.0, 0.68], hspace=0.32, wspace=0.28
    )

    title = fig_title or f"Decomposition for {J.shape[0]} spins"
    title2 = f"Rank r = {r}  |  MSE = {mse_scalar:.3e}"
    fig.suptitle(f"{title}\n{title2}", fontsize=14, y=0.98)

    # --- (1) 原始 J ---
    axJ = fig.add_subplot(gs[0, 0])
    im0 = axJ.imshow(J, cmap=cmap_J, vmin=vmin, vmax=vmax)
    axJ.set_title("Original Coupling Matrix $J$", fontsize=12)
    axJ.set_xlabel("Index"); axJ.set_ylabel("Index")
    cb0 = fig.colorbar(im0, ax=axJ, fraction=0.046, pad=0.02)
    cb0.set_label("Coupling strength", fontsize=9)

    # --- (2) 重构 J_rec ---
    axR = fig.add_subplot(gs[0, 1])
    im1 = axR.imshow(J_rec, cmap=cmap_J, vmin=vmin, vmax=vmax)
    axR.set_title(r"Reconstructed $J_{\rm rec}=\sum_k w_k v_k v_k^{\top}$", fontsize=12)
    axR.set_xlabel("Index"); axR.set_ylabel("Index")
    cb1 = fig.colorbar(im1, ax=axR, fraction=0.046, pad=0.02)
    cb1.set_label("Coupling strength (reconstructed)", fontsize=9)

    # --- (3) MSE map ---
    axE = fig.add_subplot(gs[0, 2])

    # 计算最小值、最大值（排除 nan，防止溢出）
    mse_min = float(np.nanmin(mse_map))
    mse_max = float(np.nanmax(mse_map))

    # 显式指定 vmin/vmax，颜色映射和数值一致
    im2 = axE.imshow(mse_map, cmap=cmap_mse, vmin=mse_min, vmax=mse_max)
    axE.set_title(r"MSE Map $(J-J_{\rm rec})^2$", fontsize=12)
    axE.set_xlabel("Index")
    axE.set_ylabel("Index")

    # colorbar 范围与 ticks 一致
    num_ticks = 5
    tick_vals = np.linspace(mse_min, mse_max, num_ticks)
    cb2 = fig.colorbar(im2, ax=axE, fraction=0.046, pad=0.02, ticks=tick_vals)
    cb2.ax.set_yticklabels([f"{v:.1e}" for v in tick_vals])  # 科学计数法
    cb2.set_label(f"squared error  (min={mse_min:.1e}, max={mse_max:.1e})", fontsize=9)

    # --- (4) 权重长条图（独占一整行） ---
    axW = fig.add_subplot(gs[1, :])

    # 排序后 w 已按 |w| 递减；x 轴为 0..r-1
    x = np.arange(r)
    w_pos = np.where(w > 0, w, 0.0)
    w_neg = np.where(w < 0, w, 0.0)

    # 更顺眼的配色/描边
    c_pos, c_neg = "#e60000", "#87CEEB"     # 正负长方图的颜色
    e_pos, e_neg = "#DC143C", "#87CEFA"     # 边框的颜色

    # 正负分两层绘制，alpha & edgecolor 更细腻
    axW.bar(x, w_pos, color=c_pos, edgecolor=e_pos, linewidth=0.4, width=0.92, label="positive $w_k$", alpha=0.95)
    axW.bar(x, w_neg, color=c_neg, edgecolor=e_neg, linewidth=0.4, width=0.92, label="negative $w_k$", alpha=0.95)

    # 零线、网格、范围
    axW.axhline(0.0, color="#888888", lw=0.9, alpha=0.8)
    maxabs = float(np.max(np.abs(w))) if r > 0 else 1.0
    axW.set_ylim(-1.1*maxabs, 1.1*maxabs)
    axW.set_xlim(-0.5, r-0.5)
    axW.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

    # 轴标题与图名
    axW.set_title("Weights $w_k$ (ordered by $|w|$ desc; index left→right)", fontsize=12)
    axW.set_xlabel("index $k$\in[0,r-1]")
    axW.set_ylabel("weight $w_k$")

    # 稀疏 x-ticks，避免拥挤
    if r > 24:
        ticks = np.linspace(0, r-1, 13, dtype=int)
    else:
        ticks = np.arange(r)
    axW.set_xticks(ticks)

    # 图例（使用自定义 Patch，保持颜色与上面一致）
    handles = [
        patches.Patch(facecolor=c_pos, edgecolor=e_pos, label="positive $w_k$"),
        patches.Patch(facecolor=c_neg, edgecolor=e_neg, label="negative $w_k$")
    ]
    axW.legend(handles=handles, ncols=2, loc="upper right", frameon=False)

    # 收尾：紧凑排版
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 非阻塞 or 阻塞
    if mode == "block":
        plt.show()
    else:
        plt.show(block=False)
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass
        plt.pause(pause_s)


# vectors, weights, r, order = decompose_ising_coupling_with_rank(J=J_Coupling, tol=1e-10, return_full=False)  # vectors:(N,r), weights:(r,) 列向量与权重的最大也就 r
def decompose_ising_coupling_with_rank(J, tol=1e-10, return_full=True):
    """
    对称实矩阵 J ∈ R^{N×N} 的秩-1分解，并按 |weight| 降序排列：
        J ≈ Σ_{k=0}^{r-1}  w_k · (v_k v_k^T)

    约定与原版一致：
      - 先做特征分解 J = U Λ U^T（eigh，保证实正交）
      - 对每个特征向量 q_k 按 max-abs 归一化： v_k = q_k / max(|q_k|)
      - 权重定义： w_k = λ_k · (max(|q_k|))^2
      - 然后按 |w_k| 从大到小排序

    参数
    ----
    J : (N, N) np.ndarray, 对称实矩阵
    tol : float, 判零阈值（同时用于特征值与 max(|q|)）
    return_full : bool
        True  -> 返回 N×N 的 vectors 和长度 N 的 weights（后面全 0）
        False -> 只返回前 r 列 / 前 r 个权重

    返回
    ----
    vectors : np.ndarray
        若 return_full=True : 形状 (N, N)，前 r 列有效，之后全 0
        若 return_full=False: 形状 (N, r)
    weights : np.ndarray
        若 return_full=True : 形状 (N,)，前 r 个有效，之后全 0
        若 return_full=False: 形状 (r,)
    rank_r : int
        有效秩 r（|λ| > tol）
    order : np.ndarray
        排序后的特征索引（按 |weight| 降序），长度 r
    """
    J = np.asarray(J)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square matrix")
    if not np.allclose(J, J.T, atol=tol):
        raise ValueError("J must be a real symmetric matrix")

    N = J.shape[0]

    # eigh: λ 升序；U 的列为特征向量
    eigenvals, eigenvecs = np.linalg.eigh(J)

    # 过滤掉 |λ| 很小的特征值（数值秩）
    mask = np.abs(eigenvals) > tol
    lam = eigenvals[mask]          # (r,)
    U   = eigenvecs[:, mask]       # (N, r)
    r   = lam.size

    if r == 0:
        # 退化情形：全零矩阵
        if return_full:
            return np.zeros((N, N), dtype=float), np.zeros(N, dtype=float), 0, np.array([], dtype=int)
        else:
            return np.zeros((N, 0), dtype=float), np.zeros((0,), dtype=float), 0, np.array([], dtype=int)

    # 计算按 max-abs 归一化后的向量与对应权重
    maxabs = np.max(np.abs(U), axis=0)         # (r,)
    # 避免除 0
    safe = maxabs > tol
    v = np.zeros_like(U)                       # (N, r)
    w = np.zeros((r,), dtype=float)

    v[:, safe] = U[:, safe] / maxabs[safe]     # 每列归一化到 max|v|=1
    w[safe]    = lam[safe] * (maxabs[safe]**2)
    # 如果某列 maxabs≈0，则该列向量与权重都置 0

    # 按 |w| 降序排序
    order = np.argsort(-np.abs(w))             # (r,)
    v = v[:, order]
    w = w[order]

    if return_full:
        vectors = np.zeros((N, N), dtype=float)
        weights = np.zeros((N,), dtype=float)
        vectors[:, :r] = v
        weights[:r]    = w
        return vectors, weights, r, order
    else:
        return v, w, r, order


# 输入实对称矩阵 J，进行分解；return vectors, weights
# 分解为 rank(J) 个秩为 1 的矩阵，返回的 vectors 中的第 k 列以及 weights 中的第 k 个权重，weight_k*np.outer(v_k,v_k)
# 输出 type(vectors)=<class 'numpy.ndarray'>, vectors.shape=(10, 10),vectors.dtype=float64,
# type(weights)=<class 'numpy.ndarray'>, weights.shape=(10,),weights.dtype=float64
# 这里面的实数通过 DPM 编码成两个纯相位图后一个相位板[0~2pi]了，更何况还要叠加原本的自旋spins的状态

def decompose_ising_coupling(J):
    """
    Decompose symmetric coupling matrix J (N×N) into:
        J = sum_{i=0}^{r-1} weights[i] * (vectors[:, i] @ vectors[:, i].T)

    Returns:
        vectors: N×N np.ndarray, each column is a normalized vector (max|v|=1),
                 only first r=rank(J) columns are non-zero.
        weights: (N,) np.ndarray, first r entries are non-zero weights, rest are 0.
    """
    if not np.allclose(J, J.T, atol=1e-10):
        raise ValueError("J must be a real symmetric matrix!")

    N = J.shape[0]
    eigenvals, eigenvecs = np.linalg.eigh(J)

    tol = 1e-10
    mask = np.abs(eigenvals) > tol
    eigenvals = eigenvals[mask]
    eigenvecs = eigenvecs[:, mask]
    r = len(eigenvals)  # effective rank

    # Initialize output arrays
    vectors = np.zeros((N, N), dtype=np.float64)
    weights = np.zeros(N, dtype=np.float64)

    for i in range(r):
        lam = eigenvals[i]
        q = eigenvecs[:, i]  # shape (N,)
        max_abs = np.max(np.abs(q))

        if max_abs < tol:
            v = q.copy()
            w = 0.0
        else:
            v = q / max_abs
            w = lam * (max_abs ** 2)

        vectors[:, i] = v  # assign to i-th column
        weights[i] = w

    return vectors, weights



# 输入为 J_mat = ising_2d_coupling(L, J=1.0, pbc=True)
# 返回的是 (100, 100) float64 <class 'numpy.ndarray'>
# A simple simulation of the Ising model
def ising_2d_coupling(L: int, J: float = 1.0, pbc: bool = True):
    """
    返回 L*L × L*L 的耦合矩阵 J_mat（numpy 二维数组）
    L: 线性尺寸（10 表示 10×10 格子）
    J: 耦合强度，默认 1.0
    pbc: True  用周期性边界
         False 用开放边界（边缘邻居不存在）
    """
    N = L * L
    J_mat = np.zeros((N, N), dtype=float)

    for r in range(L):
        for c in range(L):
            i = r * L + c
            # 四个邻居
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if pbc:
                    rr %= L
                    cc %= L
                else:
                    if rr < 0 or rr >= L or cc < 0 or cc >= L:
                        continue
                j = rr * L + cc
                J_mat[i, j] = J
    return J_mat.astype(np.float32)


def ising_coupling_rect(
    Lx: int,
    Ly: int,
    Jx: float = 1.0,             # x(行)方向最近邻耦合
    Jy: float = 1.0,             # y(列)方向最近邻耦合
    pbc_x: bool = True,          # x方向是否周期
    pbc_y: bool = True,          # y方向是否周期
    antiperiodic_x: bool = False,# x方向反周期（跨 x 边界的键取负号）
    antiperiodic_y: bool = False,# y方向反周期
    J2: float = 0.0,             # 对角下一近邻耦合
    index_order: str = "col",    # 'col' 列优先；'row' 行优先
    dtype=np.float32
):
    """
    生成 (Lx*Ly)×(Lx*Ly) 的耦合矩阵。index_order 决定二维(r,c)如何映射到线性下标 i：
      - 'col'：i = c*Lx + r   （一列一列排：列快变、行慢变）
      - 'row'：i = r*Ly + c   （一行一行排：行快变、列慢变）
    """
    assert index_order in ("col", "row")
    N = Lx * Ly
    J_mat = np.zeros((N, N), dtype=dtype)

    if index_order == "col":
        idx = lambda r, c: c * Lx + r
    else:
        idx = lambda r, c: r * Ly + c

    # 最近邻 + 可选对角下一近邻
    for r in range(Lx):
        for c in range(Ly):
            i = idx(r, c)

            # x 方向（上下）
            for dr, J_here in [(-1, Jx), (1, Jx)]:
                rr = r + dr
                wrap_x = False
                if rr < 0 or rr >= Lx:
                    if not pbc_x:
                        continue
                    wrap_x = True
                    rr %= Lx
                j = idx(rr, c)
                Jij = J_here
                if wrap_x and antiperiodic_x:
                    Jij = -Jij
                J_mat[i, j] += Jij

            # y 方向（左右）
            for dc, J_here in [(-1, Jy), (1, Jy)]:
                cc = c + dc
                wrap_y = False
                if cc < 0 or cc >= Ly:
                    if not pbc_y:
                        continue
                    wrap_y = True
                    cc %= Ly
                j = idx(r, cc)
                Jij = J_here
                if wrap_y and antiperiodic_y:
                    Jij = -Jij
                J_mat[i, j] += Jij

            # 对角下一近邻（J2）
            if J2 != 0.0:
                for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    rr, cc = r + dr, c + dc
                    wrap_x = wrap_y = False
                    if rr < 0 or rr >= Lx:
                        if not pbc_x:
                            continue
                        wrap_x = True
                        rr %= Lx
                    if cc < 0 or cc >= Ly:
                        if not pbc_y:
                            continue
                        wrap_y = True
                        cc %= Ly
                    j = idx(rr, cc)
                    Jij = J2
                    # 反周期：跨越哪个方向就在哪个方向取负；若两个方向都跨越，符号翻两次又回正
                    if (wrap_x and antiperiodic_x) ^ (wrap_y and antiperiodic_y):
                        Jij = -Jij
                    J_mat[i, j] += Jij

    return J_mat



# ==================== Example & Verification ====================
if __name__ == "__main__":
    # J = np.array([
    #     [1, 8 / 9, -1 / 2],
    #     [8 / 9, 1, -4 / 5],
    #     [-1 / 2, -4 / 5, 1]
    # ])
    # J=np.random.randn(10, 10)
    # J=J+J.T
    # J=np.random.randn(10, 1)
    # print(J)
    # J=J @ J.T
    # print("Original coupling matrix J:")
    # print(J)
    #
    # vectors, weights = decompose_ising_coupling(J)
    # print(rf"type(vectors)={type(vectors)},type(weights)={type(weights)},vectors.shape={vectors.shape},vectors.dtype={vectors.dtype},weights.shape={weights.shape},weights.dtype={weights.dtype}")
    # N = J.shape[0]
    # r = np.sum(np.abs(weights) > 1e-10)
    # print(f"\nDecomposed into {r} non-zero rank-1 terms (out of {N} columns):")
    #
    # # Reconstruct J
    # J_rec = np.zeros_like(J)
    # for i in range(N):
    #     if abs(weights[i]) > 1e-12:
    #         v = vectors[:, i].reshape(-1, 1)
    #         J_rec += weights[i] * (v @ v.T)
    #         print(f"\n--- Term {i + 1} ---")
    #         print(f"Weight = {weights[i]:.6f}")
    #         print(f"Vector = {vectors[:, i]}")
    #
    # error = J - J_rec
    # print("\n" + "=" * 50)
    # print("Verification:")
    # print("Reconstructed J_rec =")
    # print(J_rec)
    # print("\nDifference J - J_rec =")
    # print(error)
    # print(f"\nReconstruction error (Frobenius norm): {np.linalg.norm(error):.2e}")
    #
    # # Show full vectors matrix and weights
    # print("\nFull vectors matrix (N×N):")
    # print(vectors)
    # print("\nFull weights vector (length N):")
    # print(weights)

    # 例如标准二维伊辛模型，每个自旋仅与上下左右 4 个最近邻相互作用，耦合常数通常取为 J > 0 （铁磁）或 J < 0 （反铁磁）。
    # ---------------- demo ----------------
    L = 10
    J_mat = ising_2d_coupling(L, J=1.0, pbc=True)
    print(J_mat.shape, J_mat.dtype, type(J_mat))  # (100, 100)
    print(J_mat[:5, :5])  # 左上角 5×5 预览
    print(J_mat[0,:])

    J=J_mat

    vectors, weights, r, order = decompose_ising_coupling_with_rank(J, return_full=False)  # vectors:(N,r), weights:(r,) 列向量与权重的最大也就 r

    print(rf"r={r},order={order},weights={weights}")
    k = r+10
    J_approx = sum(weights[i] * np.outer(vectors[:, i], vectors[:, i]) for i in range(k))
    err = np.linalg.norm(J - J_approx)
    print(f"top-{k} approximation error: {err:.3e}")

    vectors, weights = decompose_ising_coupling(J)
    print(
        rf"type(vectors)={type(vectors)},type(weights)={type(weights)},vectors.shape={vectors.shape},vectors.dtype={vectors.dtype},weights.shape={weights.shape},weights.dtype={weights.dtype}")
    N = J.shape[0]
    r = np.sum(np.abs(weights) > 1e-10) # 非 0
    print(f"\nDecomposed into {r} non-zero rank-1 terms (out of {N} columns):")


    # Reconstruct J
    J_rec = np.zeros_like(J)
    for i in range(N):
        if abs(weights[i]) > 1e-12:
            v = vectors[:, i].reshape(-1, 1)
            J_rec += weights[i] * (v @ v.T)
            print(f"\n--- Term {i + 1} ---")
            print(f"Weight = {weights[i]:.6f}")
            print(f"Vector = {vectors[:, i]}")

    error = J - J_rec
    print("\n" + "=" * 50)
    print("Verification:")
    print("Reconstructed J_rec =")
    print(J_rec)
    print("\nDifference J - J_rec =")
    print(error)
    print(f"\nReconstruction error (Frobenius norm): {np.linalg.norm(error):.2e}")

    # Show full vectors matrix and weights
    print("\nFull vectors matrix (N×N):")
    print(vectors)
    print("\nFull weights vector (length N):")
    print(weights)