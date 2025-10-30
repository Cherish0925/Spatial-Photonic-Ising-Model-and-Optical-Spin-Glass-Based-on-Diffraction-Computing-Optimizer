# 输入一个复振幅，即可得到哈密顿量
import torch
from typing import Union, Tuple, Optional
import numpy as np

# ------------------------------ 工具：中心窗口强度向量 ------------------------------
def _parse_win(win):
    # 支持 int 或 (h,w)；返回 (wh, ww)
    if isinstance(win, int):
        return win, win
    if isinstance(win, (tuple, list)) and len(win) == 2:
        return int(win[0]), int(win[1])
    raise ValueError("win must be an int or a (h,w) tuple/list.")

def central_intensity_vector(E_field: torch.Tensor,
                             win: Union[int, Tuple[int, int]] = 10) -> torch.Tensor:
    """
    E_field: [r,H,W] complex
    返回每幅中心窗口 |E|^2 求和，shape [r]
    """
    wh, ww = _parse_win(win)

    r, H, W = E_field.shape
    ch, cw = H // 2, W // 2
    rh, rw = wh // 2, ww // 2
    ys, ye = ch - rh, ch - rh + wh
    xs, xe = cw - rw, cw - rw + ww

    I = torch.abs(E_field) ** 2               # [r,H,W]
    I_patch = I[:, ys:ye, xs:xe]              # [r,wh,ww]
    I_vec = I_patch.sum(dim=(-2, -1))         # [r]
    return I_vec


# ------------------------------ 哈密顿量 ------------------------------
def hamiltonian_from_Eout(E_out: torch.Tensor,
                          weights: Union[np.ndarray, torch.Tensor],
                          win: Union[int, Tuple[int, int]] = 10,
                          device: Optional[torch.device] = None) -> torch.Tensor:
    """
    H = - <weights, sum_center(|E|^2)>
    返回标量张量（在 device 上）
    """
    if device is None:
        device = E_out.device

    I_vec = central_intensity_vector(E_out, win=win).to(device=device, dtype=torch.float32)  # [r]
    if isinstance(weights, np.ndarray):
        w = torch.from_numpy(weights).to(device=device, dtype=torch.float32)
    else:
        w = weights.to(device=device, dtype=torch.float32)

    r_ = min(I_vec.numel(), w.numel())
    H = - torch.dot(w[:r_], I_vec[:r_])
    return H