from dataclasses import dataclass
import numpy as np
import random
import torch

# Set fixed seed for reproducibility


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Configuration options for training


@dataclass
class TrainConfig:
    n_pde: int = 3000
    n_origin: int = 500
    n_interface: int = 600
    n_rayleigh: int = 3000
    origin_width: float = 0.5
    interface_width: float = 0.2
    lambda_rayleigh: float = 1.0
    lambda_norm: float = 10.0
    lambda_boundary: float = 10.0
    lambda_pde_cap: float = 0.8
    lambda_match_cap: float = 0.5
    warmup_steps: int = 2000
    ramp_steps: int = 3000
    grad_balance_every: int = 25
    grad_balance_ema: float = 0.5
    grad_balance_pde_scale: float = 0.5
    grad_balance_match_scale: float = 0.5
    grad_balance_min: float = 1e-4
    grad_balance_eps: float = 1e-12


def sample_pde_points(cfg, device, dtype, r_min, r_max, R=None):
    # Sample the full interval
    pts = [r_min + (r_max - r_min) * torch.rand(cfg.n_pde,
                                                1, device=device, dtype=dtype)]

    # Sample near the origin
    pts.append(r_min + torch.rand(cfg.n_origin, 1,
               device=device, dtype=dtype) * cfg.origin_width)

    # Sample near discontinuity
    if R is not None:
        lo = R - cfg.interface_width
        hi = R + cfg.interface_width
        pts.append(lo + (hi - lo) * torch.rand(cfg.n_interface,
                   1, device=device, dtype=dtype))

    # Assemble
    r = torch.cat(pts, dim=0)
    r.requires_grad_(True)
    return r


def sample_rayleigh_points(cfg, device, dtype, r_min, r_max):
	# Sample the full interval
    r = r_min + (r_max - r_min) * torch.rand(cfg.n_rayleigh,
                                             1, device=device, dtype=dtype)
    r.requires_grad_(True)
    # the 2nd output approximates the interval size
    return r, (r_max - r_min) / cfg.n_rayleigh
