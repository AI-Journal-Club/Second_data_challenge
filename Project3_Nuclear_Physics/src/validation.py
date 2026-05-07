import matplotlib.pyplot as plt
import numpy as np
import torch

from consts import KE_factor

def validate(r_eval, u_scipy, R_scipy, E_scipy, model):
    model.eval()

    # Ensure r has shape (num_samples, 1) for consistency with the network's expected input
    r = torch.tensor(r_eval, device=model.device,
                     dtype=model.dtype, requires_grad=True).reshape(-1, 1)

    u = model(r)
    du = torch.autograd.grad(
        u, r, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(
        du, r, grad_outputs=torch.ones_like(du), create_graph=False)[0]

    v = model.potential(r)
    rf = r.squeeze()
    uf = u.squeeze()
    duf = du.squeeze()

    norm = torch.trapezoid(uf ** 2, rf)
    norm_loss = (norm - 1) ** 2

    kinetic = KE_factor * torch.trapezoid(duf ** 2, rf)
    potential = torch.trapezoid(v.squeeze() * uf ** 2, rf)
    rayleigh_energy = (kinetic + potential) / norm

    residual = - KE_factor * d2u.squeeze() + v.squeeze() * uf - \
        model.energy.item() * uf
    pde_loss = torch.mean(residual ** 2)

    u_np = uf.detach().cpu().numpy()
    r_np = rf.detach().cpu().numpy()
    u_norm = u_np / np.sqrt(norm.item())

    if np.mean(u_norm[:20]) * np.mean(u_scipy[:20]) < 0:
        u_norm = -u_norm
    l2 = np.sqrt(np.trapz((u_norm - u_scipy) ** 2, r_np))

    metrics = {
        'rayleigh_energy': rayleigh_energy.item(),
        'norm_loss': norm_loss.item(),
        'pde_loss': pde_loss.item(),
        'u_rmax': uf[-1].item(),
        'l2': l2,
        'learned_energy': model.energy.item(),
        'energy_error': abs(model.energy.item() - E_scipy),
    }
    return metrics, u_norm, u_norm/r_np


def print_metrics(metrics):
    for key, value in metrics.items():
        if key == 'label':
            print(value)
        elif isinstance(value, float):
            print(f'  {key}: {value:.8g}')
