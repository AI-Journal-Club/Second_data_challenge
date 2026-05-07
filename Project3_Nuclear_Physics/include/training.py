import numpy as np
from soap import SOAP

from losses import Rayleigh_loss
from losses import grad_norm, schedule_caps, balanced_weight, Rayleigh_PDE_loss
from sampling import set_seed

# Train model with Rayleigh loss only


def train_Rayleigh(model, cfg, steps=10000, lr=3e-3, seed=42, stop=1e-6):
    set_seed(seed)

    opt = SOAP(model.parameters(), lr=lr, betas=(0.95, 0.95),
               weight_decay=0.0, precondition_frequency=10)
    params = [p for p in model.parameters() if p.requires_grad]

    total_losses = []
    history = []

    for step in range(1, steps + 1):
        loss, terms = Rayleigh_loss(model, cfg)
        total_losses.append(loss.item())
        model.energy.data = terms['rayleigh_energy'].detach().reshape(
            model.energy.shape)

        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Logging
        if step == 1 or step % 500 == 0 or step == steps:
            row = {'step':          step,
                   'loss':          loss.item(),
                   'rayleigh':      terms['rayleigh_energy'].item(),
                   'norm_loss':     terms['norm_loss'].item(),
                   'boundary_loss': terms['boundary_loss'].item(),
                   'E':       model.energy.item(),
                   }
            history.append(row)
            print(row)

        # Early stopping
        if step > 3000 and abs(np.mean(total_losses[-200:]) - loss.item()) < stop:
            print(f"Early stopping at step {step}")
            break

    return history


def train_Rayleigh_PDE(model, cfg, steps=10000, lr=3e-3, seed=42, stop=1e-6):
    set_seed(seed)

    opt = SOAP(model.parameters(), lr=lr, betas=(0.95, 0.95),
               weight_decay=0.0, precondition_frequency=10)
    params = [p for p in model.parameters() if p.requires_grad]
    lambda_pde = 0.0
    lambda_match = 0.0
    total_losses = []
    history = []

    for step in range(1, steps + 1):
        pde_cap, match_cap = schedule_caps(step, cfg)

        loss, terms = Rayleigh_PDE_loss(model, cfg, lambda_pde, lambda_match)
        total_losses.append(loss.item())

        # Update lambda_pde and lambda_match before backward
        if step == 1 or step % cfg.grad_balance_every == 0:
            g_ray = grad_norm(terms['rayleigh_energy'], params)
            g_pde = grad_norm(terms['pde_loss'],        params)
            g_match = grad_norm(terms['match_loss'],      params)
            next_pde = balanced_weight(
                g_ray, g_pde,   pde_cap,   cfg.grad_balance_pde_scale,   cfg)
            next_match = balanced_weight(
                g_ray, g_match, match_cap, cfg.grad_balance_match_scale, cfg)
            ema = cfg.grad_balance_ema
            lambda_pde = next_pde if step == 1 else ema * \
                lambda_pde + (1.0 - ema) * next_pde
            lambda_match = next_match if step == 1 else ema * \
                lambda_match + (1.0 - ema) * next_match
        lambda_pde = min(pde_cap,   max(0.0, lambda_pde))
        lambda_match = min(match_cap, max(0.0, lambda_match))

        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Logging
        if step == 1 or step % 500 == 0 or step == steps:
            row = {
                'step':          step,
                'loss':          loss.item(),
                'rayleigh':      terms['rayleigh_energy'].item(),
                'pde_loss':      terms['pde_loss'].item(),
                'match_loss':    terms['match_loss'].item(),
                'norm_loss':     terms['norm_loss'].item(),
                'boundary_loss': terms['boundary_loss'].item(),
                'E':       model.energy.item(),
                'lambda_pde':    lambda_pde,
                'lambda_match':  lambda_match,
            }
            history.append(row)
            print(row)

        # Early stopping
        if step > 3000 and abs(np.mean(total_losses[-200:]) - loss.item()) < stop:
            print(f"Early stopping at step {step}")
            break

    return history
