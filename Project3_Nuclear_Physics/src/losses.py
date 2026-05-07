import torch

from consts import KE_factor
from sampling import sample_pde_points, sample_rayleigh_points

# Loss with Rayleigh, Norm, and Boundary


def Rayleigh_loss(model, cfg):
	r, w = sample_rayleigh_points(
	    cfg, model.device, model.dtype, model.r_min, model.r_max)
	u = model(r)

	# Norm
	norm = w * torch.sum(u.squeeze() ** 2)
	norm_loss = (norm - 1) ** 2

	# Rayleigh
	du = torch.autograd.grad(u, r, grad_outputs=torch.ones_like(
	    u), create_graph=True, retain_graph=True)[0]

	kinetic = w * KE_factor * torch.sum(du.squeeze() ** 2)
	V = model.potential(r)
	potential = w * torch.sum(V.squeeze() * u.squeeze() ** 2)
	rayleigh_loss = (kinetic + potential) / torch.clamp(norm, min=1e-12)

	# Boundary
	r_boundary = torch.tensor(
	    [[model.r_max]], device=model.device, dtype=model.dtype, requires_grad=True)
	boundary_loss = model(r_boundary).squeeze() ** 2

	# Assemble
	loss = cfg.lambda_rayleigh * rayleigh_loss + cfg.lambda_norm * \
	    norm_loss + cfg.lambda_boundary * boundary_loss

	terms = {'rayleigh_energy': rayleigh_loss,
    		 'norm_loss': norm_loss,
    		 'boundary_loss': boundary_loss,
    		}

	return loss, terms

# Loss with Rayleigh, PDE, matching, Norm, and Boundary
def grad_norm(term, parameters):
	if not term.requires_grad:
		return 0.0

	grads = torch.autograd.grad(term, parameters, retain_graph=True, create_graph=False, allow_unused=True)
	total = torch.zeros((), device=term.device, dtype=term.dtype)

	for g in grads:
		if g is not None:
			total = total + g.detach().pow(2).sum()

	return float(torch.sqrt(total).cpu())

def schedule_caps(step, cfg):
	if step <= cfg.warmup_steps:
		progress = 0.0
	else:
		progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.ramp_steps))

	return progress * cfg.lambda_pde_cap, progress * cfg.lambda_match_cap

def balanced_weight(anchor_norm, term_norm, cap, scale, cfg):
	if cap <= 0.0 or anchor_norm <= cfg.grad_balance_eps or term_norm <= cfg.grad_balance_eps:
		return 0.0
	raw = scale * anchor_norm / (term_norm + cfg.grad_balance_eps)
	return min(cap, max(min(cfg.grad_balance_min, cap), raw))

def Rayleigh_PDE_loss(model, cfg, lambda_pde, lambda_match):
    # Norm
    r_R, w = sample_rayleigh_points(cfg, model.device, model.dtype, model.r_min, model.r_max)
    u_R = model(r_R)
    norm = w * torch.sum(u_R.squeeze() ** 2)
    norm_loss = (norm - 1) ** 2

    # Rayleigh
    du_R = torch.autograd.grad(u_R, r_R, grad_outputs=torch.ones_like(u_R), create_graph=True, retain_graph=True)[0]

    kinetic_R = w * KE_factor * torch.sum(du_R.squeeze() ** 2)
    V_R = model.potential(r_R)
    potential_R = w * torch.sum(V_R.squeeze() * u_R.squeeze() ** 2)
    rayleigh_loss = (kinetic_R + potential_R) / torch.clamp(norm, min=1e-12)

    # Boundary
    r_boundary = torch.tensor([[model.r_max]], device=model.device, dtype=model.dtype, requires_grad=True)
    boundary_loss = model(r_boundary).squeeze() ** 2

    # PDE
    r_pde = sample_pde_points(cfg, model.device, model.dtype, model.r_min, model.r_max, model.R)
    u_pde = model(r_pde)
    du_pde = torch.autograd.grad(u_pde, r_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True, retain_graph=True)[0]
    d2u_pde = torch.autograd.grad(du_pde, r_pde, grad_outputs=torch.ones_like(du_pde), create_graph=True, retain_graph=True)[0]
    residual = - KE_factor * d2u_pde + model.potential(r_pde) * u_pde - model.energy * u_pde
    pde_loss = torch.mean(residual ** 2)

    # Match
    match_loss = (model.energy - rayleigh_loss) ** 2

    # Assemble
    loss = cfg.lambda_rayleigh * rayleigh_loss + \
           lambda_pde * pde_loss + \
           lambda_match * match_loss + \
           cfg.lambda_norm * norm_loss + \
           cfg.lambda_boundary * boundary_loss

    terms = {'rayleigh_energy': rayleigh_loss,
             'pde_loss': pde_loss,
             'match_loss': match_loss,
             'norm_loss': norm_loss,
             'boundary_loss': boundary_loss,
        	}

    return loss, terms
