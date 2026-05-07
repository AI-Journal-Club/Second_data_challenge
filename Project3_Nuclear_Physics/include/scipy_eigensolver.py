from consts import KE_factor
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def solve_1channel(V, r):
    """
    Solve the radial Schrödinger equation for the ground state.

    Parameters
    ----------
    V : callable, V(r) returning potential in MeV
    r : np.ndarray, radial grid in fm

    Returns
    -------
    E : float, ground state energy in MeV
    u : np.ndarray, reduced wavefunction u(r), normalized
    R : np.ndarray, radial wavefunction R(r) = u(r)/r
    """
    r_min, r_max = r[0], r[-1]

    # ---- ODE system ----
    def ode(r, y, e):
        u, up = y
        d2u = (V(r) - e) * u / KE_factor
        return [up, d2u]

    # ---- Shooting function ----
    def shoot(e):
        sol = solve_ivp(
            lambda r, y: ode(r, y, e),
            [r_min, r_max],
            [0.0, 1.0],
            t_eval=[r_max],
            rtol=1e-6, atol=1e-8
        )
        return sol.y[0, -1]

    # ---- Auto-bracket: scan for first sign change ----
    V_min = min(V(r))
    E_scan = np.linspace(V_min * 0.99, -1e-3, 500)
    u_vals = [shoot(e) for e in E_scan]

    E_left, E_right = None, None
    for i in range(len(u_vals) - 1):
        if u_vals[i] * u_vals[i+1] < 0:
            E_left = E_scan[i]
            E_right = E_scan[i+1]
            break

    if E_left is None:
        raise ValueError(
            "No bound state found. Try deeper/wider potential or larger r_max.")

    # ---- Find eigenenergy ----
    E = brentq(shoot, E_left, E_right, xtol=1e-8)
    print(f"Ground state energy E = {E:.6f} MeV")

    # ---- Solve for full wavefunction ----
    sol = solve_ivp(
        lambda r, y: ode(r, y, E),
        [r_min, r_max],
        [0.0, 1.0],
        t_eval=r,
        rtol=1e-6, atol=1e-8
    )
    u = sol.y[0]

    # ---- Normalize ----
    norm = np.trapz(u**2, r)
    u /= np.sqrt(norm)
    R = u / r

    return E, u, R
