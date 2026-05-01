from consts import KE_factor
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def solve_radial_schrodinger(V, r_min, r_max, n_points=1000, const=20.735):
    """
    Solve the radial Schrödinger equation for the ground state.
    
    Parameters
    ----------
    V       : callable, V(r) returning potential in MeV
    r_min   : float, inner boundary in fm
    r_max   : float, outer boundary in fm
    n_points: int, number of grid points for wavefunction
    const   : float, ℏ²/2m in MeV·fm² (default: neutron)
    
    Returns
    -------
    E_sol   : float, ground state energy in MeV
    r_eval  : np.ndarray, radial grid
    u       : np.ndarray, reduced wavefunction u(r), normalized
    psi     : np.ndarray, radial wavefunction Ψ(r) = u(r)/r
    """
    # ---- ODE system ----
    def ode(r, y, e):
        u, up = y
        d2u = (V(r) - e) * u / const
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
    V_min = min(V(np.linspace(r_min, r_max, 200)))  # well bottom
    E_scan = np.linspace(V_min * 0.99, -1e-3, 500)
    u_vals = [shoot(e) for e in E_scan]

    # Find first sign change = ground state bracket
    E_left, E_right = None, None
    for i in range(len(u_vals) - 1):
        if u_vals[i] * u_vals[i+1] < 0:
            E_left  = E_scan[i]
            E_right = E_scan[i+1]
            break

    if E_left is None:
        raise ValueError("No bound state found. Try deeper/wider potential or larger r_max.")

    # ---- Find eigenenergy ----
    E_sol = brentq(shoot, E_left, E_right, xtol=1e-8)
    print(f"Ground state energy E = {E_sol:.6f} MeV")

    # ---- Solve for full wavefunction ----
    r_eval = np.linspace(r_min, r_max, n_points)
    sol = solve_ivp(
        lambda r, y: ode(r, y, E_sol),
        [r_min, r_max],
        [0.0, 1.0],
        t_eval=r_eval,
        rtol=1e-6, atol=1e-8
    )
    u = sol.y[0]

    # ---- Normalize ----
    norm = np.trapezoid(u**2, r_eval)
    u /= np.sqrt(norm)
    psi = u / r_eval

    return E_sol, r_eval, u, psi


# ---- Plot helper ----
def plot_wavefunction(r_eval, u, psi, E_sol, title="Radial Schrödinger", skip=5):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r_eval,       u,        color='green', lw=2, label='$u(r)$')
    ax.plot(r_eval[skip:], psi[skip:], color='blue',  lw=2, label=r'$\Psi(r)=u(r)/r$')
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('Wavefunction')
    ax.set_title(f'{title}\nE = {E_sol:.4f} MeV')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
