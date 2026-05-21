"""
Phase shift utilities for nuclear potential fitting.

Scattering setup (l = 0 channel)
---------------------------------
For positive energy E > 0, the radial Schrödinger equation is

    -ℏ²/(2m) d²u/dr² + V(r) u = E u,    u(0) = 0

with wavenumber k = sqrt(2m E) / ℏ.  Far from the scatterer (V ≈ 0)
the solution takes the asymptotic form

    u(r) → A sin(k r + δ₀)

The l = 0 phase shift δ₀ is extracted by matching the numerical solution
to this free-particle form at a matching radius r_match where V ≈ 0:

    δ₀ = atan2(k · u(r_match), u'(r_match)) − k · r_match     (mod π)

Phase shifts at successive k values are unwrapped to ensure continuity.

Fitting objective
-----------------
Given a parametric potential V(r; θ), the parameters θ are found by
minimising the chi-squared

    χ²(θ) = Σᵢ [δ₀(kᵢ; θ) − δ̃(kᵢ)]²

where δ̃(kᵢ) is the experimental phase shift (in degrees) at wavenumber kᵢ.
This is a single-channel (l = 0) fit; each potential shape is fitted
independently.

Kinematic conversions
---------------------
Experimental data often comes in lab-frame or momentum variables.
The conversions to relative wavenumber k (fm⁻¹) are:

    T_rel  (relative / CM kinetic energy, MeV) :
        k = sqrt(2 m T_rel) / ℏ

    p_rel  (relative momentum, MeV/c = MeV in natural units ℏc = 197.327 MeV·fm) :
        k = p_rel / ℏ

    T_lab  (lab kinetic energy of one nucleon hitting a stationary nucleon, MeV) :
        T_rel = m_n / (m_p + m_n) * T_lab   (non-relativistic, proton beam convention)
        k = sqrt(2 m T_rel) / ℏ
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from consts import KE_factor, m, hbar, mp, mn


# ---------------------------------------------------------------------------
# Kinematic conversion
# ---------------------------------------------------------------------------

def convert_to_wavenumber(values, input_type='T_rel'):
    """
    Convert kinematic variables to relative wavenumber k (fm⁻¹).

    Parameters
    ----------
    values : array-like
        Energies in MeV  (if input_type='T_rel' or 'T_lab') or
        momenta in MeV/c (if input_type='p_rel').
    input_type : {'T_rel', 'p_rel', 'T_lab'}
        'T_rel' : relative (CM) kinetic energy in MeV.
                  k = sqrt(2 m T_rel) / ℏ
        'p_rel' : relative momentum in MeV/c (natural units, ℏc = 197.327 MeV·fm).
                  k = p_rel / ℏ,  T_rel = ℏ² k² / (2m)
        'T_lab' : lab kinetic energy in MeV of one nucleon hitting a stationary
                  nucleon of the other type (standard np/pp scattering convention).
                  T_rel = m_n / (m_p + m_n) * T_lab  (non-relativistic)
                  k = sqrt(2 m T_rel) / ℏ

    Returns
    -------
    k : np.ndarray, wavenumber in fm⁻¹
    E : np.ndarray, relative kinetic energy in MeV
    """
    values = np.asarray(values, dtype=float)

    if input_type == 'T_rel':
        E = values
        k = np.sqrt(2.0 * m * E) / hbar
    elif input_type == 'p_rel':
        k = values / hbar
        E = KE_factor * k ** 2
    elif input_type == 'T_lab':
        E = mn / (mp + mn) * values   # non-relativistic CM energy
        k = np.sqrt(2.0 * m * E) / hbar
    else:
        raise ValueError(f"input_type must be 'T_rel', 'p_rel', or 'T_lab', got '{input_type!r}'")

    return k, E


# ---------------------------------------------------------------------------
# Phase shift computation
# ---------------------------------------------------------------------------

def _phase_shift_single(V, k, r_min, r_max, r_match):
    """Compute the raw (unwrapped) l=0 phase shift in radians for one k."""
    E = KE_factor * k ** 2  # ℏ² k² / (2m) in MeV

    def ode(r, y):
        u, du = y
        return [du, (V(r) - E) / KE_factor * u]

    sol = solve_ivp(
        ode, [r_min, r_max], [0.0, 1.0],
        t_eval=[r_match], method='RK45',
        rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed at k={k:.4f} fm⁻¹: {sol.message}")

    u  = sol.y[0, -1]
    du = sol.y[1, -1]

    # u(r_match) = A sin(k r_match + δ)  =>  δ = atan2(k u, u') − k r_match
    return np.arctan2(k * u, du) - k * r_match


def compute_phase_shifts(V, k_array, r_min=1e-4, r_max=30.0, r_match=None,
                         degrees=True, n_bound_states=None):
    """
    Compute l=0 phase shifts for an array of wavenumbers.

    k_array should be sorted in ascending order so that np.unwrap can
    maintain continuity across the array.

    Parameters
    ----------
    V : callable, V(r) → float (MeV)
    k_array : array-like, wavenumbers in fm⁻¹ (sorted ascending)
    r_min : float, inner boundary (fm)
    r_max : float, outer boundary (fm)
    r_match : float or None
        Asymptotic matching radius (fm); defaults to 0.9 * r_max.
        Must satisfy V(r_match) ≈ 0.
    degrees : bool
        If True (default), return phase shifts in degrees; else radians.
    n_bound_states : int or None
        If provided, the phase shift array is shifted by an integer multiple
        of π so that δ(k_array[0]) ≈ n_bound_states · π, enforcing Levinson's
        theorem at the lowest supplied k value.  If None, no shift is applied.

    Returns
    -------
    delta : np.ndarray, phase shifts (degrees or radians)
    """
    k_array = np.asarray(k_array, dtype=float)
    if r_match is None:
        r_match = 0.9 * r_max

    raw = np.array([
        _phase_shift_single(V, k, r_min, r_max, r_match)
        for k in k_array
    ])

    # Unwrap to remove artificial π-jumps from atan2 branch cuts
    delta = np.unwrap(raw)

    # Align to Levinson's theorem: δ(k→0) = n_bound_states · π
    if n_bound_states is not None:
        target = n_bound_states * np.pi
        n_shift = round((target - delta[0]) / np.pi)
        delta += n_shift * np.pi

    return np.degrees(delta) if degrees else delta


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------

def fit_potential(V_template, params_init, k_data, delta_data,
                  bounds=None, r_min=1e-4, r_max=30.0, r_match=None,
                  n_bound_states=None, method='L-BFGS-B', options=None):
    """
    Fit potential parameters to phase shift data by minimising

        χ²(θ) = Σᵢ [δ₀(kᵢ; θ) − δ̃(kᵢ)]²

    Works with any potential shape; pass V_template(r, *params) where
    *params are the parameters to fit.

    Important: the initial parameters params_init must yield the same number
    of bound states as the target data.  If the number of bound states
    changes during optimisation, the phase-shift branch can flip and the
    chi-squared landscape becomes discontinuous.  Use bounds to prevent this.

    Parameters
    ----------
    V_template : callable
        V_template(r, *params) → potential in MeV.
        Example for a Gaussian:
            def V_gauss(r, V0, R):
                return -V0 * np.exp(-r**2 / R**2)
    params_init : array-like
        Initial guesses for the parameters.  Must yield n_bound_states
        bound states (see note above).
    k_data : array-like
        Wavenumbers in fm⁻¹, sorted ascending.
    delta_data : array-like
        Target phase shifts in degrees (same ordering as k_data).
    bounds : list of (lo, hi) pairs or None
        Parameter bounds passed to scipy.optimize.minimize.
        Strongly recommended to prevent the potential from losing/gaining
        bound states during optimisation.
    r_min, r_max : float
        Integration limits in fm.
    r_match : float or None
        Asymptotic matching radius (fm); defaults to 0.9 * r_max.
    n_bound_states : int or None
        Number of bound states in the target potential.  If provided,
        Levinson's theorem is used to fix the phase-shift branch at each
        evaluation (δ(k_data[0]) ≈ n_bound_states · 180°), which is more
        robust than aligning to the data's first value.
        For the deuteron: n_bound_states=1.
    method : str
        scipy.optimize.minimize method.
        'L-BFGS-B' (default, supports bounds) or 'Nelder-Mead' (no bounds).
    options : dict or None
        Extra options forwarded to scipy.optimize.minimize.

    Returns
    -------
    params_opt : np.ndarray
        Fitted parameters.
    result : scipy.optimize.OptimizeResult
        Full result object (includes .fun = final χ², .success, .message).
    """
    k_data     = np.asarray(k_data,     dtype=float)
    delta_data = np.asarray(delta_data, dtype=float)

    # Sort by k so that np.unwrap inside compute_phase_shifts is valid
    order = np.argsort(k_data)
    k_sorted     = k_data[order]
    delta_sorted = delta_data[order]

    def chi2(params):
        V = lambda r: V_template(r, *params)
        delta_pred = compute_phase_shifts(
            V, k_sorted, r_min=r_min, r_max=r_max, r_match=r_match,
            degrees=True, n_bound_states=n_bound_states,
        )
        if n_bound_states is None:
            # Fallback: align to data's first point by the nearest 180° shift
            n_shift = round((delta_sorted[0] - delta_pred[0]) / 180.0)
            delta_pred += n_shift * 180.0
        return float(np.sum((delta_pred - delta_sorted) ** 2))

    default_opts = (
        {'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7}
        if method == 'L-BFGS-B'
        else {'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
    )

    result = minimize(
        chi2, np.asarray(params_init, dtype=float),
        method=method, bounds=bounds,
        options=options or default_opts,
    )

    return result.x, result
