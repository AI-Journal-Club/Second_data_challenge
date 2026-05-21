# Welcome to Nuclear Physics Project

## Team members
Phong Dang, Xiaoliang Wan, Evander Espinoza, Jake Gilson, Skye Strain, Dana Browne

## Problem statement
We aim to build a PDE solver for solving simple Hamiltonian of atomic nuclei. As a starting point, we consider the simplest system, a deuteron which consists of one proton and one neutron. The advantage is that its Hamiltonian

$$
H = \frac{\mathbf{p}_p^2}{2m_p} + \frac{\mathbf{p}_n^2}{2m_n} + V(|\mathbf{r}_p - \mathbf{r}_n|)
$$

can be reformulated into a one-body Hamiltonian with a reduced mass $m = m_p m_n /(m_p + m_n)$ as

$$
H = \frac{\mathbf{p}^2}{2m} + V(r).
$$

Here the kinetic energy of the center of mass has been set to zero to remove spurity. The Schrodinger equation is

$$
 \frac{-\hbar^2}{2m} \nabla^2 \Psi + (V(r) - E)\Psi = 0.
$$

By separating the wave function into a radial and an angular part, $\Psi(r,\theta,\phi) = R(r) Y(\theta,\phi)$, where the normalized angular wave function $Y$ is a spherical harmonics

$$
Y_{lm}(\theta,\phi) = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}} e^{im\phi} P_{lm}(\cos\theta),
$$

we are left with the following equation

$$
\frac{1}{R} \frac{d}{dr}\left(r^2 \frac{dR}{dr} \right) - \frac{2mr^2}{\hbar^2}(V(r) - E) = l(l+1).
$$

We can change variable to simplify things by introducing $u(r) = r R(r)$, then we have the radial equation

$$
\frac{-\hbar^2}{2 m} \frac{d^2 u}{dr^2} + \left[V + \frac{\hbar^2}{2 m} \frac{l(l+1)}{r^2} - E \right] u = 0.
$$

Normalization of the wave function is equivalent to $\int_0^\infty |u|^2 dr = 1$.

Our goal is to set up a PDE solver that can be used to solve the radial Schrodinger equation for different phenomenological potentials $V(r)$. Some simple choices are

1. Square well with parameters $V_0=59.66$ MeV and $R_0 = 1.5$ fm

$$
V(r) = \begin{cases} 
-V_0 & \text{if } r < R_0 \\
0 & \text{if } r \ge R_0 
\end{cases}
$$

2. Gaussian central with parameters $V_0=70.92$ MeV and $R = 1.5$ fm

$$
V(r) = -V_0 e^{-r^2/R^2}
$$

3. Yukawa potential with parameters $g=8.22$ MeV$^{1/2}$ and $\mu = 1/a$ with $a=1.5$ fm

$$
V(r) = -g^2 \frac{e^{-\mu r}}{r}
$$

## Phase shift fitting

The potential parameters should be constrained by experimental nucleon-nucleon scattering data rather than set by hand. The file `include/phase_shifts.py` provides utilities for this.

### Scattering setup

For positive energy $E > 0$, the radial Schrödinger equation has the same form as the bound-state case but with $E > 0$. Far from the scatterer where $V(r) \approx 0$, the $l = 0$ solution takes the asymptotic form

$$
u(r) \xrightarrow{r \to \infty} A \sin(kr + \delta_0), \qquad k = \frac{\sqrt{2mE}}{\hbar}
$$

The phase shift $\delta_0$ is extracted by matching the numerical solution to this free-particle form at a large matching radius $r_m$:

$$
\delta_0 = \text{atan2}\!\left(k\, u(r_m),\, u'(r_m)\right) - k\, r_m \pmod{\pi}
$$

By Levinson's theorem, $\delta_0(k \to 0) = n_b \pi$ where $n_b$ is the number of bound states. For the deuteron $n_b = 1$.

### Fitting objective

Given a parametric potential $V(r;\theta)$, the parameters $\theta$ are found by minimising the chi-squared

$$
\chi^2(\theta) = \sum_i \left[\delta_0(k_i;\theta) - \tilde\delta(k_i)\right]^2
$$

where $\tilde\delta(k_i)$ are the experimental phase shifts (in degrees) at wavenumber $k_i$. This is a single-channel ($l = 0$) fit.

### Kinematic conversions

Experimental data is often given in lab-frame or momentum variables. The conversions to relative wavenumber $k$ (fm$^{-1}$) are:

- **Relative kinetic energy** $T_{rel}$ (MeV): $\quad k = \sqrt{2m T_{rel}}\, /\, \hbar$
- **Relative momentum** $p_{rel}$ (MeV/$c$): $\quad k = p_{rel} / \hbar$

See [`test/Phase_shifts.ipynb`](test/Phase_shifts.ipynb) for a worked example with all three potentials.

**Note:** The initial parameters must support the same number of bound states as the target data. Use `bounds` to prevent the optimizer from crossing a bound-state threshold, which would flip the phase-shift branch and break the chi-squared landscape.
