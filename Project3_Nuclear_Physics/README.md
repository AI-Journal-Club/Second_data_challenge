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

1. Square well with parameters $V_0, R$

$$
V(r) = \begin{cases} 
-V_0 & \text{if } r < R \\
0 & \text{if } r \ge R 
\end{cases}
$$

2. Gaussian central with parameters $V_0, R$

$$
V(r) = -V_0 e^{-r^2/R^2}
$$

3. Yukawa potential with parameters $g, \mu$

$$
V(r) = -g^2 \frac{e^{-\mu r}}{r}
$$

## Timeline

### Week 1: 02/20/26
- Determine the starting point
- Set the agenda
- Choose a PDE solver package: [`pinns-torch`](https://github.com/rezaakb/pinns-torch)

### Week 2: 02/27/26
-  Xiaoliang presents an introduction on PINNs

### Week 3: 03/06/26
- Skye presents the tutorial notebook from `pinns-torch`
- Jake presents an example `pinns-torch`

### Week 4: 03/13/26
- Spring break

### Week 5: 03/20/26
- Formulate loss functions

### Week 6 and forward: 
- Implementation
- Xiaoliang suggested [`SOAP`](https://github.com/nikhilvyas/SOAP/tree/main) optimizer, which can be easily integrated with `torch` and seems to outperform the common Adam optimizer.
