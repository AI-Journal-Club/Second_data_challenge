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

Here the kinetic energy of the center of mass has been set to zero to remove spurity.

Our goal is to set up a PDE solver that can be used to solve for different phenomenological potentials $V(r)$. Some simple choices are

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
- Xiaoliang suggested [`SOAP`][https://github.com/nikhilvyas/SOAP/tree/main] optimizer, which can be easily integrated with `torch` and seems to outperform the common Adam optimizer.
