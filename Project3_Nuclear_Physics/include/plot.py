import matplotlib.pyplot as plt

def plot_wavefunction(r, u, R, E, title="Radial Schrödinger", skip=5):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(r, u, color='green', lw=2, label=r'$u(r)$')

    ax.plot(r[skip:], R[skip:], color='blue',  lw=2, label=r'$R(r)=u(r)/r$')

    ax.set_xlabel('r (fm)')
    ax.set_xlim(min(r), max(r))
    ax.set_ylabel('Wave function')
    ax.set_title(f'{title}: E = {E:.4f} MeV')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_wavefunctions(r, u_scipy, R_scipy, u_nn, R_nn, E, title="Radial Schrödinger", skip=5):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(r, u_nn, color='green', lw=2, label=r'$u(r)$ (torch)')
    ax.plot(r, u_scipy, color='green', lw=8,
            label=r'$u(r)$ (scipy)', alpha=0.2)

    ax.plot(r[skip:], R_nn[skip:], color='blue',
            lw=2, label=r'$\Psi(r)$ (torch)')
    ax.plot(r[skip:], R_scipy[skip:], color='blue',
            lw=8, label=r'$\Psi(r)$ (scipy)', alpha=0.2)

    ax.set_xlabel('r (fm)')
    ax.tick_params(axis='y', labelcolor='blue')

    ax.set_xlabel('r (fm)')
    ax.set_xlim(min(r), max(r))
    ax.set_ylabel('Wave function')
    ax.set_title(f'{title}: E = {E:.4f} MeV')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
