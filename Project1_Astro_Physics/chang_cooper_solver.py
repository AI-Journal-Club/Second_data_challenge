import numpy as np
import matplotlib.pyplot as plt
from pychangcooper import ChangCooper

class MySolver(ChangCooper):
    def __init__(self):

        # we have no injection, so we must have an
        # initial non-zero distribution function

        init_distribution = np.ones(100)

        # must pass up to the super class so that
        # all terms are setup after initialization

        super(MySolver, self).__init__(
            n_grid_points=100,
            delta_t=1.0,
            max_grid=1e5,
            initial_distribution=init_distribution,
            store_progress=True,  # store each time step
        )

    def _define_terms(self):

        # energy dependent heating and dispersion terms
        # must be evaluated at half grid points.

        # These half grid points are automatically
        # calculated about object creation.

        self._heating_term = self._half_grid

        self._dispersion_term = self._half_grid2


solver = MySolver()

# amount of time that has gone by
print(solver.current_time)

# number of
print(solver.n_iterations)

# current solution
print(solver.n)

solver.solve_time_step()


# amount of time that has gone by
print(solver.current_time)

# number of
print(solver.n_iterations)

# current solution
print(solver.n)


_time_ene_sol_ = []
for i in range(50):
    solver.solve_time_step()
    sol = solver.n
    _time_ene_sol_.append(sol)
_time_ene_sol_ = np.array(_time_ene_sol_)

print(len(_time_ene_sol_), len(_time_ene_sol_[0]))

plt.figure()
plt.imshow(_time_ene_sol_)
plt.xlabel(r'$\gamma$')
plt.ylabel('Time [steps]')
plt.colorbar()


plt.figure()
for t in range(0, 50, 5):
    plt.plot(np.arange(100), _time_ene_sol_[t], label=f't$_{t}$')
plt.yscale('log')
plt.xlabel(r'$\gamma$')
plt.ylabel('N($\gamma$, t)')

plt.show()



# solver.plot_evolution(alpha=0.8)

