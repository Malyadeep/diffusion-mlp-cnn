import numpy as np
import numba


@numba.njit
def solver(u_initial, Nx, iterations, Fo):
    u = np.zeros((iterations + 1, Nx), dtype=np.float32)
    u[0, :] = u_initial.copy()
    for t in range(1, iterations + 1):
        for x in range(Nx):
            x_left = (x - 1) % Nx
            x_right = (x + 1) % Nx
            d2udx2 = (u[t - 1, x_left] + u[t - 1, x_right] - 2 * u[t - 1, x])
            u[t, x] = u[t - 1, x] + Fo * d2udx2
    return u


class State:
    def __init__(self, L):
        self.L = L
        self.del_x = 1
        self.del_t = 1
        self.Nx = int(self.L // self.del_x + 1)
        self.x = np.linspace(-L/2, L/2, self.Nx)
        self.u_initial = np.zeros(self.Nx, dtype=np.float32)

    def initialize(self, func, params):
        self.u_initial = func(self.x, *params)

    def evolve(self, iterations, D, verbose=True):
        if verbose:
            print("Solving system in time")
        Fo = D * self.del_t / self.del_x**2
        self.time_scale = self.L**2 / D
        if verbose:
            if Fo <= 0.5:
                print(f"Chosen Fourier number: {Fo} <= 0.5")
                print("Stable discretization!")
            else:
                print(f"Chosen Fourier number: {Fo} > 0.5")
                print("Unstable discretization!")
        self.u = solver(self.u_initial, self.Nx, iterations, Fo)
        if verbose:
            print("Solution complete!")
