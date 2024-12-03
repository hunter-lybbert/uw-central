"""The main file for solving the vorticity strem function from hw 5 of scientific computing"""

import time

from typing import Any, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import solve_ivp
from scipy.fftpack import fft2, ifft2
from scipy.sparse import linalg as sp_la
from scipy import linalg as reg_la

from src.common.file_io_helper import incriment_file
from src.scientific_computing.common.difference_matrices import build_matrices
from src.scientific_computing.vorticity_stream_function.linear_system_solvers import LinearSystemSolveMethods


DEFAULT_OMEGA_PARAMS = [
    {
        "amplitudes": np.array([1]),
        "locations": np.array([[0, 0]]),
        "variances":np.array([[1, 20]]),
    }
]


def omega_0_func(
    x: np.array,
    y: np.array,
    amplitude: float = 1,
    variance_in_x: float = 4,
    variance_in_y: float = 1,
    x_location: float = 0,
    y_location: float = 0,
) -> np.array:
    """
    Initial conditions for vorticity aka omega.

    :param x: the array indicating the linspace in the x spatial dimenstino
    :param y: the array indicating the linspace in the y spatial dimenstino
    :param amplitude: the height of the gaussian at x,y = (0,0)
    :param variance_in_x: measure of the spread of the gaussian in the x direction
    :param variance_in_y: measure of the spread of the gaussian in the y direction

    """
    exponent = (
        -(x - x_location)**2/variance_in_x
        -(y - y_location)**2/variance_in_y
    )
    return amplitude*np.exp(exponent)


def multi_omega_0_func(
    x, y, amplitudes, locations, variances
) -> np.array:
    """"""
    omega_0 = np.zeros_like(x)
    for k in range(len(amplitudes)):
        omega_0 += omega_0_func(
            x=x,
            y=y,
            amplitude=amplitudes[k],
            variance_in_x=variances[k][0],
            variance_in_y=variances[k][1],
            x_location=locations[k][0],
            y_location=locations[k][1],
        )
    return omega_0


class VorticityStreamFunctionSolver:
    """A clase to configure and solve the Vorticity Stream Function Differential Equation."""
    
    def __init__(
        self,
        nu: float,
        L: int,
        n: float,
        omega_0_params: dict[str, Any],
        solve_method_name: str = LinearSystemSolveMethods.fourier.name,
        tol: Optional[float] = 1e-6,
    ) -> None:
        """
        Initialize Everything.
        
        :param nu: customizable parameter, no clear interpretation yet

        :returns: None
        """
        self.L: int = L
        self.n: float = n
        self.nu: float = nu

        self.solve_method_name = solve_method_name
        # TODO Change back
        self.solve_method_func = LinearSystemSolveMethods[solve_method_name].value
        self.solve_method_params = dict()

        self._setup_omega_0(omega_0_params)
        self._setup_diff_matrices()

        # TODO Change back
        if self.solve_method_name == LinearSystemSolveMethods.fourier.name:
            self._setup_wavenumbers()
        else:
            self._setup_solve_method_params(tol=tol)

    def _setup_omega_0(self, omega_0_params: dict[str, Any]) -> None:
        """
        """
        x_full = np.linspace(-self.L, self.L, self.n + 1)
        y_full = np.linspace(-self.L, self.L, self.n + 1)

        x_trunc = x_full[:self.n]
        y_trunc = y_full[:self.n]

        self.X, self.Y = np.meshgrid(x_trunc, y_trunc)

        omega_0_mat: np.array = multi_omega_0_func(
            x=self.X,
            y=self.Y,
            **omega_0_params,
        )
        omega_0_vec: np.array = omega_0_mat.reshape(self.n*self.n)
        self.omega_0 = omega_0_vec

    def _setup_diff_matrices(self) -> None:
        """
        """
        delta_x = ( self.L - (-self.L))/self.n
        self.A, self.B, self.C = build_matrices(n=self.n, delta_x=delta_x, output_as_csc=True)

    def _setup_wavenumbers(self) -> None:
        """
        """
        scale_factor = 2 * np.pi / (self.L - (-self.L))
        kx = scale_factor * np.concatenate((np.arange(0, self.n/2), np.arange(-self.n/2, 0)))
        ky = scale_factor * np.concatenate((np.arange(0, self.n/2), np.arange(-self.n/2, 0)))

        # avoid divide by zero with floating point precision error
        kx[0] = 1e-6
        ky[0] = 1e-6
        self.kx, self.ky = np.meshgrid(kx, ky)
        self.solve_method_params = {
            "kx": self.kx,
            "ky": self.ky,
        }
    
    def _setup_solve_method_params(self, tol: Optional[float] = 1e-6) -> None:
        """"""
        self.A[0,0] = 2
        self.solve_method_params = {
            "laplacian": self.A,
            "reshape_dim": self.n,
        }
        # TODO Change back
        methods_use_tol = [LinearSystemSolveMethods.bicgstab.name, LinearSystemSolveMethods.gmres.name]
        if self.solve_method_name in methods_use_tol:
            self.solve_method_params.update({"tol": tol})

    def omega_ode_rhs(self, t: float, omega: np.array):
        """Define the ODE."""
        omega_mat = omega.reshape((self.n, self.n))

        psi_mat = self.solve_method_func(omega_mat, **self.solve_method_params)
        psi = psi_mat.reshape(self.n*self.n)

        diffusion = self.nu * self.A@omega
        advection = ((self.B@psi) * (self.C@omega)) - ((self.C@psi) * (self.B@omega))
        final = diffusion - advection

        return final
    
    def solve(self, tspan: np.array) -> None:
        """Solve the ODE with solve_ivp."""

        start_time = time.time()
        self.omega_sol = solve_ivp(
            self.omega_ode_rhs,
            t_span=(tspan[0], tspan[-1]),
            y0=self.omega_0,
            method="RK45",
            t_eval=tspan,
        )
        end_time = time.time()
        self.time_to_solve = end_time - start_time
        
        return self.omega_sol

    def plot(self, step: int) -> None:
        """"""
        plt.pcolor(self.omega_sol.y[:,step].reshape((self.n,self.n)))
        plt.show()

    def create_animation(self, tspan: np.array, save: bool):
        """Create animation of the system as time evolved."""
        animation.writer = animation.writers['ffmpeg']

        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # write the update function, specifically including the ax.clear() function this was important.
        def update(i):
            ax.clear()
            ax.pcolor(self.X, self.Y, self.omega_sol.y[:,i].reshape((self.n, self.n)), cmap='viridis')
            ax.set_title("Evolution of Vorticity Stream Function")
            return ax

        self.ani = animation.FuncAnimation(fig, update, frames=range(len(tspan)), interval=25)

        if save:
            self._save_animation()

    def _save_animation(self) -> None:
        # Save gif
        file_name = incriment_file("vorticity_stream_function.gif", "visuals")
        self.ani.save(file_name, writer='pillow')

        # Save as MP4
        file_name_mp4 = incriment_file("vorticity_stream_function.mp4", "visuals")
        writer = animation.writers['ffmpeg']
        writer = writer(metadata=dict(artist='Hunter Lybbert'), fps=25)
        self.ani.save(file_name_mp4, writer=writer)


def run(
    L: int = 10,
    n: int = 64,
    nu: float = 0.001,
    tspan_stop: float = 4.5,
    tspan_step_size: float = .5,
    solve_method_name: str = LinearSystemSolveMethods.fourier.name,
    omega_params: list[dict[str, np.array]] = DEFAULT_OMEGA_PARAMS,
    animate: bool = True,
    save_animation: bool = True,
) -> VorticityStreamFunctionSolver:
    """
    Run a the Vorticity Stream Function Solver given the setup params passed in here.

    :param L: the length of the square domain to solve the vorticity strem function pde in
    :param n: the number of intervals to chop up the domain into
    :param nu: the parameter nu in the pde
    :param tspan_stop: the stop time for evolving the solution forward in time
    :param tspan_step_size: the size of time steps for using solve ivp
    :param omega_params: the locations and setup parameters for the initial conditions of omega
    :param animate: bool whether to animate or not
    :param save_animation: bool to save animation or not

    :returns: The VorticityStreamFunctionSolver object
    """
    tspan = np.arange(0, tspan_stop, tspan_step_size)

    for omega_0_params in omega_params:
        VSFSolver = VorticityStreamFunctionSolver(
            nu=nu,
            L=L,
            n=n,
            omega_0_params=omega_0_params,
            solve_method_name=solve_method_name,
        )
        VSFSolver.solve(tspan=tspan)

        if animate:
            VSFSolver.create_animation(tspan, save=save_animation)

    return VSFSolver
