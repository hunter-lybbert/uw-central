"""Methods used to solve the linear system in the vorticity stream function pde"""

from enum import Enum, member

import numpy as np

from scipy.fftpack import fft2, ifft2
from scipy.sparse import linalg as sp_la
from scipy import linalg as reg_la


def fourier_solve(
    omega: np.array,
    kx: np.array,
    ky: np.array,
) -> np.array:
    """This should be an nxn array in and out"""
    derivative_terms = kx**2 + ky**2
    omega_ft = -fft2(omega)/derivative_terms
    psi = ifft2(omega_ft)
    return np.real(psi)


def spab_solve(
    omega: np.array,
    laplacian: np.array,
    reshape_dim: int
) -> np.array:
    """"""
    omega_vec = omega.reshape(reshape_dim**2)
    psi_vec = sp_la.spsolve(laplacian, omega_vec)
    psi_mat = psi_vec.reshape((reshape_dim, reshape_dim))
    return psi_mat


def ab_solve(
    omega: np.array,
    laplacian: np.array,
    reshape_dim: int
) -> np.array:
    """"""
    laplacian = laplacian.toarray()
    omega_vec = omega.reshape(reshape_dim**2)

    psi_vec = reg_la.solve(laplacian, omega_vec)
    psi_mat = psi_vec.reshape((reshape_dim, reshape_dim))
    return psi_mat


def splu_solve(
    omega: np.array,
    laplacian: np.array,
    reshape_dim: int,
) -> np.array:
    """"""
    omega_vec = omega.reshape(reshape_dim**2)
    A = sp_la.splu(laplacian)
    psi_vec = A.solve(omega_vec)
    psi_mat = psi_vec.reshape((reshape_dim, reshape_dim))
    return psi_mat


def lu_solve(
    omega: np.array,
    laplacian: np.array,
    reshape_dim: int,
) -> np.array:
    """"""
    laplacian = laplacian.toarray()
    omega_vec = omega.reshape(reshape_dim**2)

    P, L, U = reg_la.lu(laplacian)

    P_omega = np.dot(P, omega_vec)
    LP_omega = reg_la.solve_triangular(L, P_omega, lower=True)
    psi_vec = reg_la.solve_triangular(U, LP_omega)
    psi_mat = psi_vec.reshape((reshape_dim, reshape_dim))
    
    return psi_mat


def bicgstab_solve(
    omega: np.array,
    laplacian: np.array,
    reshape_dim: int,
    tol: float = 1e-6,
) -> np.array:
    """"""
    omega_vec = omega.reshape(reshape_dim**2)
    psi_vec, exit_code = sp_la.bicgstab(laplacian, omega_vec, atol=tol)
    psi_mat = psi_vec.reshape((reshape_dim, reshape_dim))
    return psi_mat


def gmres_solve(
    omega: np.array,
    laplacian: np.array,
    reshape_dim: int,
    tol: float = 1e-6,
) -> np.array:
    """"""
    omega_vec = omega.reshape(reshape_dim**2)
    psi_vec, exit_code = sp_la.gmres(laplacian, omega_vec, atol=tol)
    psi_mat = psi_vec.reshape((reshape_dim, reshape_dim))
    return psi_mat

class LinearSystemSolveMethods(Enum):
    """
    Options of how to solve system
    """
    fourier = member(fourier_solve)
    spab = member(spab_solve)
    ab = member(ab_solve)
    splu = member(splu_solve)
    lu = member(lu_solve)
    bicgstab = member(bicgstab_solve)
    gmres = member(gmres_solve)
