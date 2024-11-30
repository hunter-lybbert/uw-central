"""Construct difference matrices intended for use in solving large systems of differential equations."""

from typing import Optional

import numpy as np
from scipy.sparse import spdiags, dia_matrix


def partial_in_x(n: int) -> dia_matrix:
    """
    :param n: the dimension of the (n, n) matrix
    """
    upper = np.ones(n*n)
    lower = -1*np.ones(n*n)

    matrix = spdiags([upper, lower, lower, upper], diags=np.array([n, -n, n*(n - 1), -n*(n - 1)]))

    return matrix


def partial_in_y(n: int) -> dia_matrix:
    """
    :param n: the dimension of the (n, n) matrix
    """
    upper = np.tile(np.repeat([0, 1], [1, n-1]), n)
    lower = np.tile(np.repeat([-1, 0], [n-1, 1]), n)

    upper_wrap_around = np.tile(np.repeat([0, -1], [n-1, 1]), n)
    lower_wrap_around = np.tile(np.repeat([1, 0], [1, n-1]), n)

    matrix = spdiags([upper, lower, upper_wrap_around, lower_wrap_around], diags=np.array([1, -1, n - 1, -n + 1]))

    return matrix


def second_gradient_x_y(n: int) -> dia_matrix:
    """
    :param n: the dimension of the (n, n) matrix
    """
    main_diag = -4*np.ones(n*n)
    upper = np.tile(np.repeat([0, 1], [1, n-1]), n)
    lower = np.tile(np.repeat([1, 0], [n-1, 1]), n)
    
    upper_wrap_around = np.tile(np.repeat([0, 1], [n-1, 1]), n)
    lower_wrap_around = np.tile(np.repeat([1, 0], [1, n-1]), n)

    off_diag = np.ones(n*n)

    data = [
        main_diag,
        upper,
        lower,
        upper_wrap_around,
        lower_wrap_around,
        off_diag,
        off_diag,
        off_diag,
        off_diag
    ]
    diags = np.array(
        [
            0,
            1,
            -1,
            n - 1,
            -n + 1,
            n*(n - 1),
            -n*(n - 1),
            n,
            -n
        ]
    )

    matrix = spdiags(data, diags=diags)

    return matrix


def build_matrices(
    n: int,
    delta_x: float,
    delta_y: Optional[float] = None,
    output_as_np: Optional[bool] = False,
    output_as_csc: Optional[bool] = False,
) -> np.array:
    """
    
    :param n: the dimension of the (n,n) matrices
    :param delta_x: the step size in x
    :param delta_y: the step size in y
    :param output_as_np:
    :param output_as_csc:

    """
    if not delta_y:
        delta_y = delta_x

    if delta_x == delta_y:
        A = (1/(delta_x**2))*second_gradient_x_y(n)
        B = (1/(2*delta_x))*partial_in_x(n)
        C = (1/(2*delta_x))*partial_in_y(n)

    else:
        raise NotImplementedError("Currently do not support different step sizes in x and y")
    
    if output_as_np:
        A = A.toarray()
        B = B.toarray()
        C = C.toarray()

    if output_as_csc:
        A = A.tocsc()
        B = B.tocsc()
        C = C.tocsc()

    return A, B, C
