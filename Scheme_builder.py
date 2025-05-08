# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 22:41:21 2025

@author: Henry Yue
"""
import numpy as np
from numpy.fft import fft, ifft

# %% Dependencies


def solve_circulant(A, b):
    """
    Solve the circulant system A x = b in O(n log n) time,
    where A is the n x n circulant matrix whose first row is c.

    Parameters
    ----------
    c : array_like
        Length-n array representing the first row of A.
    b : array_like
        Length-n vector (right-hand side).

    Returns
    -------
    x : complex ndarray
        The solution vector (length n).
        Typically you'll want the real part if everything was real.
    """
    c = A[:, 0]
    # Compute eigenvalues via FFT of the first row
    lam = fft(c)

    # FFT of b
    B = fft(b)

    # Invert element-wise (assuming no zero eigenvalues)
    X = B / lam

    # Inverse FFT to get the solution
    x = ifft(X)

    return x.real


def matrix_periodic_linag_constructor(phi, L, Diag, R):
    """
    Generate the solution matrix corresponding to a system of linear equations based on phi_j, phi_j+1, phi_j-1 and thier coefficients

    Parameters
    ----------
    phi : Array of the numerical space (initial conditions)
    L (phi_j-1) : coefficient of phi_j-1 terms
    Diag (phi_j): coefficient of phi_j terms
    R (phi_j+1): coefficient of phi_j+1 terms

    Returns
    -------
    M : The solution matrix

    """
    M = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M[i, i] = Diag

        # Left and right of diagonals (phi_j-1 and phi_j+1 )with periodic boundary
        M[i, (i - 1) % len(phi)] = L
        M[i, (i + 1) % len(phi)] = R
    return M


def Generic_scheme_builder(phi, u, K, dt, dx, params):
    """
    Generate the solution matrix for the LHS (y^n+1 terms coefficients)
    and RHS (y^n terms coefficients) for a scheme given the params (recipe).

    parameters
    ----------
    phi : array_like
        Array of the numerical space (initial conditions)
    u : float
        Advection velocity
    K : float
        Diffusion coefficient
    dt : float
        Time step size
    dx : float
        Spatial step size
    params : dict
        Dictionary containing scheme parameters [xi_a, xi_d, eta_a, eta_d, eta_s]

    returns
    -------
    LHS_y1terms_matrix : array_like
        The solution matrix for the LHS (y^n+1 terms coefficients)
    RHS_y0terms_matrix : array_like
        The solution matrix for the RHS (y^n terms coefficients)
    """
    C = u * dt / dx
    D = K * dt / dx**2
    # Extract the scheme variables from params
    xi_a, xi_d, eta_a, eta_d = (
        params[key] for key in ["xi_a", "xi_d", "eta_a", "eta_d"]
    )
    # Build the matricies by subsituting the generic equation for centered difference schemes
    LHS_y1terms_matrix = matrix_periodic_linag_constructor(
        phi,
        (-C * eta_a * xi_a / 2) - (D * eta_d * xi_d),
        (2 * D * eta_d * xi_d + 1),
        (C * eta_a * xi_a / 2) - (D * eta_d * xi_d),
    )
    RHS_y0terms_matrix = -1 * matrix_periodic_linag_constructor(
        phi,
        (C * eta_a * xi_a / 2)
        - (C * eta_a / 2)
        + (D * eta_d * xi_d)
        - (D * eta_d),
        (-2 * D * eta_d * xi_d + 2 * D * eta_d - 1),
        (-C * eta_a * xi_a / 2)
        + (C * eta_a / 2)
        + (D * eta_d * xi_d)
        - (D * eta_d),
    )

    return LHS_y1terms_matrix, RHS_y0terms_matrix


def Generic_scheme_solver(phi, source, u, K, dt, dx, nt, recipe):
    """
    Run the scheme specified in the recipe for nt time steps.
    Parameters
    ----------
    phi : array_like
        Array of the numerical space (initial conditions)
    source : expression
        Source term to be added to the RHS
    u : float
        Advection velocity
    K : float
        Diffusion coefficient
    dt : float
        Time step size
    dx : float
        Spatial step size
    nt : int
        Number of time steps to run the scheme
    recipe : dict
        Dictionary containing scheme parameters [xi_a, xi_d, eta_a, eta_d, eta_s]
    returns
    -------
    phi : array_like
        Updated array of the numerical space after nt time steps.
    """

    # Initialize the matricies to record the instruction to be executed as described in the recipe
    LHS_array = []
    RHS_array = []
    eta_s_array = []
    # Loop through the recipe to build the LHS and RHS matrices for each intermediate step and store the instructions
    for key, params in recipe.items():
        LHS, RHS = Generic_scheme_builder(phi, u, K, dt, dx, params)
        LHS_array.append(LHS)
        RHS_array.append(RHS)
        eta_s_array.append(params["eta_s"])

    # Perform the instructions
    for i in range(nt):
        for LHS, RHS, eta_s in zip(LHS_array, RHS_array, eta_s_array):
            phi = solve_circulant(LHS, RHS @ phi + dt * source * eta_s)

    return phi


# %% ADE no source term recipes
BTCS_recipe = {
    "BTCS advdif": {"xi_a": 1, "xi_d": 1, "eta_a": 1, "eta_d": 1, "eta_s": 0},
}

BTCS_AD_recipe = {
    "BTCS adv": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1,
        "eta_d": 0,
        "eta_s": 0,
    },
    "BTCS dif": {
        "xi_a": 0,
        "xi_d": 1,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 0,
    },
}

BTCS_DA_recipe = {
    "BTCS dif": {
        "xi_a": 0,
        "xi_d": 1,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 0,
    },
    "BTCS adv": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1,
        "eta_d": 0,
        "eta_s": 0,
    },
}

BTCS_ADA_recipe = {
    "BTCS adv1": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
    "BTCS dif": {
        "xi_a": 0,
        "xi_d": 1,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 0,
    },
    "BTCS adv2": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
}

BTCS_DAD_recipe = {
    "BTCS dif": {
        "xi_a": 0,
        "xi_d": 1,
        "eta_a": 0,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
    "BTCS adv": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1,
        "eta_d": 0,
        "eta_s": 0,
    },
    "BTCS dif2": {
        "xi_a": 0,
        "xi_d": 1,
        "eta_a": 0,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
}

CNCS_recipe = {
    "CNCS advdif": {
        "xi_a": 1 / 2,
        "xi_d": 1 / 2,
        "eta_a": 1,
        "eta_d": 1,
        "eta_s": 0,
    }
}

CNCS_AD_recipe = {
    "CNCS adv": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1,
        "eta_d": 0,
        "eta_s": 0,
    },
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 0,
    },
}

CNCS_DA_recipe = {
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 0,
    },
    "CNCS adv": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1,
        "eta_d": 0,
        "eta_s": 0,
    },
}

CNCS_ADA_recipe = {
    "CNCS adv1": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 0,
    },
    "CNCS adv2": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
}

CNCS_DAD_recipe = {
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
    "CNCS adv": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1,
        "eta_d": 0,
        "eta_s": 0,
    },
    "CNCS dif2": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
}

CNCS_EI_ADA_recipe = {
    "CNCS Eadv": {
        "xi_a": 0,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 0,
    },
    "CNCS Iadv": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
}

CNCS_EI_DAD_recipe = {
    "CNCS Edif": {
        "xi_a": 0,
        "xi_d": 0,
        "eta_a": 0,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
    "CNCS adv": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1,
        "eta_d": 0,
        "eta_s": 0,
    },
    "CNCS Idif": {
        "xi_a": 0,
        "xi_d": 1,
        "eta_a": 0,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
}

CNCS_IE_ADA_recipe = {
    "CNCS Iadv": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 0,
    },
    "CNCS Eadv": {
        "xi_a": 0,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
}

CNCS_IE_DAD_recipe = {
    "CNCS Idif": {
        "xi_a": 0,
        "xi_d": 1,
        "eta_a": 0,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
    "CNCS adv": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1,
        "eta_d": 0,
        "eta_s": 0,
    },
    "CNCS Edif": {
        "xi_a": 0,
        "xi_d": 0,
        "eta_a": 0,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
}

# %% ADE with source term recipes
CNCS_source_advdif_recipe = {
    "CNCS advdif": {
        "xi_a": 1 / 2,
        "xi_d": 1 / 2,
        "eta_a": 1,
        "eta_d": 1,
        "eta_s": 1,
    },
}

CNCS_Imp_Source_Exp_recipe = {
    "CNCS Imp": {
        "xi_a": 1,
        "xi_d": 1,
        "eta_a": 1 / 2,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
    "Source": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1},
    "CNCS Exp": {
        "xi_a": 0,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
}

CNCS_Exp_Source_Imp_recipe = {
    "CNCS Exp": {
        "xi_a": 0,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
    "Source": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1},
    "CNCS Imp": {
        "xi_a": 1,
        "xi_d": 1,
        "eta_a": 1 / 2,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
}

BTCS_Source_Imp_recipe = {
    "Source": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1},
    "BTCS Imp": {"xi_a": 1, "xi_d": 1, "eta_a": 1, "eta_d": 1, "eta_s": 0},
}

CN_Source_CN_recipe = {
    "CNCS adv1": {
        "xi_a": 1 / 2,
        "xi_d": 1 / 2,
        "eta_a": 1 / 2,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
    "Source": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1},
    "CNCS adv2": {
        "xi_a": 1 / 2,
        "xi_d": 1 / 2,
        "eta_a": 1 / 2,
        "eta_d": 1 / 2,
        "eta_s": 0,
    },
}
