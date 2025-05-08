# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 07:23:58 2024

@author: Henry Yue
"""

import numpy as np
import math as m


def Construct_Sinewave(phi, A, k, phase, x_start, x_end):
    """
    Constructs a sine wave on the array phi from x_start to x_end with wave number k

    Parameters
    ----------
    phi : Initial space to construct upon [array]
    A : Wave amplitude [float/int]
    k : Wave number
    phase : Phase
    x_start : x value in phi to start at
    x_end : x value in phi to end at

    Returns
    -------
    phi with defined wave constructed

    """
    length = x_end - x_start
    for j in range(x_start, x_end):
        phi[j] = A * np.sin((2 * np.pi * k / length) * j + phase) + phi[j]
    return phi


def analytical_square_adv_dif(x, t, A, x_start, x_end, K, u):
    """
    Analytical solution for 1D advection diffusion equation of a square wave (non-periodic)

    Parameters
    ----------
    x : Spatial coordinate
    t : Simulation time
    A : Square wave amplitude
    x_start : Start of the square wave
    x_end : End of the square wave
    K : Diffusion coefficent
    u : Advection Velocity

    Returns
    -------
    Value (phi) of the adv-dif equation at location x after time t

    """
    phi = 0.5 * (
        m.erf(((x - u * t) - x_start) / m.sqrt(4 * K * t))
        - m.erf(((x - u * t) - x_end) / m.sqrt(4 * K * t))
    )
    return phi


def analytical_sine_adv_dif(u, K, k, L, A, x, t):
    """
    Analytical solution for 1D advection diffusion equation of a sine wave (periodic)

    Parameters
    ----------
    u : Advection Velocity
    K : Diffusion coefficent
    L : Total length of simulation space
    A : Wave amplitude
    x : Spatial coordinate
    t : Simulation time

    Returns
    -------
    Value (phi) of the adv-dif equation at location x after time t

    """
    phi_analytic = (
        A
        * np.exp(-K * (2 * np.pi * k / (L)) ** 2 * t)
        * np.sin((2 * np.pi * k / L) * (x - u * t))
    )
    return phi_analytic


def analytical_sine_source_adv(u, k, L, A, x, t):
    phi_analytic = -np.cos(x) / u
    return phi_analytic


def analytical_sine_source_adv_dif(u, K, k, L, A, x, t):
    """
    Analytical solution for 1D advection diffusion equation of a sine wave (periodic)

    Parameters
    ----------
    u : Advection Velocity
    K : Diffusion coefficent
    L : Total length of simulation space
    A : Wave amplitude
    x : Spatial coordinate
    t : Simulation time

    Returns
    -------
    Value (phi) of the adv-dif equation at location x after time t

    """
    phi_analytic = np.sin(x)
    return phi_analytic
