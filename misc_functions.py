# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 04:57:10 2024

@author: Henry Yue
"""
from scipy import stats
import math as m
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt

line_styles = [
    "o",
    "^",
    "v",
    "D",
    "s",
    "p",
    "h",
]  # Solid, dashed, dash-dot, dotted


def RMSE(numerical_solution, analytical_solution):
    """
    Calculate the Root Mean Square Error (RMSE) between a numerical solution and an analytical solution.

    Parameters:
    - numerical_solution (array-like): The numerical solution data.
    - analytical_solution (array-like): The analytical solution data.

    Returns:
    - float: The RMSE value.
    """
    # Ensure inputs are NumPy arrays for efficient operations
    numerical_solution = np.array(numerical_solution)
    analytical_solution = np.array(analytical_solution)

    # Check if the lengths match
    if numerical_solution.shape != analytical_solution.shape:
        raise ValueError(
            "Numerical and analytical solutions must have the same shape."
        )

    # Compute RMSE
    mse = np.mean(
        (numerical_solution - analytical_solution) ** 2
    )  # Mean squared error
    rmse = np.sqrt(mse)  # Root mean squared error

    return rmse


def l2_norm(phi_num, phi_analytical):
    error = phi_num - phi_analytical
    l2 = np.sqrt(np.sum(error**2)) / np.sqrt(
        np.sum(phi_analytical**2)
    )
    return l2


def plot_scheme(
    x,
    nt,
    dt,
    phi_initial,
    phi_analytical,
    phi_schemes_n_labels,
    dom=[],
    ran=[],
):
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis
    ax.set_xlim(dom[0], dom[1])  # Set domain (x-axis) limits
    ax.set_ylim(ran[0], ran[1])  # Set range (y-axis) limits

    # Plot the initial condition and analytical solution
    ax.plot(
        x,
        phi_initial,
        "--",
        alpha=0.5,
        label="Initial Condition",
        color="blue",
    )
    ax.plot(x, phi_analytical, alpha=0.5, label="Analytic")

    # Plot each numerical scheme
    for scheme, label in phi_schemes_n_labels:
        ax.plot(x, scheme, label=label)

    # Add title, legend, and layout adjustments
    ax.set_title(f"T={nt*dt:.2f}, nt={nt:.2f}, nx={len(x)}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    plt.tight_layout()




def plot_scheme_separate(
    x,
    nt,
    dt,
    phi_initial,
    phi_analytical,
    phi_schemes_n_labels,
    dom=[],
    ran=[],
):
    for scheme, label in phi_schemes_n_labels:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(dom[0], dom[1])
        ax.set_ylim(ran[0], ran[1])

        ax.plot(
            x,
            phi_initial,
            "--",
            alpha=0.5,
            label="Initial Condition",
            color="blue",
        )
        plt.plot(x, phi_analytical, alpha=0.5, label="Analytic")
        plt.plot(x, scheme, label=label)
        plt.title(f"T={nt*dt}")

        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        plt.tight_layout()
        plt.show()


def plot_error(phi_errors_n_labels, dx, dt, C, D, log_scale=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel("RMSE")
    color_palette = plt.cm.viridis(
        np.linspace(0, 1, len(phi_errors_n_labels))
    )  # Optional: Use a colormap for colors

    # Loop to plot errors and print average error
    for i, (label, error_values) in enumerate(phi_errors_n_labels.items()):
        color = color_palette[i]
        line_style = line_styles[i % len(line_styles)]
        ax.plot(error_values, line_style, label=label, color=color)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    if log_scale:
        ax.set_xscale("log")  # Use a logarithmic scale for dx
        ax.set_yscale("log")  # Use a logarithmic scale for errors


    # Chat GPT provides:
    # Sort the errors by their mean value (ascending)
    error_ranking_sorted = sorted(
        phi_errors_n_labels.items(),  # Get (label, error_values) pairs
        key=lambda x: np.mean(x[1]),  # Sort by mean error
    )

    # Print the ranking
    print("\nError Rankings (Lowest to Highest Mean Error):")
    for rank, (label, error_values) in enumerate(error_ranking_sorted, 1):
        print(f"{rank}. {label}: Mean Error = {np.mean(error_values):.3f}")


def rank(quantity_n_labels, name):
    error_ranking_sorted = sorted(
        quantity_n_labels.items(),  # Get (label, error_values) pairs
        key=lambda x: np.mean(x[1]),  # Sort by mean error
    )

    # Print the ranking
    print(f"\n{name} (Lowest to Highest):")
    for rank, (label, error_values) in enumerate(error_ranking_sorted, 1):
        print(f"{rank}. {label}: {name} = {np.mean(error_values):.6f}")
    return


def calc_slope(dx_list, error_list):
    log_dx = np.log10(dx_list)
    log_error = np.log10(error_list)

    slope, intercept, r_value, p_value, std_err = linregress(log_dx, log_error)

    return slope


def find_max(C_values, D_values, kdx_list, A_eq):
    """
    Generate a array (i corresponds to Courant number, j corresponds to Diffusion number) of maximum amplifying factor given by the equation A_eq through iterating

    Parameters
    ----------
    C_values : Array of Courant numbers to test for (0.00001 to inf)
    D_values : Array of Diffusion numbers to test for (0.00001 to inf)
    kdx_list : k*dx values (0 to 2*pi)
    A_eq : Equation for the amplifcation factor (a function(D,C,kdx))

    Returns
    -------
    A_max : Array (C by D) of maximum amplification factor

    """
    A_max = np.zeros((len(C_values), len(D_values)))
    # Compute maximum A for each pair of (C, D)
    for i, C in enumerate(C_values):
        for j, D in enumerate(D_values):
            for kdx in kdx_list:
                max_A = -np.inf
                A = A_eq(D, C, kdx)
                if A > max_A:
                    max_A = A
            A_max[i, j] = max_A

    return A_max

