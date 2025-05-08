# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:03:14 2025

@author: Henry Yue
"""

import numpy as np
from initial_conditions import *
from misc_functions import *
from advection_diffusion_FFT import *
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Scheme_builder import *

def plot_error(
    x_axis,
    phi_errors_n_labels,
    ax,
    log_scale=False,
    ylabel="$\\ell_2$ error norm",
    xlabel=None,
    title=None
):
    """
    Plot error norms on a provided Axes.

    Parameters
    ----------
    x_axis : array_like
        The x values (e.g. dt or dx).
    phi_errors_n_labels : dict[label -> list of errors]
        Mapping from scheme label to list of error values.
    ax : matplotlib.axes.Axes
        The axes on which to draw.
    log_scale : bool, optional
        If True, use log–log scale.
    ylabel : str, optional
        Label for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    title : str, optional
        Plot title.
    """
    # loop through each scheme
    color_palette = plt.cm.viridis(np.linspace(0, 1, len(phi_errors_n_labels)))
    for i, (label, error_values) in enumerate(phi_errors_n_labels.items()):
        color = color_palette[i]
        line_style = line_styles[i % len(line_styles)]
        ax.plot(x_axis, error_values, line_style, label=label, color=color)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # ranking output
    error_ranking_sorted = sorted(
        phi_errors_n_labels.items(),
        key=lambda x: np.mean(x[1])
    )
    print("\nError Rankings (Lowest to Highest Mean Error):")
    for rank, (lbl, vals) in enumerate(error_ranking_sorted, 1):
        print(f"{rank}. {lbl}: Mean Error = {np.mean(vals):.3e}")
    
    plt.legend()

    return ax


def plot_scheme(
    x,
    nt,
    dt,
    phi_initial,
    phi_analytical,
    phi_schemes_n_labels,
    ax,
    dom=None,
    ran=None,
):
    """
    Plot initial, analytic, and numerical-scheme solutions on a provided Axes.

    Parameters
    ----------
    x : array_like
        Spatial grid.
    nt : int
        Number of time steps.
    dt : float
        Time-step size.
    phi_initial : array_like
        Initial condition φ(x,0).
    phi_analytical : array_like
        Analytical solution φ(x,t).
    phi_schemes_n_labels : list of (phi_array, label)
        Each entry is (numerical φ, its label).
    ax : matplotlib.axes.Axes
        The axes on which to draw.
    dom : tuple (xmin, xmax), optional
        x‐axis limits.
    ran : tuple (ymin, ymax), optional
        y‐axis limits.
    """
    # set default limits if not provided
    if dom is None:
        dom = (x.min(), x.max())
    if ran is None:
        all_vals = np.hstack([phi_initial, phi_analytical] + [phi for phi, _ in phi_schemes_n_labels])
        ran = (all_vals.min(), all_vals.max())

    ax.set_xlim(dom)
    ax.set_ylim(ran)

    # initial + analytic
    ax.plot(x, phi_initial, "--", alpha=0.5, label="Initial Condition")
    ax.plot(x, phi_analytical,    alpha=0.5, label="Analytic")

    # numerical schemes
    for phi_num, label in phi_schemes_n_labels:
        ax.plot(x, phi_num, label=label)

    ax.set_title(f"T={nt*dt:.2f}, nt={nt}, $\Delta t$={dt}")
    ax.legend()

    return ax



CNCS_advdif_recipe = {
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

Source_Imp_recipe = {
    "Source": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1},
    "BTCS Imp": {"xi_a": 1, "xi_d": 1, "eta_a": 1, "eta_d": 1, "eta_s": 0},
}

Imp_Source_recipe={
    "BTCS Imp": {"xi_a": 1, "xi_d": 1, "eta_a": 1, "eta_d": 1, "eta_s": 0},
    "Source": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1},
}


Exp_Source_recipe = {
    "FTCS Exp":{"xi_a": 0, "xi_d": 0, "eta_a": 1, "eta_d": 1, "eta_s": 0},
    "Source": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1},
    }

Source_CN_Source_recipe ={
    "Source1": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1/2},
    "CNCS advdif": {
        "xi_a": 1 / 2,
        "xi_d": 1 / 2,
        "eta_a": 1,
        "eta_d": 1,
        "eta_s": 0,
    },
    "Source2": {"xi_a": 0, "xi_d": 0, "eta_a": 0, "eta_d": 0, "eta_s": 1/2},
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

CNCS_ADA_SourceOnD_recipe = {
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
        "eta_s": 1,
    },
    "CNCS adv2": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
}

CNCS_EI_ADA_SourceOnD_recipe = {
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
        "eta_s": 1,
    },
    "CNCS Iadv": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 0,
    },
}

CNCS_ADA_SourceOnA_recipe = {
    "CNCS adv1": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 1/2,
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
        "eta_s": 1/2,
    },
}


CNCS_EI_ADA_SourceOnAll_recipe = {
    "CNCS Eadv": {
        "xi_a": 0,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 1/3,
    },
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 1/3,
    },
    "CNCS Iadv": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 1/3,
    },
}
CNCS_IE_ADA_SourceOnAll_recipe = {
    "CNCS Eadv": {
        "xi_a": 1,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 1/3,
    },
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 1/3,
    },
    "CNCS Iadv": {
        "xi_a": 0,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 1/3,
    },
}

CNCS_ADA_SourceOnAll_recipe = {
    "CNCS adv1": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 1/3,
    },
    "CNCS dif": {
        "xi_a": 0,
        "xi_d": 1 / 2,
        "eta_a": 0,
        "eta_d": 1,
        "eta_s": 1/3,
    },
    "CNCS adv2": {
        "xi_a": 1 / 2,
        "xi_d": 0,
        "eta_a": 1 / 2,
        "eta_d": 0,
        "eta_s": 1/3,
    },
}
A = 1
k = 1

# Domain size
start_length = 0
end_length = 2*np.pi
L = end_length - start_length
dom = [start_length, end_length]
ran = [-A - 1, A + 1]
# Scheme settings

u = 1
K = 0
nx = 100  # Number of spatial grid points
nt = 1 # Number of time steps per simulation run
endtime = 10
t = endtime
u_values = np.linspace(0, u, 50)  # Range of Courant numbers

phi_analytic = analytical_sine_source_adv_dif
dx = L / nx
dt = endtime / nt

steps = 10
x = np.arange(0, L, dx)

phi_initial = analytical_sine_source_adv_dif(u, K, k, L, A, x, 0)
source = K * np.sin(x) + u * np.cos(x)
use_schemes = [
    #("BTCS LS",Imp_Source_recipe),
    #("BTCS SL",Source_Imp_recipe),
    #("CNCS LSL", CN_Source_CN_recipe),
    #("CNCS SLS", Source_CN_Source_recipe),
    
    ("CNCS ADA Src on A", CNCS_ADA_SourceOnA_recipe),
    ("CNCS ADA Src on D", CNCS_ADA_SourceOnD_recipe),
    ("CNCS ADA Src on All", CNCS_ADA_SourceOnAll_recipe),
    ("CNCS ADA FB Src on All", CNCS_EI_ADA_SourceOnAll_recipe),
    ("CNCS ADA BF Src on All", CNCS_IE_ADA_SourceOnAll_recipe ),
]

errors = {phi_label: [] for phi_label, recipie in use_schemes}
dt_list = []

fig,ax = plt.subplots(1,3,figsize=(20,8),dpi=600)

for i in range(steps):
    phis = []

    dt = endtime / nt
    dt_list.append(dt)
    for scheme_label, scheme_recipie in use_schemes:

        phi_scheme_result = Generic_scheme_solver(
            phi_initial, source, u, K, dt, dx, nt, scheme_recipie
        )
        phi_analytic_result = analytical_sine_source_adv_dif(
            u, K, k, L, A, x, endtime
        )
        phis.append((phi_scheme_result, scheme_label))
        errors[scheme_label].append(
            np.sqrt(np.mean((phi_scheme_result - phi_analytic_result) ** 2))
        )
    print(f"dt = {dt:.4f}, nt = {nt}, nx = {nx}, Courant number = {u * dt / dx:.2f}, Diffusion number = {K*dt/dx**2:.2f}")
    if i == 0:
         plot_scheme(
            x,
            nt,
            dt,
            analytical_sine_source_adv_dif(u, K, k, L, A, x, 0),
            analytical_sine_source_adv_dif(u, K, k, L, A, x, endtime),
            phis,
            ax[0],
            dom,
            ran,
        )
         
    elif i == steps-1:
        ax[1]= plot_scheme(
            x,
            nt,
            dt,
            analytical_sine_source_adv_dif(u, K, k, L, A, x, 0),
            analytical_sine_source_adv_dif(u, K, k, L, A, x, endtime),
            phis,
            ax[1],
            dom,
            ran,
        )
    nt = nt + 1
plot_error(dt_list ,errors, ax[2],log_scale=True,ylabel="$\ell_2$ error norm", xlabel="dt",title="$\ell_2$ error norm against dt")
a: float = 0.05 #Tweakable
dt2_line_yvalues = a*np.array(dt_list)**2
dt_line_yvalues = 10*a*np.array(dt_list)

ax[2].plot(dt_list,dt2_line_yvalues, color='grey',linestyle='--',label="$\propto \Delta t^2$ Line" )
#ax[2].plot(dt_list,dt_line_yvalues, color='grey',linestyle='--',label="$\propto \Delta t$ Line" )
ax[2].hlines(10**-3.1, dt_list[0], dt_list[-1],color='green',linestyle='--',label="Exact Solution Line")
plt.legend()
print("Order of Convergence:")
for name, error_list in errors.items():
    slope = calc_slope(dt_list, error_list)
    print(f"{name}: Order = {slope:.2f}")

fig.savefig("at constant dx.png", dpi=600, bbox_inches="tight")