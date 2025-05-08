#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:33:48 2025

@author: henryyue
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:25:04 2025

@author: Henry Yue
"""
import numpy as np
from initial_conditions import *
from misc_functions import *
from Scheme_builder import *
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.size":       14,   # base font size for text
    "axes.titlesize":  16,   # axes titles
    "axes.labelsize":  16,   # x- and y-labels
    "xtick.labelsize": 12,   # tick labels
    "ytick.labelsize": 12,
    "legend.fontsize": 16,   # legend text
    "figure.titlesize": 18,  # suptitle, if you use one
})
figure_path = ""
A = 1
k = 1

# Domain size
start_length = 0
end_length = 10
L = end_length - start_length

# Scheme settings
K = 1
u = 1
nx = 10  # Number of spatial grid points
nt = 1  # Number of time steps per simulation run
endtime = 4
t = endtime
u_values = np.linspace(0, u, 50)  # Range of Courant numbers
K_values = np.linspace(0, K, 50)  # Range of Diffusion numbers

phi_analytic = analytical_sine_adv_dif
dx = L / nx
dt = endtime / nt
x = np.arange(0, L, dx)
C_values = u_values * dt / dx
D_values = K_values * dt / (dx**2)

use_schemes = [
    #(CNCS_recipe, "CNCS", 0),
    (CNCS_recipe, "CNCS3", 0),
    #(CNCS_AD_recipe, "CNCS AD", "CNCS"),
    #(CNCS_DA_recipe, "CNCS DA", "CNCS"),
    (CNCS_ADA_recipe, "CNCS ADA", "CNCS3"),
    (CNCS_DAD_recipe, "CNCS DAD", "CNCS3"),
    #(CNCS_EI_ADA_recipe, "CNCS EI ADA", "CNCS"),
    #(CNCS_EI_DAD_recipe, "CNCS EI DAD", "CNCS"),
    #(CNCS_IE_ADA_recipe, "CNCS IE ADA", "CNCS"),
    #(CNCS_IE_DAD_recipe, "CNCS IE DAD", "CNCS"),
    

    #(BTCS_recipe, "BTCS", 0),
    #(BTCS_AD_recipe, "BTCS AD", "BTCS"),
    #(BTCS_DA_recipe, "BTCS DA", "BTCS"),
    #(BTCS_ADA_recipe, "BTCS ADA", "BTCS"),
    #(BTCS_DAD_recipe, "BTCS DAD", "BTCS"),

]

results = {
    (u_val, K_val): {
        "analytical": None,
        **{phi_label: None for _, phi_label, _ in use_schemes},
    }
    for u_val in u_values
    for K_val in K_values
}

model_error = {phi_label: [] for _, phi_label, _ in use_schemes}
model_split_error = {phi_label: [] for _, phi_label, _ in use_schemes}

# generate initial condition
phi_init = analytical_sine_adv_dif(u, K, k, L, A, x, 0)

# Generate the analytical solution field
for j, u in enumerate(u_values):
    for i, K in enumerate(K_values):
        phi_true = phi_analytic(u, K, k, L, A, x, nt * dt)
        results[(u, K)]["analytical"] = phi_true

# Generate numerical scheme solution field
for recipie, phi_label, compaired_against in use_schemes:
    error_numerical = np.zeros((len(u_values), len(K_values)))
    for j, u in enumerate(u_values):
        for i, K in enumerate(K_values):
            if phi_label == "CNCS3":
                phi_numerical = Generic_scheme_solver(
                    phi_init.copy(), 0, u, K, endtime / 3, dx, 3, recipie
                )
            else:
                phi_numerical = Generic_scheme_solver(
                    phi_init.copy(), 0, u, K, dt, dx, nt, recipie
                )

            results[u, K][phi_label] = phi_numerical
            error_numerical[i, j] = l2_norm(
                phi_numerical, results[(u, K)]["analytical"]
            )
    model_error[phi_label] = error_numerical

for recipie, phi_label, compaired_against in use_schemes:
    if compaired_against != 0:
        error_split = np.zeros((len(u_values), len(K_values)))
        error_split = (
            (model_error[compaired_against] - model_error[phi_label])
            / model_error[compaired_against].max()
        ) * 100
        model_split_error[phi_label] = error_split

u_mid = (u_values.min() + u_values.max()) / 2
K_mid = (K_values.min() + K_values.max()) / 2

Quadrants = {
    "UL": {phi_label: [] for phi_label in results[(0, 0)].keys()},
    "UR": {phi_label: [] for phi_label in results[(0, 0)].keys()},
    "LR": {phi_label: [] for phi_label in results[(0, 0)].keys()},
    "LL": {phi_label: [] for phi_label in results[(0, 0)].keys()},
}
for u_val in u_values:
    for K_val in K_values:
        for phi_label in results[(u_val, K_val)]:
            result = results[(u_val, K_val)][phi_label]
            if u_val >= u_mid and K_val < K_mid:
                Quadrants["LR"][phi_label].append(result)
            elif u_val < u_mid and K_val < K_mid:
                Quadrants["LL"][phi_label].append(result)
            elif u_val < u_mid and K_val >= K_mid:
                Quadrants["UL"][phi_label].append(result)
            elif u_val >= u_mid and K_val >= K_mid:
                Quadrants["UR"][phi_label].append(result)

for quadrant in Quadrants.keys():
    for phi_label in Quadrants[quadrant].keys():
        profiles = Quadrants[quadrant][phi_label]
        Quadrants[quadrant][phi_label] = np.mean(profiles, axis=0)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# map each quadrant key to its (row, col) in ax
quad2idx = {
    "UL": (0, 0),
    "UR": (0, 1),
    "LL": (1, 0),
    "LR": (1, 1),
}
for recipie, phi_label, compaired_against in use_schemes:
    if compaired_against == 0:
        # Skip concurrent (non-splitting) schemes
        continue

    # Create a figure: left = split error heatmap, right = quadrant diagrams
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # Left: splitting error heatmap
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(
        model_split_error[phi_label],
        origin="lower",
        extent=[C_values.min(), C_values.max(), D_values.min(), D_values.max()],
        aspect="auto",
        cmap="coolwarm",
        norm=CenteredNorm(),
    )
    ax0.set_xlabel("Courant Number (C)")
    ax0.set_ylabel("Diffusion Number (D)")
    ax0.set_title(
        f"{phi_label} (Split Error)",
        fontsize=18,
        fontweight='bold'
    )
    fig.colorbar(
        im,
        ax=ax0,
        shrink=1,
        aspect=20 * 0.7,
        label="Splitting Scheme Error Percentage (%)",
    )

    # Right: quadrant average profiles
    gs_right = gs[1].subgridspec(2, 2, wspace=0.3, hspace=0.3)
    quadrant_keys = ["UL", "UR", "LL", "LR"]
    for idx, quadrant in enumerate(quadrant_keys):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs_right[row, col])
        # Plot initial condition
        ax.plot(x, phi_init, label="Initial Condition", alpha=0.5, color="grey")
        # Analytical solution
        ax.plot(
            x,
            Quadrants[quadrant]["analytical"],
            linestyle="-",
            label="Analytical",
            alpha=1,
            color="b"
        )
        # Splitting scheme result
        ax.plot(
            x,
            Quadrants[quadrant][compaired_against],
            "--",
            label=compaired_against,
            alpha=1
        )
        
        
        ax.plot(
            x,
            Quadrants[quadrant][phi_label],
            "--",
            label=phi_label,
            alpha=1
        )
        ax.set_title(
            quadrant,
            fontsize=16,
            fontweight='bold'
        )
        
        ax.legend()
        
    

    plt.tight_layout()
    filename = f"{phi_label.replace(' ', '_')}_split_samecost_error_quadrants.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved plot as {filename}")

print(f"dt = {dt:.4f}, dx = {dx:.4f}, pe = {u*dx/K}")
