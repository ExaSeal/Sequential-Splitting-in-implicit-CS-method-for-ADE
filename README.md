# Sequential-Splitting-in-implicit-CS-method-for-ADE
Impact of Sequential Splitting in Implicit Centered-in-Space Schemes For the Advection Diffusion Equation

initial_conditions.py: code for initial conditions for sinusoidal, block (error function). Contains both ADE and Avection equation initial conditions. Required to run other code.

misc_functions.py: code for various plotting, calculation functions, etc, that is not related to the numerical solver.

Scheme_builder.py: code for the ADE core solver, required. To use, generate a scheme in the form of a 'recipie', a dictionary containing  {'abatriary name': {
        "xi_a": int,
        "xi_d": int,
        "eta_a": int,
        "eta_d": int,
        "eta_s": int,
}
'next step'....
}
Customize by adding more steps in the same manner, some recipes are available in the code itself. 

Error Heat Map.py: code for generating the splitting difference heat map.

At constant dx-steady state.py: code for generating the splitting difference heat map under similar computational cost

Jupyter notebooks contain the code used for algerbaic analysis
