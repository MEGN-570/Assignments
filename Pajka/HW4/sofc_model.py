# sofc_model.py

lower = -0.05
upper = 1.3

"========= IMPORT MODULES ========="
from math import exp
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import font_manager
import numpy as np
from scipy.integrate import solve_ivp
from scikits.odes.dae import dae
from sofc_funcs import residual
import sofc_init



# Import your inputs:
from sofc_inputs import params

""" Initialize the model:
    - create SV ptr
    - create SV_0
"""
ptr = sofc_init.ptr(params)

SV_0 = sofc_init.initialize(params, ptr)

# "========= RUN / INTEGRATE MODEL ========="
if params.dae_flag:
    # Add index for algebraic terms:
    alg_vars = np.concatenate((ptr.phi_an_el, ptr.phi_elyte))
    alg_vars_idx = alg_vars.tolist()

    options =  {'user_data':(params, ptr), 'compute_initcond':'yp0', 'rtol':1e-2,
                'atol':1e-4, 'algebraic_vars_idx':alg_vars_idx}

    solver = dae('ida', residual, **options)
    t_out = np.linspace(0, 1e-3, 10000)
    # Create an initial array of time derivatives and runs the integrator:
    SVdot_0 = np.zeros_like(SV_0)
    # SVdot_0 = -calc_residual(SV_0, SVdot_0, SVdot_0, (params, ptr))
    solution = solver.solve(t_out, SV_0, SVdot_0)
else:
    # Function call expects inputs (residual function, time span, initial value).
    solution = solve_ivp(residual, [0, .001], SV_0, args=(params, ptr),
                         method='BDF', rtol = 1e-6, atol = 1e-8)



"========= PLOTTING AND POST-PROCESSING ========="
# Depending on what you stored in SV, perform any necessary calculations to extract the
#   potentials of:
#       -The electrolyte at the anode interface
#       -The electrolyte at the cathode interface
#       -The cathode.
#   Using 'approach 1' above, these are direclty stored in your solution vector.


# Plotting formatting:
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=10)

# Create the figure:
fig, ax = plt.subplots()
# Set color palette:
ax.set_prop_cycle('color',
                  [plt.cm.plasma(i) for i in np.linspace(0.25,1,params.nvars_tot+5)]) # Adding five weeds out some of the yellow colors.
# Set figure size
fig.set_size_inches((4,3))
# Plot the data, using ms for time units:
ax.plot(1e3*solution.values.t, solution.values.y)#, label=labels)

# Set y-axis limits
ax.set_ylim((lower, upper))

# Label the axes
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Cell Potential (V)')

# Create legend
# ax.legend(prop=font, frameon=False, loc='upper right', ncols=3)

# Clean up whitespace, etc.
fig.tight_layout()

# Uncomment to save the figure, if you want. Name it however you please:
plt.savefig('HW4_results_0.png', dpi=400)
# Show figure:
plt.show()