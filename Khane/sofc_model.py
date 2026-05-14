# sofc_model.py
print("HELLO THIS IS THE EDITED SOFC MODEL FILE")
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
print("i_ext =", params.i_ext)
print("sigma_io =", params.sigma_io)
print("i_o_an =", params.i_o_an)

""" Initialize the model:
    - create SV ptr
    - create SV_0
"""
ptr = sofc_init.ptr(params)

SV_0 = sofc_init.initialize(params, ptr)

# "========= RUN / INTEGRATE MODEL ========="
if params.dae_flag:

    options =  {'user_data':(params, ptr), 'compute_initcond':'yp0', 'rtol':1e-2,
            'atol':1e-4, 'algebraic_vars_idx': list(ptr.phi_an_el[1:]) + list(ptr.phi_elyte)}

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

# Zoomed plot of anode ionic phase potentials
fig2, ax2 = plt.subplots()
fig2.set_size_inches((4, 3))

for i in range(params.npts_an):
    ax2.plot(1e3 * solution.values.t, solution.values.y[:, ptr.phi_an_io[i]])

ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Anode Ionic Potential (V)')
ax2.set_title('Anode Ionic Phase Potential')


ax2.set_ylim((0.397, 0.401))

fig2.tight_layout()
# Show figure:
plt.show()