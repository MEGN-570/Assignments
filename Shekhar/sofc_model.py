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
# if params.dae_flag:

#     options =  {'user_data':(params, ptr), 'compute_initcond':'yp0', 'rtol':1e-2,
#                 'atol':1e-4, 'algebraic_vars_idx':[ptr.phi_an_el[1:],
#                 ptr.phi_elyte[1:]]}

#     solver = dae('ida', residual, **options)
#     t_out = np.linspace(0, 1e-3, 10000)
#     # Create an initial array of time derivatives and runs the integrator:
#     SVdot_0 = np.zeros_like(SV_0)
#     # SVdot_0 = -calc_residual(SV_0, SVdot_0, SVdot_0, (params, ptr))
#     solution = solver.solve(t_out, SV_0, SVdot_0)
# else:
#     # Function call expects inputs (residual function, time span, initial value).
#     solution = solve_ivp(residual, [0, .001], SV_0, args=(params, ptr),
#                          method='BDF', rtol = 1e-6, atol = 1e-8)



if params.dae_flag:

    # Algebraic variables:
    # anode electronic potentials except first reference node
    # all electrolyte potentials
    algebraic_idx = list(ptr.phi_an_el[1:]) + list(ptr.phi_elyte)

    options = {
        'user_data': (params, ptr),
        'compute_initcond': 'yp0',
        'rtol': 1e-5,
        'atol': 1e-7,
        'algebraic_vars_idx': algebraic_idx,
        'max_steps': 50000
    }

    solver = dae('ida', residual, **options)

    t_out = np.linspace(0, 1e-3, 1000)

    SVdot_0 = np.zeros_like(SV_0)

    solution = solver.solve(t_out, SV_0, SVdot_0)
    


"========= PLOTTING AND POST-PROCESSING ========="

# # Extract time and solution matrix
# t = np.asarray(solution.values.t)
# Y = np.asarray(solution.values.y)

# # Make sure Y is shaped as [time, variables]
# if Y.shape[0] != len(t):
#     Y = Y.T

"========= PLOTTING AND POST-PROCESSING ========="

# Check solver output before plotting
if not hasattr(solution, "values"):
    raise RuntimeError("DAE solver failed: solution has no values.")

if solution.values.y is None:
    raise RuntimeError("DAE solver failed: solution.values.y is None.")

t = np.asarray(solution.values.t)
Y = np.asarray(solution.values.y)

if Y.ndim == 0 or Y.size == 0:
    raise RuntimeError(
        "DAE solver failed before producing solution data. "
        "Check initial conditions and residual equations."
    )

# Make sure Y is shaped as [time, variables]
if Y.shape[0] != len(t):
    Y = Y.T


# FIGURE 1: Representative SOFC potentials only
fig, ax = plt.subplots()
fig.set_size_inches((5, 3.5))

# Pick only representative potentials, not all variables
phi_an_cc = Y[:, ptr.phi_an_el[0]]       # anode current collector / electronic reference
phi_an_io_mem = Y[:, ptr.phi_an_io[-1]]  # anode ionic phase near membrane
phi_elyte_ca = Y[:, ptr.phi_elyte[-1]]   # electrolyte near cathode
phi_ca = Y[:, ptr.phi_ca[0]]             # cathode potential

ax.plot(1e3 * t, phi_ca, label='Cathode')
ax.plot(1e3 * t, phi_elyte_ca, label='Electrolyte cathode side')
ax.plot(1e3 * t, phi_an_io_mem, label='Anode ionic membrane side')
ax.plot(1e3 * t, phi_an_cc, label='Anode current collector')

ax.set_ylim((lower, upper))
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Cell Potential (V)')
ax.set_title('Potentials')
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
plt.savefig('HW4_representative_potentials.png', dpi=400)
plt.show()


# FIGURE 2: Zoomed anode ionic phase potential
fig, ax = plt.subplots()
fig.set_size_inches((5, 3.5))

for i in range(params.npts_an):
    ax.plot(
        1e3 * t,
        Y[:, ptr.phi_an_io[i]],
        label=f'Anode io {i+1}'
    )

ax.set_xlabel('Time (ms)')
ax.set_ylabel(r'Anode Ionic Potential, $\phi_{an,io}$ (V)')
ax.set_title('Zoomed Anode Ionic Phase Potential')
#ax.legend(frameon=False, fontsize=7, ncol=2)

# Used for zoom
ax.set_xlim((0.3328, 0.3336))
ax.set_ylim((0.3973, 0.4001))


fig.tight_layout()
plt.savefig('HW4_i0_an_5e5.png', dpi=400)
plt.show()