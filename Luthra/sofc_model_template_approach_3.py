# sofc_model.py

"========= IMPORT MODULES ========="
from math import exp
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import font_manager
import numpy as np
from scipy.integrate import solve_ivp

# Plotting formatting:
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=12)
ncolors = 3 # how many colors?
ind_colors = np.linspace(0, 1.15, ncolors)
colors = np.zeros_like(ind_colors)
cmap = colormaps['plasma']
colors = cmap(ind_colors)

"========= LOAD INPUTS AND OTHER PARAMETERS ========="
phi_ca_0 = 1.2      # Initial cathode voltage, relative to anode (V)
phi_elyte_0 = 0.6   # Initial electrolyte voltage at equilibrium, relative to anode (V)
nvars = 2           # Number of variables in solution vector SV.  Set this manually,
                    #for now

class params:
    i_ext = 500     # External current (A/m2)
    T = 973         # Temperature (K)

# Positions in solution vector
class ptr:
    # Approach 1: store the actual material electric potentials:

    # # Approach 2: store the double layer potentials, plus the electrolyte potential
    # #   at the cathode interface:
    # phi_dl_an = 0
    # phi_elyte_ca = 1
    # phi_dl_ca = 2

    # # Approach 3: store the double layer potentials ONLY; handle the electrolyte
    # #   completely external to the integration:
    # #   NOTE: SET nvars = 2, FOR THIS APPROACH
    # phi_dl_an = 0
    # phi_dl_ca = 1

    phi_dl_an = 0
    phi_dl_ca = 1
    # # Approach 4, 5, 6, etc...

# Additional parameter calculations:
R = 8.3135              # Universal gas constant, J/mol-K
F = 96485               # Faraday's constant, C/mol of charge
beta = 0.5              # Symmetry parameter
n_elec = 1              # electrons transferred


"========= INITIALIZE MODEL ========="
SV_0 = np.zeros((nvars,))
# Set initial values, according to your approach:  eg:
SV_0[ptr.phi_dl_an] = U_an
SV_0[ptr.phi_dl_ca] = U_ca
# Add the other values:



"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(_, SV, params, ptr):

    dSV_dt = np.zeros_like(SV)

    i_ext = params.i_ext
    T = params.T

    U_an = -0.4
    i0_an = 5e-2
    C_dl_an = 5e-6

    U_ca = 0.6
    i0_ca = 1e-2
    C_dl_ca = 1e-4

    phi_dl_an = SV[ptr.phi_dl_an]
    phi_dl_ca = SV[ptr.phi_dl_ca]

    eta_an = phi_dl_an - U_an
    eta_ca = phi_dl_ca - U_ca

    # optional stability
    eta_an = np.clip(eta_an, -1, 1)
    eta_ca = np.clip(eta_ca, -1, 1)

    i_far_an = i0_an * (
        np.exp(-beta * n_elec * F * eta_an / (R * T)) -
        np.exp((1 - beta) * n_elec * F * eta_an / (R * T))
    )

    i_far_ca = i0_ca * (
        np.exp(-beta * n_elec * F * eta_ca / (R * T)) -
        np.exp((1 - beta) * n_elec * F * eta_ca / (R * T))
    )

    i_dl_an = i_ext - i_far_an
    i_dl_ca = - i_ext - i_far_ca

    dSV_dt[ptr.phi_dl_an] = - i_dl_an / C_dl_an
    dSV_dt[ptr.phi_dl_ca] = i_dl_ca / C_dl_ca
    dSV_dt[ptr.ph_ca] = 0

    return dSV_dt

"========= RUN / INTEGRATE MODEL ========="
# Function call expects inputs (residual function, time span, initial value).
solution = solve_ivp(derivative, [0, 1e-4], SV_0, args=(params, ptr),max_step=1e-8)


"========= PLOTTING AND POST-PROCESSING ========="
# Depending on what you stored in SV, perform any necessary calculations to extract the
#   potentials of:
#       -The electrolyte at the anode interface
#       -The electrolyte at the cathode interface
#       -The cathode.
#   Using 'approach 1' above, these are direclty stored in your solution vector.

phi_dl_an = solution.y[0]
phi_dl_ca = solution.y[1]

phi_elyte_an = -phi_dl_an
phi_elyte_ca = phi_ca_0 - phi_dl_ca
phi_ca = phi_ca_0 * np.ones_like(phi_dl_an)


# Define the labels for your legend
labels = [r'$\phi_{elyte, an}$', r'$\phi_{elyte, ca}$', r'$\phi_{ca}$']

# Create the figure:
fig, ax = plt.subplots()
# Set color palette:
ax.set_prop_cycle('color', [plt.cm.plasma(i) for i in np.linspace(0.25,1,nvars+1)])
# Set figure size
fig.set_size_inches((4,3))
# Plot the data, using ms for time units:
ax.plot(1e3*solution.t, solution.y.T, label=labels)

# Label the axes
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Cell Potential (V)')

# Create legend
ax.legend(prop=font, frameon=False)

# Clean up whitespace, etc.
fig.tight_layout()

# Uncomment to save the figure, if you want. Name it however you please:
plt.savefig('HW2_results.png', dpi=400)
# Show figure:
plt.show()