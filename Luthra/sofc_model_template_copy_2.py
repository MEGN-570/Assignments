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
phi_an_0 = 0        # Initial anode voltage
phi_ca_0 = 1.2      # Initial cathode voltage, relative to anode (V)
phi_elyte_0 = 0.6   # Initial electrolyte voltage at equilibrium, relative to anode (V)
nvars = 2           # Number of variables in solution vector SV.  Set this manually, for now

class params:
    i_ext = 0     # External current (A/m2)
    T = 973         # Temperature (K)

# Positions in solution vector
class ptr:
    phi_dl_an = 0
    phi_dl_ca = 1

# Additional parameter calculations:
R = 8.3135              # Universal gas constant, J/mol-K
F = 96485               # Faraday's constant, C/mol of charge
beta = 0.5              # Symmetry parameter
n_elec = 1              # electrons transferred


"========= INITIALIZE MODEL ========="
SV_0 = np.zeros((nvars,))
# Set initial values, according to your approach:  eg:
SV_0[ptr.phi_dl_an] = phi_elyte_0
SV_0[ptr.phi_dl_ca] = phi_elyte_0

# Add the other values:
U_an = -0.4
i0_an = 5e-2
C_dl_an = 5e-6

U_ca = 0.6
i0_ca = 1e-2
C_dl_ca = 1e-4


"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(_, SV, params, ptr):

    # initalize incremental updates
    dSV_dt = np.zeros_like(SV)

    # reading in parameters
    i_ext = params.i_ext
    T = params.T
    

    phi_elyte_an = SV[ptr.phi_elyte_an]
    phi_elyte_ca = SV[ptr.phi_elyte_ca]
    phi_ca = SV[ptr.phi_ca]

    phi_dl_an = -phi_elyte_an
    phi_dl_ca = phi_ca - phi_elyte_ca

    eta_an = phi_dl_an - U_an
    eta_ca = phi_dl_ca - U_ca

    i_far_an = i0_an * (np.exp(-beta * n_elec * F * eta_an / (R * T)) - np.exp((1 - beta) * n_elec * F * eta_an / (R * T)))

    i_far_ca = i0_ca * (np.exp(-beta * n_elec * F * eta_ca / (R * T)) - np.exp((1 - beta) * n_elec * F * eta_ca / (R * T)))

    i_dl_an = i_ext - i_far_an
    i_dl_ca = - i_ext - i_far_ca

    dSV_dt[ptr.phi_elyte_an] = -i_dl_an / C_dl_an
    dSV_dt[ptr.phi_elyte_ca] = -i_dl_ca / C_dl_ca
    dSV_dt[ptr.phi_ca] = 0

    return dSV_dt

"========= RUN / INTEGRATE MODEL ========="
# Function call expects inputs (residual function, time span, initial value).
solution = solve_ivp(derivative, [0, .0001], SV_0, args=(params, ptr))

phi_elyte_an = phi_an_0 - solution[ptr.phi_dl_an]
phi_elyte_ca = solution[ptr.phi_dl_an] + solution[ptr.phi_dl_ca]
phi_ca = 


"========= PLOTTING AND POST-PROCESSING ========="
# Depending on what you stored in SV, perform any necessary calculations to extract the
#   potentials of:
#       -The electrolyte at the anode interface
#       -The electrolyte at the cathode interface
#       -The cathode.
#   Using 'approach 1' above, these are direclty stored in your solution vector.





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