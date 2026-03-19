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
nvars = 3           # Number of variables in solution vector SV.  Set this manually,
                    #for now

class params:
    i_ext = 0     # External current (A/m2)
    T = 973         # Temperature (K)

# Positions in solution vector
class ptr:
    phi_elyte_an = 0
    phi_elyte_ca = 1
    phi_ca = 2

# Additional parameter calculations:
R = 8.3135              # Universal gas constant, J/mol-K
F = 96485               # Faraday's constant, C/mol of charge
beta = 0.5              # Symmetry parameter
n_elec = 2              # electrons transferred

bnFRT = beta * n_elec * F / R / params.T


"========= INITIALIZE MODEL ========="
SV_0 = np.zeros((nvars,))

# Set initial values, according to your approach:  eg:
SV_0[ptr.phi_ca] = phi_ca_0 # Change this if needed, to fit your ptr approach
SV_0[ptr.phi_elyte_ca] = phi_elyte_0
SV_0[ptr.phi_elyte_an] = phi_elyte_0

phi_an = 0

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
    
    # current system state
    phi_elyte_an = SV[ptr.phi_elyte_an]
    phi_elyte_ca = SV[ptr.phi_elyte_ca]
    phi_ca = SV[ptr.phi_ca]

    phi_an = 0.0  # reference

    # double layer potentials
    delta_phi_dl_an = phi_an - phi_elyte_an
    delta_phi_dl_ca = phi_ca - phi_elyte_ca

    # overpotentials
    eta_an = delta_phi_dl_an - U_an
    eta_ca = delta_phi_dl_ca - U_ca

    # faradaic currents (Butler-Volmer)
    i_far_an = i0_an * (np.exp(-bnFRT * eta_an) - np.exp(bnFRT * eta_an))
    i_far_ca = i0_ca * (np.exp(-bnFRT * eta_ca) - np.exp(bnFRT * eta_ca))

    # Double layer currents
    i_dl_an = -i_ext - i_far_an
    i_dl_ca = -i_ext - i_far_ca

    # ODEs
    d_delta_phi_dl_an_dt = - i_dl_an / C_dl_an
    d_delta_phi_dl_ca_dt = - i_dl_ca / C_dl_ca

    # ODEs
    d_delta_phi_dl_an_dt =  - i_dl_an / C_dl_an
    d_phi_ca_dt 
    d_phi_elyte_ca_dt = d_phi_ca_dt + i_dl_ca / C_dl_ca

    # electrode and electrolyte potentials
    dSV_dt[ptr.phi_elyte_an] = phi_an - d_delta_phi_dl_an_dt
    dSV_dt[ptr.phi_elyte_ca] = d_delta_phi_dl_an_dt
    dSV_dt[ptr.phi_ca] = d_delta_phi_dl_ca_dt - d_delta_phi_dl_an_dt

    return dSV_dt

"========= RUN / INTEGRATE MODEL ========="
# Function call expects inputs (residual function, time span, initial value).
solution = solve_ivp(derivative, [0, .0001], SV_0, args=(params, ptr))


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