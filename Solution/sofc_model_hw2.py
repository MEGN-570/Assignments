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

# Plotting formatting:
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=10)
ncolors = 3 # how many colors?
ind_colors = np.linspace(0, 1.15, ncolors)
colors = np.zeros_like(ind_colors)
cmap = colormaps['plasma']
colors = cmap(ind_colors)

"========= LOAD INPUTS AND OTHER PARAMETERS ========="
phi_ca_0 = 1.1      # Initial cathode voltage, relative to anode (V)
phi_elyte_0 = 0.6   # Initial electrolyte voltage at equilibrium, relative to anode (V)
nvars = 3           # Number of variables in solution vector SV.  Set this manually,
                    #   for now
sigma_io = 0.08     # Electrolyte ionic conductivity (S/m)
dy_elyte = 10e-6    # Electrolyte thickness (m)

class params:
    # Boundary conditions:
    i_ext = 0     # External current (A/m2)
    T = 973         # Temperature (K)

    # Equilibrium potentials:
    E_an = -0.4     # Equilibrium potential at anode interface (anode - elyte, V)
    E_ca = 0.6      # Equilibrium potential at cathode interface (cathode - elyte, V)

    # Kinetics:
    i_o_ca = 0.1e3  # Cathode exchange current density, A/m2 of total SOFC area.
    i_o_an = 0.5e3  # Anode exchange current density, A/m2 of total SOFC area.

    n_elec_an = 2   # Number of electrical charge transferred per mol rxn, anode rxn
    n_elec_ca = 2   # Number of electrical charge transferred per mol rxn, cathode rxn

    beta_an = 0.5   # Symmetry parameter, anode charge transfer
    beta_ca = 0.5   # Symmetry parameter, cathode charge transfer

    # Double layer:
    C_dl_an = 5e-2  # anode-electrolyte interface capacitance, F/m2 total SOFC area.
    C_dl_ca = 1e0   # cathode-electrolyte interface capacitance, F/m2 total SOFC area.

# Positions in solution vector
class ptr:
    # Approach 1: store the actual material electric potentials:
    phi_elyte_an = 0
    phi_elyte_ca = 1
    phi_ca = 2

# Additional parameter calculations:
R = 8.3135              # Universal gas constant, J/mol-K
F = 96485               # Faraday's constant, C/mol of charge


# Derived parameters:
#   Beta*nF/RT for each reaction (Beta*n renamed alpha_fwd, here)
#   (1-Beta)*nF/RT for each reaction ((1-Beta)*n renamed alpha_rev, here)
params.aF_RT_an_fwd = params.beta_an * params.n_elec_an * F / R / params.T
params.aF_RT_an_rev = (1- params.beta_an) * params.n_elec_an * F / R / params.T

params.aF_RT_ca_fwd = params.beta_ca * params.n_elec_ca * F / R / params.T
params.aF_RT_ca_rev = (1- params.beta_ca) * params.n_elec_ca * F / R / params.T

# Electric potential drop across the electrolyte:
dPhi_elyte = params.i_ext * dy_elyte / sigma_io

"========= INITIALIZE MODEL ========="
# Initialize the solution vector:
SV_0 = np.zeros((nvars,))

# Set initial values, according to your approach:  eg:
SV_0[ptr.phi_ca] = phi_ca_0 # Change this if needed, to fit your ptr approach

# Electrolyte potential at cathode interface:
SV_0[ptr.phi_elyte_ca] = phi_elyte_0 - 0.5*dPhi_elyte

# Ellectrolyte potential at anode interface:
SV_0[ptr.phi_elyte_an] = phi_elyte_0 + 0.5*dPhi_elyte

"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(_, SV, pars, ptr):

    # Initialize the derivative / residual:
    dSV_dt = np.zeros_like(SV)

    # Anode double layer
    #   Overpotential:
    eta = -SV[ptr.phi_elyte_an] - pars.E_an # Note phi_an = 0
    #   Faradaic current:
    i_Far_an = pars.i_o_an*(exp(-pars.aF_RT_an_fwd*eta) - exp(pars.aF_RT_an_rev*eta))
    #   Double-layer current:
    i_dl_an = -(pars.i_ext + i_Far_an)
    # This is for the electrolyte potential, which is equal to -dPhi_dl_an:
    dSV_dt[ptr.phi_elyte_an] =  i_dl_an / pars.C_dl_an

    # phi_elyte_ca evolves at exactly same rate as phi_elyte_ca:
    dSV_dt[ptr.phi_elyte_ca] = dSV_dt[ptr.phi_elyte_an]

    # Cathode double layer:
    #   Overpotential:
    eta = (SV[ptr.phi_ca]-SV[ptr.phi_elyte_ca]) - pars.E_ca
    #   Faradaic current:
    i_Far_ca = pars.i_o_ca*(exp(-pars.aF_RT_ca_fwd*eta) - exp(pars.aF_RT_ca_rev*eta))
    print(eta, i_Far_ca)
    #   Double-layer current:
    i_dl_ca = pars.i_ext - i_Far_ca
    #   Cathode potential evolves at the rate of the local elyte, minus double layer:
    dSV_dt[ptr.phi_ca] = dSV_dt[ptr.phi_elyte_ca] - i_dl_ca / pars.C_dl_ca

    return dSV_dt

"========= RUN / INTEGRATE MODEL ========="
# Function call expects inputs (residual function, time span, initial value).
solution = solve_ivp(derivative, [0, .001], SV_0, args=(params, ptr), method='BDF',
                     rtol = 1e-6, atol = 1e-8)

"========= PLOTTING AND POST-PROCESSING ========="
# Depending on what you stored in SV, perform any necessary calculations to extract the
#   potentials of:
#       -The electrolyte at the anode interface
#       -The electrolyte at the cathode interface
#       -The cathode.
#   Using 'approach 1' above, these are direclty stored in your solution vector.


# Define the labels for your legend
labels = ['$\phi_{elyte, an}$','$\phi_{elyte, ca}$','$\phi_{ca}$']

# Create the figure:
fig, ax = plt.subplots()
# Set color palette:
ax.set_prop_cycle('color', [plt.cm.plasma(i) for i in np.linspace(0.25,1,nvars+1)])
# Set figure size
fig.set_size_inches((4,3))
# Plot the data, using ms for time units:
ax.plot(1e3*solution.t, solution.y.T, label=labels)

# Set y-axis limits
ax.set_ylim((lower, upper))

# Label the axes
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Cell Potential (V)')

# Create legend
ax.legend(prop=font, frameon=False, loc='upper right', ncols=3)

# Clean up whitespace, etc.
fig.tight_layout()

# Uncomment to save the figure, if you want. Name it however you please:
plt.savefig('HW2_results_500.png', dpi=400)
# Show figure:
plt.show()