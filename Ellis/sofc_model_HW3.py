# sofc_model.py

"========= IMPORT MODULES ========="
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
phi_ca_0 = 1.1          # Initial cathode voltage, relative to anode (V)
phi_elyte_0 = 0.6       # Initial electrolyte voltage at equilibrium, relative to anode (V)
nvars = 2               # Number of variables in solution vector SV.  Set this manually

class params:
    i_ext = 0           # [A/m^2] External current (positive = load on cell)
    T = 973             # [K] Operating temperature (700 C)
    
    beta = 0.5          # [N/A] Symmetry factor in Butler-Volmer equation
    n = 2               # [N/A] Number of electrons transferred in the reaction
    
    # Electrode reversible potentials
    U_an = -0.4         # [V] Anode reversible potential vs. reference
    U_ca = 0.6          # [V] Cathode reversible potential vs. reference
    
    # Exchange current densities
    i0_an = 5e2         # [A/m^2] Anode exchange current density
    i0_ca = 1e2         # [A/m^2] Cathode exchange current density
    
    # Double-layer capacitances
    C_dl_an = 5e-2      # [F/m^2] Anode double-layer capacitance
    C_dl_ca = 1         # [F/m^2] Cathode double-layer capacitance
    
    # Guessing at the value to match plots in assignment
    R_elyte = 1e-4      # [Ohm*m^2] Electrolyte resistance

# Positions in solution vector
class ptr:
    # Approach 1: store the actual material electric potentials:
    # phi_elyte_an = 0
    # phi_elyte_ca = 1
    # phi_ca = 2

    # # Approach 2: store the double layer potentials, plus the electrolyte potential
    # # at the cathode interface:
    # phi_dl_an = 0
    # phi_dl_ca = 1
    # phi_ca = 2

    # Approach 3: store the double layer potentials ONLY; handle the electrolyte
    #   completely external to the integration:
    #   NOTE: SET nvars = 2, FOR THIS APPROACH
    phi_dl_an = 0
    phi_dl_ca = 1

    # # Approach 4, 5, 6, etc...


# Additional parameter calculations:
R = 8.3135      # [J/mol-K] Universal gas constant
F = 96485       # [C/mol] Faraday's constant, charge per mole of electrons


"========= INITIALIZE MODEL ========="
SV_0 = np.zeros((nvars,))
# Set initial values, according to your approach:
SV_0[ptr.phi_dl_an] = 0 - phi_elyte_0
SV_0[ptr.phi_dl_ca] = phi_ca_0 - phi_elyte_0


"========= BUTLER VOLMER ========="
def butler_volmer(i0, eta, p):
    # prevent numerical overflow (physically activation region only)
    eta_lim = np.clip(eta, -0.3, 0.3)

    return i0 * (np.exp(-p.beta*p.n*F*eta_lim/(R*p.T)) - np.exp((1-p.beta)*p.n*F*eta_lim/(R*p.T)))


"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(_, SV, params, ptr):
    dSV_dt = np.zeros_like(SV)

    phi_dl_an = SV[ptr.phi_dl_an]
    phi_dl_ca = SV[ptr.phi_dl_ca]

    # Overpotentials
    eta_an = phi_dl_an - params.U_an
    eta_ca = phi_dl_ca - params.U_ca

    # Faradaic currents
    i_Far_an = butler_volmer(params.i0_an, eta_an, params)
    i_Far_ca = butler_volmer(params.i0_ca, eta_ca, params)

    # Anode
    i_dl_an = -params.i_ext - i_Far_an
    phi_dl_an = -i_dl_an / params.C_dl_an
    
    # Cathode
    i_dl_ca = params.i_ext - i_Far_ca
    phi_dl_ca = -i_dl_ca / params.C_dl_ca    
    
    dSV_dt[ptr.phi_dl_an] = phi_dl_an
    dSV_dt[ptr.phi_dl_ca] = phi_dl_ca

    return dSV_dt


"========= RUN / INTEGRATE MODEL ========="
# Function call expects inputs (residual function, time span, initial value).
solution = solve_ivp(derivative, [0, .0001], SV_0, args=(params, ptr))


"========= POST-PROCESSING & PLOTTING ========="
# Extract potentials from soln vec
phi_dl_an = solution.y[0]
phi_dl_ca = solution.y[1]
phi_an = 0

# Electrolyte potentials
phi_elyte_an = phi_an - phi_dl_an
phi_elyte_ca = phi_elyte_an - (params.i_ext * params.R_elyte)
phi_ca = phi_dl_ca + phi_elyte_ca

# Define the labels for your legend
labels = ['$\phi_{elyte, an}$','$\phi_{elyte, ca}$','$\phi_{ca}$']

# Create the figure:
fig, ax = plt.subplots()
# Set color palette:
ax.set_prop_cycle('color', [plt.cm.plasma(i) for i in np.linspace(0.25,1,nvars+1)])
# Set figure size
fig.set_size_inches((4,3))
# Plot the data, using ms for time units:
plot_vars = np.vstack((phi_elyte_an, phi_elyte_ca, phi_ca)).T
ax.plot(1e3*solution.t, plot_vars, label=labels)

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