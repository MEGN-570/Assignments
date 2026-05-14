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
phi_ca_0 = 1.1      # Initial cathode voltage, relative to anode (V)
phi_elyte_0 = 0.6   # Initial electrolyte voltage at equilibrium, relative to anode (V)
nvars = 3           # Number of variables in solution vector SV.  Set this manually,
                    #for now

class params:
    i_ext = 500     # External current (A/m2)
    T = 973         # Temperature (K)

# Positions in solution vector
class ptr:
    # Approach 1: store the actual material electric potentials:
    phi_elyte_an = 0
    phi_elyte_ca = 1
    phi_ca = 2

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

    # # Approach 4, 5, 6, etc...

# Additional parameter calculations:
R = 8.3135              # Universal gas constant, J/mol-K
F = 96485               # Faraday's constant, C/mol of charge


"========= INITIALIZE MODEL ========="
SV_0 = np.zeros((nvars,))
# Set initial values, according to your approach:  eg:
SV_0[ptr.phi_ca] = phi_ca_0 # Change this if needed, to fit your ptr approach
# Add the other values:

"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(_, SV, params, ptr):
    dSV_dt = np.zeros_like(SV)

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
labels = ['$\phi_{elyte, an}$','$\phi_{elyte, ca}$','$\phi_{ca}$']

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