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


# Import your inputs:
from sofc_inputs import params

# Positions in solution vector
class ptr:
    # Approach 1: store the actual material electric potentials:
    phi_an = np.arange(0, params.nvars_an_tot, params.nvars_an)
    phi_elyte = np.arange(params.nvars_an_tot,
                         params.nvars_an_tot + params.nvars_elyte_tot,
                         params.nvars_elyte)
    phi_ca = np.arange(params.nvars_an_tot + params.nvars_elyte_tot,
                       params.nvars_tot,
                       params.nvars_ca)


# Electric potential drop across the electrolyte (from Ohm's Law)
dPhi_elyte = params.i_ext * params.dy_elyte /params.sigma_io

"========= INITIALIZE MODEL ========="
# Initialize the solution vector:
SV_0 = np.zeros((params.nvars_tot,))

# Set initial values, according to your approach:  eg:
SV_0[ptr.phi_ca] = params.phi_ca_0 # Change this if needed, to fit your ptr approach

for i in range(params.npts_elyte):
    SV_0[ptr.phi_elyte[i]] = (params.phi_elyte_0 - ((2*i + 1) - params.npts_elyte)
                              * dPhi_elyte / (2*params.npts_elyte))


"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(_, SV, pars, ptr):

    # Initialize the derivative / residual:
    dSV_dt = np.zeros_like(SV)

    # Anode potential does not change.  Leave those at zero.  Assumes very high
    #   electrical conductivity.


    # Anode double layer
    #   Overpotential:
    eta = -SV[ptr.phi_elyte[0]] - pars.E_an # Note phi_an = 0
    #   Faradaic current:
    i_Far_an = pars.i_o_an*(exp(-pars.aF_RT_an_fwd*eta) - exp(pars.aF_RT_an_rev*eta))
    #   Double-layer current:
    i_dl_an = -(pars.i_ext + i_Far_an)
    # This is for the electrolyte potential, which is equal to -dPhi_dl_an:
    dSV_dt[ptr.phi_elyte[0]] =  i_dl_an / pars.C_dl_an

    # Cathode double layer:
    #   Overpotential:
    eta = (SV[ptr.phi_ca[0]]-SV[ptr.phi_elyte[-1]]) - pars.E_ca
    #   Faradaic current:
    i_Far_ca = pars.i_o_ca*(exp(-pars.aF_RT_ca_fwd*eta) - exp(pars.aF_RT_ca_rev*eta))

    #   Double-layer current:
    i_dl_ca = pars.i_ext - i_Far_ca
    #   Cathode potential evolves at the rate of the local elyte, minus double layer:
    dSV_dt[ptr.phi_ca] =  - i_dl_ca / pars.C_dl_ca

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


# Plotting formatting:
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=10)
ncolors = 3 # how many colors?
ind_colors = np.linspace(0, 1.15, ncolors)
colors = np.zeros_like(ind_colors)
cmap = colormaps['plasma']
colors = cmap(ind_colors)

# Define the labels for your legend
labels = ['$\phi_{elyte, an}$','$\phi_{elyte, ca}$','$\phi_{ca}$']

# Create the figure:
fig, ax = plt.subplots()
# Set color palette:
ax.set_prop_cycle('color',
                  [plt.cm.plasma(i) for i in np.linspace(0.25,1,params.nvars_tot+1)])
# Set figure size
fig.set_size_inches((4,3))
# Plot the data, using ms for time units:
ax.plot(1e3*solution.t, solution.y.T)#, label=labels)

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