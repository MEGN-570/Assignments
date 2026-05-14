# MEGN570A
# Peter Pajka
# Adapted from sofc_model_template.py (Dr. DeCaluwe)


"=== MODULES ==="
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import font_manager
import numpy as np
from scipy.integrate import solve_ivp


# Plotting formatting (copied from template):
font = font_manager.FontProperties(style='normal', size=12)
ncolors = 3
ind_colors = np.linspace(0, 1.15, ncolors)
colors = np.zeros_like(ind_colors)
cmap = colormaps['plasma']
colors = cmap(ind_colors)


"=== GLOBAL CONSTANTS ==="
R = 8.3135              # universal gas const [J/mol-K]
F = 96485               # Faraday's const [C/mol]
n_elec = 2              # num of electrons transfered
beta = 0.5              # symmetry factor for B–V


"=== AUXILIARY INPUTS/PARAMETERS ==="
phi_ca_0 = 1.1          # initial cathode voltage, relative to anode [V]
phi_elyte_0 = 0.6       # initial electrolyte voltage, relative to anode [V]
phi_an = 0              # anode voltage (reference) [V]
nvars = 2               # num of vars in solution vec


"=== PARAMETERS ==="
class par:
    '''
    *****
    i_ext is set to 0.5 A/cm^2 -> 5000 A/m^2 in assignment
    Needs to be 500 A/m^2 to get correct drop
    Is this a type?
    '''
    i_ext = 5e+2        # external current [A/m^2]

    T = 973.15          # temperature (isothermal) [K]

    U_an = -0.4         # anode equil potential diff [V]
    i_0_an = 5e+2       # anode exchange current density [A/m^2]
    C_dl_an = 5e-2      # anode double layer capacitance [F/m^2]

    U_ca = 0.6          # cathode equil potential diff [V]
    i_0_ca = 1e+2       # cathode exchange current density [A/m^2]
    C_dl_ca = 1         # cathode double layer capacitance [F/m^2]


    '''
    *****
    R_elyte = delta_y_elyte / sigma_ion
    Not provided — guessing R_elyte based on ohmic drop in assignment soln
    '''
    R_elyte = 1e-4      # electrolyte resistance [Ohm-m^2]


"=== POINTERS ==="
# Approach 3:
class ptr:
    phi_dl_an = 0       # double layer potential at anode
    phi_dl_ca = 1       # double layer potential at cathode


"=== INITIALIZE MODEL ==="
SV_0 = np.zeros((nvars,))
# phi_dl_0 = phi_ed_0 - phi_elyte_0
SV_0[ptr.phi_dl_an] = phi_an - phi_elyte_0
SV_0[ptr.phi_dl_ca] = phi_ca_0 - phi_elyte_0


"=== DEFINE BUTLER–VOLMER FUNCTION"
def BV(eta, i_0, par):
    i_Far = i_0 * (
        np.exp(-beta * n_elec * F * eta / (R * par.T))
        - np.exp((1 - beta) * n_elec * F * eta / (R * par.T))
    )
    return i_Far


"=== DEFINE RESIDUAL FUNCTION ==="
def derivative(_, SV, par, ptr):
    dSV_dt = np.zeros_like(SV)  # preallocate

    # Anode:
    eta_an = SV[ptr.phi_dl_an] - par.U_an  # overpotential
    i_Far_an = BV(eta_an, par.i_0_an, par)  # Faradaic current at interface
    i_dl_an = -par.i_ext - i_Far_an  # following assignment convention
    dSV_dt[ptr.phi_dl_an] = -i_dl_an / par.C_dl_an

    # Cathode:
    eta_ca = SV[ptr.phi_dl_ca] - par.U_ca  # overpotential
    i_Far_ca = BV(eta_ca, par.i_0_ca, par)  # Faradaic current at interface
    i_dl_ca = par.i_ext - i_Far_ca
    dSV_dt[ptr.phi_dl_ca] = -i_dl_ca / par.C_dl_ca

    return dSV_dt


"=== RUN / INTEGRATE MODEL ==="
# Function call expects inputs (residual function, time span, initial value).
soln = solve_ivp(
    derivative, 
    [0, 1e-3], 
    SV_0, 
    args=(par, ptr), 
    method='BDF'  # BDF for stability
    )


"=== POST-PROCESSING ==="
# Extract double-layer potentials from soln vec
phi_dl_an = soln.y[0]
phi_dl_ca = soln.y[1]

## Algebraic reltaions
# (1) phi_dl_an = phi_an - phi_elyte_an
# (2) i_ext = (phi_elyte_an - phi_elyte_ca) / R_elyte
# (3) phi_dl_ca = phi_ca - phi_elyte_ca
phi_elyte_an = -phi_dl_an
phi_elyte_ca = phi_elyte_an - par.i_ext * par.R_elyte
phi_ca = phi_dl_ca + phi_elyte_ca


"=== PLOTTING ==="
# Define labels for legend
labels = [r'$\phi_{elyte,an}$',
          r'$\phi_{elyte,ca}$',
          r'$\phi_{ca}$']

# Create fig and set size
fig, ax = plt.subplots()
fig.set_size_inches((4, 3))  # made larger

# Set color palette
ax.set_prop_cycle(
    'color', 
    [plt.cm.plasma(i) for i in np.linspace(0.25, 1, nvars+1)]
    )

# Plot each variable (time in ms)
ax.plot(1e3*soln.t, phi_elyte_an, label=labels[0])
ax.plot(1e3*soln.t, phi_elyte_ca, label=labels[1])
ax.plot(1e3*soln.t, phi_ca, label=labels[2])

# Label the axes
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Potential (V)')

plt.ylim(-0.1, 1.3)  # expand ylim

# Create legend
ax.legend(
    frameon=False, 
    fontsize=12, 
    loc='upper right', 
    labelspacing=-0.25  # reduce line spacing
    )

fig.tight_layout()  # clean up whitespace
# plt.savefig('HW2_results.png', dpi=400)  # save fig
plt.show()  # show fig