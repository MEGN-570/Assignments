# sofc_model_HW3.py

"========= IMPORT MODULES ========="
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct


"========= LOAD INPUTS AND OTHER PARAMETERS ========="
F    = 96485                    # [C/mol]
R    = ct.gas_constant / 1000   # [J/(mol*K)]
n    = 2                        # number of electrons transferred
beta = 0.5                      # symmetry factor

eta_vec = np.linspace(0, 0.3, 50)   # [V]


"========= LOAD CANTERA PHASES ========="
gas        = ct.Solution("sofc.yaml", "gas")
metal      = ct.Solution("sofc.yaml", "metal")
oxide_bulk = ct.Solution("sofc.yaml", "oxide_bulk")

metal_surf = ct.Interface("sofc.yaml", "metal_surface", [gas, metal])
oxide_surf = ct.Interface("sofc.yaml", "oxide_surface", [gas, oxide_bulk])
tpb        = ct.Interface("sofc.yaml", "tpb", [metal, metal_surf, oxide_surf])

# hold surface coverages fixed
metal_surf.coverages = metal_surf.coverages
oxide_surf.coverages = oxide_surf.coverages


"========= IDENTIFY REACTION ========="
print("TPB reactions:")
for i, rxn in enumerate(tpb.reaction_equations()):
    print(f"  {i}: {rxn}")

rxn_idx = 0   # charge-transfer reaction


"========= SET ELECTRIC POTENTIALS ========="
def set_potentials(phi_metal, phi_oxide=0.0):
    metal.electric_potential      = phi_metal
    metal_surf.electric_potential = phi_metal
    oxide_bulk.electric_potential = phi_oxide
    oxide_surf.electric_potential = phi_oxide


"========= FIND DELTA_PHI_EQ ========="
# sweep metal potential to find where net current = 0 (equilibrium)
phi_sweep = np.linspace(-5, 5, 4000)
i_sweep   = []

for phi in phi_sweep:
    set_potentials(phi)
    # i = nF * rop --> current per length [A/cm]
    i_sweep.append(n * F * tpb.net_rates_of_progress[rxn_idx])

i_sweep = np.array(i_sweep)

# find where current changes sign
sc = np.where(np.diff(np.sign(i_sweep)))[0][0]

# linear interpolation between the two points around i = 0
Delta_phi_eq = (phi_sweep[sc]-i_sweep[sc]*(phi_sweep[sc+1]-phi_sweep[sc]) / (i_sweep[sc+1]-i_sweep[sc]))

print(f"\nDelta_phi_eq = {Delta_phi_eq:.4f} V")


"========= COMPUTE CURRENT FROM CANTERA ========="
i_cantera = []

for eta in eta_vec:
    # apply delta_phi = delta_phi_eq + eta
    set_potentials(Delta_phi_eq + eta)

    # convert rop to current: mol/(cm*s) * C/mol = A/cm * 1e2 = A/m
    i_val = n * F * tpb.net_rates_of_progress[rxn_idx] * 1e2
    i_cantera.append(i_val)

i_cantera = np.array(i_cantera)


"========= BUTLER-VOLMER ========="
# i = i0 * [exp((1-beta)nF eta / RT) - exp(-beta nF eta / RT)]

# evaluate everything at equilibrium state (eta = 0)
set_potentials(Delta_phi_eq)

# rate constants for forward/reverse reaction
k_f = tpb.forward_rate_constants[rxn_idx]
k_r = tpb.reverse_rate_constants[rxn_idx]

# stoichiometric coefficients
nu_f = tpb.reactant_stoich_coeffs
nu_r = tpb.product_stoich_coeffs

# collect all surface concentrations into one dict
C_all = {}
for name, c in zip(metal_surf.species_names, metal_surf.concentrations):
    C_all[name] = c
for name, c in zip(oxide_surf.species_names, oxide_surf.concentrations):
    C_all[name] = c

# electron is not treated explicitly so absorb into k_f and k_r
for name in metal.species_names:
    C_all[name] = 1.0

# build mass-action terms
prod_react = 1.0
prod_prod  = 1.0

for j, sp in enumerate(tpb.kinetics_species_names):

    if sp in metal.species_names:
        continue  # skip electron

    c  = C_all.get(sp, 1.0)
    nf = nu_f[j, rxn_idx]   # reactant coefficient
    nr = nu_r[j, rxn_idx]   # product coefficient

    if nf > 0:
        prod_react *= c ** nf
    if nr > 0:
        prod_prod  *= c ** nr

# exchange current density from derived expression
i0 = (n*F*(k_f**(1-beta)) * (k_r**beta) * (prod_react**(1-beta)) * (prod_prod**beta))

i0 *= 1e2   # A/cm to A/m

# BV current (same sign convention as Cantera)
i_BV = i0*(np.exp((1-beta)*n*F*eta_vec/(R*tpb.T)) - np.exp(-beta*n*F*eta_vec/(R*tpb.T)))

print(f"i0 = {i0:.4e} A/m")


"========= PLOTTING ========="
from matplotlib import font_manager
font = font_manager.FontProperties(family='Arial', style='normal', size=12)

fig, ax = plt.subplots()
ax.set_prop_cycle('color', [plt.cm.plasma(i) for i in np.linspace(0.25, 1, 2)])
fig.set_size_inches((4, 3))

ax.plot(eta_vec, i_cantera, label='Cantera')
ax.plot(eta_vec, i_BV, '--', label='Butler-Volmer')

ax.set_xlabel('Overpotential (V)')
ax.set_ylabel('Line Current (A/m)')

ax.legend(prop=font, frameon=False)
fig.tight_layout()

# plt.savefig("sofc_iv_curve.png", dpi=400)
plt.show()