import numpy as np
from scipy.optimize import fsolve
import time
import matplotlib.pyplot as plt
import cantera as ct
print(f"Running Cantera version: {ct.__version__}")

input_file = "sofc_new.yaml"

gas = ct.Solution(input_file, "gas")
metal = ct.Solution(input_file, "metal")                    # Ni bulk (anode)
oxide = ct.Solution(input_file, "oxide_bulk")               # YSZ bulk (cathode)
metal_surface = ct.Solution(input_file, "metal_surface")    # Ni surface
oxide_surface = ct.Solution(input_file, "oxide_surface")    # YSZ surface
tpb = ct.Solution(input_file, "tpb")                        # triple phase boundary

R = ct.gas_constant # J/kmol-K
n = 2               # electrons transferred
beta = 0.5 
F = 96485e3         # C/Kmol
T = 300             # K
P = ct.one_atm     

phases = [gas, metal, oxide, metal_surface, oxide_surface, tpb]
for ph in phases:
    ph.TP = T,P

k_f = tpb.forward_rate_constants[0]
k_r = tpb.reverse_rate_constants[0]

i_Hm = metal_surface.species_names.index("H(m)")
i_Oox = oxide_surface.species_names.index("O''(ox)")
i_OHm = metal_surface.species_names.index('OH(m)')
i_m = metal_surface.species_names.index('(m)')

C_Hm = metal_surface.concentrations[i_Hm]
C_Oox = oxide_surface.concentrations[i_Oox]
C_OHm = metal_surface.concentrations[i_OHm]

# --- Compute equilibrium potential difference ---
# Using mass-action equilibrium condition
delta_phi_eq = (R * T) / (n * F) * np.log((k_f * C_Hm * C_Oox) / (k_r * C_OHm))

print(f"Delta phi_eq = {delta_phi_eq:.4e} V")

eta = np.linspace(0, 0.3, 100)
current = []

for e in eta:
    delta_phi = delta_phi_eq + e

    # Set potentials (only difference matters)
    metal.electric_potential = delta_phi
    oxide.electric_potential = 0.0

    # Update rates
    rop_f = tpb.forward_rates_of_progress[0]
    rop_r = tpb.reverse_rates_of_progress[0]

    rop_net = rop_f - rop_r

    # Current density
    i_val = n * F * rop_net
    current.append(i_val)

current = np.array(current)

# --- Plot ---
plt.figure()
plt.plot(eta, current, label='Cantera')
plt.xlabel('Overpotential η (V)')
plt.ylabel('Current density (A/m^2)')
plt.legend()
plt.grid()
plt.show()