import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

YAML_FILE = "sofc.yaml"
REACTION_ID = "edge-f2"

ETA_MIN = 0.0
ETA_MAX = 0.3
NPTS = 200

F = 96485e3           # C/kmol
R = ct.gas_constant   # J/(kmol*K)


gas = ct.Solution(YAML_FILE, "gas")
metal = ct.Solution(YAML_FILE, "metal")
oxide_bulk = ct.Solution(YAML_FILE, "oxide_bulk")

metal_surface = ct.Interface(YAML_FILE, "metal_surface", [gas])
oxide_surface = ct.Interface(YAML_FILE, "oxide_surface", [gas, oxide_bulk])

tpb = ct.Interface(
    YAML_FILE,
    "tpb",
    [metal, metal_surface, oxide_bulk, oxide_surface]
)


# CHARGE-TRANSFER REACTION
rxn_index = None

for i, rxn in enumerate(tpb.reactions()):
    rxn_id = getattr(rxn, "ID", getattr(rxn, "id", ""))

    if rxn_id == REACTION_ID:
        rxn_index = i
        break

if rxn_index is None:
    raise ValueError(f"Reaction ID {REACTION_ID} was not found.")

reaction = tpb.reaction(rxn_index)

print("Selected reaction:")
print(tpb.reaction_equations()[rxn_index])
print("Reactants:", reaction.reactants)
print("Products :", reaction.products)

n = 2

# beta from YAML file.
try:
    beta = reaction.rate.beta
except Exception:
    beta = 0.5


# FUNCTIONS
def set_delta_phi(delta_phi):
    """
    delta_phi = phi_Ni - phi_YSZ

    Only the potential difference matters.
    Set phi_YSZ = 0 and phi_Ni = delta_phi.
    """

    phi_ysz = 0.0
    phi_ni = delta_phi

    metal.electric_potential = phi_ni
    metal_surface.electric_potential = phi_ni

    oxide_bulk.electric_potential = phi_ysz
    oxide_surface.electric_potential = phi_ysz


def get_net_rate(delta_phi):
    """
    Net rate of progress for the selected TPB reaction.
    """

    set_delta_phi(delta_phi)
    return tpb.net_rates_of_progress[rxn_index]


def get_rate_constants():
    """
    Forward and reverse rate constants.
    Handles different Cantera attribute names.
    """

    if hasattr(tpb, "forward_rate_constants"):
        k_f = tpb.forward_rate_constants[rxn_index]
    else:
        k_f = tpb.fwd_rate_constants[rxn_index]

    if hasattr(tpb, "reverse_rate_constants"):
        k_r = tpb.reverse_rate_constants[rxn_index]
    else:
        k_r = tpb.rev_rate_constants[rxn_index]

    return k_f, k_r


def get_concentration_dict():
    """
    Collect species concentrations from all phases involved in the TPB reaction.
    """

    C = {}

    phase_list = [metal, metal_surface, oxide_bulk, oxide_surface, tpb]

    for phase in phase_list:
        for name, conc in zip(phase.species_names, phase.concentrations):
            C[name] = conc

    return C


def calculate_exchange_current_density():
    """
    PDF expression:

    i0 = nF k_f^(1-beta) k_r^beta
         product(C_reactants^((1-beta)*nu_f))
         product(C_products^(beta*nu_r))

    Electron concentration is skipped because the electron contribution
    is handled through the electric potential.
    """

    k_f, k_r = get_rate_constants()
    C = get_concentration_dict()

    reactant_term = 1.0
    product_term = 1.0

    for species, nu in reaction.reactants.items():
        if species == "electron":
            continue

        if species not in C:
            raise KeyError(f"Reactant species {species} not found in concentration dictionary.")

        reactant_term *= C[species] ** ((1.0 - beta) * nu)

    for species, nu in reaction.products.items():
        if species == "electron":
            continue

        if species not in C:
            raise KeyError(f"Product species {species} not found in concentration dictionary.")

        product_term *= C[species] ** (beta * nu)

    i0 = (
        n * F
        * (k_f ** (1.0 - beta))
        * (k_r ** beta)
        * reactant_term
        * product_term
    )

    return i0


# EQUILIBRIUM POTENTIAL DIFFERENCE
phi_values = np.linspace(-2.0, 2.0, 4001)
rate_values = np.zeros_like(phi_values)

for j, phi in enumerate(phi_values):
    rate_values[j] = get_net_rate(phi)

sign_change_indices = np.where(
    np.sign(rate_values[:-1]) != np.sign(rate_values[1:])
)[0]

if len(sign_change_indices) == 0:
    j_min = np.argmin(np.abs(rate_values))
    delta_phi_eq = phi_values[j_min]

else:
    j = sign_change_indices[0]

    phi_1 = phi_values[j]
    phi_2 = phi_values[j + 1]

    rate_1 = rate_values[j]
    rate_2 = rate_values[j + 1]

    delta_phi_eq = phi_1 - rate_1 * (phi_2 - phi_1) / (rate_2 - rate_1)

print("\nEquilibrium result")
print("------------------")
print(f"Delta_phi_eq = {delta_phi_eq:.8f} V")


# PART (a): CURRENT DENSITY i VS OVERPOTENTIAL eta

eta = np.linspace(ETA_MIN, ETA_MAX, NPTS)
current_density = np.zeros_like(eta)

for j, eta_j in enumerate(eta):

    # PDF definition:
    # Delta_phi = Delta_phi_eq + eta
    delta_phi = delta_phi_eq + eta_j

    q_net = get_net_rate(delta_phi)

    # Use this sign to match the Butler-Volmer equation given:
    # i_BV = i0 [ exp(-beta*nF*eta/RT)
    #           - exp((1-beta)*nF*eta/RT) ]
    
    current_density[j] = -n * F * q_net


np.savetxt(
    "part3a_current_density_vs_overpotential.csv",
    np.column_stack((eta, current_density)),
    delimiter=",",
    header="eta_V,current_density",
    comments=""
)


plt.figure(figsize=(7, 5))
plt.plot(eta, current_density, linewidth=2)

plt.xlabel("Overpotential, eta (V)")
plt.ylabel("Current density / current per TPB length")
plt.title("Current density vs overpotential")
plt.grid(True)
plt.tight_layout()
plt.savefig("part3a_current_density_vs_overpotential.png", dpi=300)
plt.show()


# PART (b): BUTLER-VOLMER OVERLAY
T = tpb.T

# Evaluate exchange current density at equilibrium
set_delta_phi(delta_phi_eq)
i0 = calculate_exchange_current_density()

# Butler-Volmer equation exactly as given
current_density_bv = i0 * (
    np.exp(-beta * n * F * eta / (R * T))
    -
    np.exp((1.0 - beta) * n * F * eta / (R * T))
)

np.savetxt(
    "part3b_butler_volmer_overlay.csv",
    np.column_stack((eta, current_density, current_density_bv)),
    delimiter=",",
    header="eta_V,current_density,current_density_BV",
    comments=""
)

plt.figure(figsize=(7, 5))
plt.plot(eta, current_density, linewidth=2, label="Cantera-based calculation")
plt.plot(eta, current_density_bv, "--", linewidth=2, label="Butler-Volmer")

plt.xlabel("Overpotential, eta (V)")
plt.ylabel("Current density / current per TPB length")
plt.title("Butler-Volmer overlay on current density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("part3b_butler_volmer_overlay.png", dpi=300)
plt.show()

# PRINT SUMMARY
print("\nPart III(a,b) summary")
print("---------------------")
print(f"Reaction used                         = {tpb.reaction_equations()[rxn_index]}")
print(f"Temperature, T                        = {T:.2f} K")
print(f"Number of electrons, n                = {n}")
print(f"Charge-transfer coefficient, beta     = {beta:.3f}")
print(f"Delta_phi_eq                          = {delta_phi_eq:.8f} V")
print(f"Exchange current density, i0          = {i0:.6e}")
print(f"Current density at eta = 0 V          = {current_density[0]:.6e}")
print(f"BV current at eta = 0 V               = {current_density_bv[0]:.6e}")
print(f"Current density at eta = 0.3 V        = {current_density[-1]:.6e}")
print(f"BV current at eta = 0.3 V             = {current_density_bv[-1]:.6e}")

print("\nFiles saved:")
print("part3a_current_density_vs_overpotential.png")
print("part3b_butler_volmer_overlay.png")
print("part3a_current_density_vs_overpotential.csv")
print("part3b_butler_volmer_overlay.csv")