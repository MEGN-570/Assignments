import cantera as ct
print(f"Running Cantera version: {ct.__version__}")

input_file = "sofc_new.yaml"


gas = ct.Solution("sofc_new.yaml")
print(gas.species_names)

gas = ct.Solution(input_file, "gas")
metal = ct.Solution(input_file, "metal")                    # Ni bulk (anode)
oxide = ct.Solution(input_file, "oxide_bulk")               # YSZ bulk (cathode)
metal_surface = ct.Solution(input_file, "metal_surface")    # Ni surface
oxide_surface = ct.Solution(input_file, "oxide_surface")    # YSZ surface
tpb = ct.Solution(input_file, "tpb")     

print(metal_surface.species_names)
print(tpb.species_names)