
# Additional parameter calculations:
R = 8.3135              # Universal gas constant, J/mol-K
F = 96485               # Faraday's constant, C/mol of charge

"========= LOAD INPUTS AND OTHER PARAMETERS ========="
class params:
    dae_flag = 1        # Is this a DAE (1) or an ODE (0)?

    # Boundary conditions:
    i_ext = 0     # External current (A/m2)
    T = 973         # Temperature (K)

    phi_ca_0 = 1.1      # Initial cathode voltage, relative to anode (V)
    phi_elyte_0 = 0.6   # Initial electrolyte voltage at equilibrium, relative to anode (V)

    sigma_io = 1        # Electrolyte ionic conductivity (S/m)
    dy_elyte = 10e-6    # Electrolyte thickness (m)

    sigma_el = 1.4e7    # Electrical conductivity of Nickel (S/m)

    # Anode parameters:
    dy_an = 1e-3        # Anode thickness (m)
    eps_g_an = .47      # Anode porosity (gas phase vol frac, -)
    eps_el_an = .3      # Anode metal phase vol frac, -)
    eps_io_an = 1 - eps_g_an - eps_el_an
    n_brugg = 0.5       # Bruggeman coefficient


    # Geometry
    npts_an = 3     # Number of finite volumes in the anode
    npts_elyte = 10  # Number of finite volumes in the electrolyte
    npts_ca = 3     # Number of finite volumes in the cathode

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

    # Total number of variables. Only storing electric potential, for now:
    nvars_an = 2
    nvars_elyte = 1
    nvars_ca = 1

    nvars_an_tot = nvars_an * npts_an
    nvars_elyte_tot = nvars_elyte * npts_elyte
    nvars_ca_tot = nvars_ca * npts_ca

    nvars_tot = nvars_an_tot + nvars_ca_tot + nvars_elyte_tot


    # Derived parameters:
    #   Beta*nF/RT for each reaction (Beta*n renamed alpha_fwd, here)
    #   (1-Beta)*nF/RT for each reaction ((1-Beta)*n renamed alpha_rev, here)
    aF_RT_an_fwd = beta_an * n_elec_an * F / R / T
    aF_RT_an_rev = (1- beta_an) * n_elec_an * F / R / T

    aF_RT_ca_fwd = beta_ca * n_elec_ca * F / R / T
    aF_RT_ca_rev = (1- beta_ca) * n_elec_ca * F / R / T