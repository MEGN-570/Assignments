import numpy as np
class ptr():

    def __init__(self, params):
        # Store the actual material electric potentials:
        self.phi_an_el = np.arange(0, params.nvars_an_tot, params.nvars_an)
        self.phi_an_io = np.arange(1, params.nvars_an_tot, params.nvars_an)
        self.phi_elyte = np.arange(params.nvars_an_tot,
                            params.nvars_an_tot + params.nvars_elyte_tot,
                            params.nvars_elyte)
        self.phi_ca = np.arange(params.nvars_an_tot + params.nvars_elyte_tot,
                        params.nvars_tot,
                        params.nvars_ca)

# def initialize(params, ptr):

#     # Electric potential drop across the electrolyte (from Ohm's Law)
#     # dPhi_elyte = params.i_ext * params.dy_elyte /params.sigma_io

#     "========= INITIALIZE MODEL ========="
#     # Initialize the solution vector:
#     SV_0 = np.zeros((params.nvars_tot,))

#     # Set initial values, according to your approach:  eg:
#     SV_0[ptr.phi_ca] = params.phi_ca_0 # Change this if needed, to fit your ptr approach

#     SV_0[ptr.phi_an_io] = params.phi_elyte_0

#     for i in range(params.npts_elyte):
#         SV_0[ptr.phi_elyte[i]] = params.phi_elyte_0

#     return SV_0

def initialize(params, ptr):

    "========= INITIALIZE MODEL ========="

    # Initialize the solution vector
    SV_0 = np.zeros((params.nvars_tot,))

    # Anode electronic potential is reference
    SV_0[ptr.phi_an_el] = 0.0

    # Initialize anode ionic potential close to anode equilibrium
    # eta_an = phi_an_el - phi_an_io - E_an = 0
    # phi_an_io = phi_an_el - E_an
    phi_an_io_eq = 0.0 - params.E_an

    SV_0[ptr.phi_an_io] = phi_an_io_eq

    # Initialize electrolyte potential close to anode ionic potential
    # Include approximate Ohmic drop if i_ext is nonzero
    dy_elyte_cv = params.dy_elyte / params.npts_elyte
    
    for i in range(params.npts_elyte):
        SV_0[ptr.phi_elyte[i]] = (
            phi_an_io_eq
            - (i + 1) * params.i_ext * dy_elyte_cv / params.sigma_io
        )

    # Initialize cathode near cathode equilibrium
    # eta_ca = phi_ca - phi_elyte_last - E_ca = 0
    # phi_ca = phi_elyte_last + E_ca
    SV_0[ptr.phi_ca] = SV_0[ptr.phi_elyte[-1]] + params.E_ca

    return SV_0