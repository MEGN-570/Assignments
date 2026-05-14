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

def initialize(params, ptr):

    # Electric potential drop across the electrolyte (from Ohm's Law)
    # dPhi_elyte = params.i_ext * params.dy_elyte /params.sigma_io

    "========= INITIALIZE MODEL ========="
    # Initialize the solution vector:
    SV_0 = np.zeros((params.nvars_tot,))

    # Set initial values, according to your approach:  eg:
    SV_0[ptr.phi_ca] = params.phi_ca_0 # Change this if needed, to fit your ptr approach

    SV_0[ptr.phi_an_io] = params.phi_elyte_0

    for i in range(params.npts_elyte):
        SV_0[ptr.phi_elyte[i]] = params.phi_elyte_0

    return SV_0
