from math import exp
import numpy as np

"========= DEFINE RESIDUAL FUNCTION ========="
def residual(t, SV, dSV_dt, resid, input):
    pars, ptr = input

    # Initialize residual
    resid[:] = 0

    # Bruggernman correlation for effective anode conductivities, Eq. (6):
    sigma_an_eff_io = pars.eps_io_an**(1 + pars.n_brugg) * pars.sigma_io
    sigma_an_eff_el = pars.eps_el_an**(1 + pars.n_brugg) * pars.sigma_el

    # Initialize current arrays
    i_el_an = np.zeros(pars.npts_an + 1)  # add 1 for faces
    i_io_an = np.zeros(pars.npts_an + 1)


    # FVM setup:
    '''<Electronic phase>'''
    '''<Ionic phase>'''
    # Current collector BC:
    i_el_an[0] = sigma_an_eff_el * (0 - SV[ptr.phi_an_el[0]]) / pars.dy_an
    i_io_an[0] = 0

    # Interior points:
    for i in range(1, pars.npts_an):
        i_el_an[i] = sigma_an_eff_el * (SV[ptr.phi_an_el[i-1]] - SV[ptr.phi_an_el[i]]) / pars.dy_an
        i_io_an[i] = sigma_an_eff_io * (SV[ptr.phi_an_io[i-1]] - SV[ptr.phi_an_io[i]]) / pars.dy_an

    # Electrolye BC:
    i_el_an[-1] = 0
    i_io_an[-1] = pars.sigma_io * (SV[ptr.phi_an_io[-1]] - SV[ptr.phi_elyte[0]]) / pars.dy_elyte


    for i in range(pars.npts_an):
        eta = SV[ptr.phi_an_el[i]] - SV[ptr.phi_an_io[i]] - pars.E_an  # overpotential Eq. (8)
        i_Far_an = pars.i_o_an * (exp(-pars.aF_RT_an_fwd * eta) - exp(pars.aF_RT_an_rev * eta))  # BV Eq. (7)

        # Charge neutrality (Eq. (1)): div(i_io) + div(i_el) = 0
        resid[ptr.phi_an_el[i]] = ((i_io_an[i] - i_io_an[i+1]) + (i_el_an[i] - i_el_an[i+1]))

        # Double-layer current (Eq. (12)):
        i_dl_an = i_el_an[i+1] - i_Far_an - i_el_an[i]
        
        # Double-layer capacitive charging (Eq. 10):
        resid[ptr.phi_an_io[i]] = (
            dSV_dt[ptr.phi_an_io[i]]
            - dSV_dt[ptr.phi_an_el[i]]
            - i_dl_an / pars.C_dl_an
        )


    # Read out the electrolyte potential in the anode at the electrolyte membrane
    #   interface. Load this as the "previous" potential for the first electrolyte
    #   volume:
    phi_elyte_o = SV[ptr.phi_an_io[-1]]

    """ NO CHANGE FOR HW 4 TO ANYTHING BELOW.  BUT THIS CODE MIGHT BE HELPFUL FOR YOUR ALGEBRAIC EQUATIONS TO BE IMPLEMENTED ABOVE."""
    for i in np.arange(pars.npts_elyte):
        # Read out electric potential of current node:
        phi_elyte_1 = SV[ptr.phi_elyte[i]]

        # Calculate ionic current into this node:
        i_io = pars.sigma_io*(phi_elyte_o - phi_elyte_1)/pars.dy_elyte

        # Algebraic equation: i_io should equal i_ext, for charge neutrality:
        resid[ptr.phi_elyte[i]] = i_io - pars.i_ext

        # Save current phi_elyte as the new 'previous' phi_elyte for the next iteration
        #   of the loop:
        phi_elyte_o = phi_elyte_1

    # Cathode double layer:
    #   Overpotential:
    eta = (SV[ptr.phi_ca[0]]-SV[ptr.phi_elyte[-1]]) - pars.E_ca

    #   Faradaic current:
    i_Far_ca = pars.i_o_ca*(exp(-pars.aF_RT_ca_fwd*eta) - exp(pars.aF_RT_ca_rev*eta))

    #   Double-layer current:
    i_dl_ca = pars.i_ext - i_Far_ca

    #   Cathode potential evolves at the rate of the local elyte, minus double layer:
    resid[ptr.phi_ca] = (dSV_dt[ptr.phi_ca] - dSV_dt[ptr.phi_elyte[-1]]
                         + i_dl_ca / pars.C_dl_ca)
