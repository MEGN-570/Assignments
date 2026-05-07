from math import exp
import numpy as np

"========= DEFINE RESIDUAL FUNCTION ========="
def residual(t, SV, dSV_dt, resid, input):
    pars, ptr = input
    # print(pars)

    # Initialize the derivative / residual:
    # resid = np.zeros_like(SV)

    # Effective conductivities:
    sigma_el_eff = (pars.eps_el_an**(1 + pars.n_brugg) * pars.sigma_el)
    sigma_io_eff = (pars.eps_io_an**(1 + pars.n_brugg) * pars.sigma_io)

    # Finite volume thicknesses:
    dy_an_cv = pars.dy_an / pars.npts_an
    dy_elyte_cv = pars.dy_elyte / pars.npts_elyte

    # Anode potential does not change, adjacent to current collector:
    resid[ptr.phi_an_el[0]] = dSV_dt[ptr.phi_an_el[0]]

    # For now, other anode potentials are explicitly set to zero:
    """ Change this, for HW 4: """
    for i in np.arange(pars.npts_an):

        # Read out local potentials:
        phi_an_el = SV[ptr.phi_an_el[i]]
        phi_an_io = SV[ptr.phi_an_io[i]]

        # Electronic current entering node:
        if i == 0:
            i_el_in = pars.i_ext
        else:
            phi_an_el_prev = SV[ptr.phi_an_el[i-1]]
            i_el_in = sigma_el_eff * (phi_an_el_prev - phi_an_el) / dy_an_cv

        # Electronic current leaving node:
        if i == pars.npts_an - 1:
            i_el_out = 0
        else:
            phi_an_el_next = SV[ptr.phi_an_el[i+1]]
            i_el_out = sigma_el_eff * (phi_an_el - phi_an_el_next) / dy_an_cv

        # Ionic current entering node:
        if i == 0:
            i_io_in = 0
        else:
            phi_an_io_prev = SV[ptr.phi_an_io[i-1]]
            i_io_in = sigma_io_eff * (phi_an_io_prev - phi_an_io) / dy_an_cv

        # Ionic current leaving node:
        if i == pars.npts_an - 1:
            phi_elyte = SV[ptr.phi_elyte[0]]
            i_io_out = pars.sigma_io * (phi_an_io - phi_elyte) / dy_elyte_cv
        else:
            phi_an_io_next = SV[ptr.phi_an_io[i+1]]
            i_io_out = sigma_io_eff * (phi_an_io - phi_an_io_next) / dy_an_cv

        # Anode double layer
        #   Overpotential:
        eta = phi_an_el - phi_an_io - pars.E_an

        #   Faradaic current:
        i_Far_an = pars.i_o_an * (exp(-pars.aF_RT_an_fwd * eta) - exp(pars.aF_RT_an_rev * eta))

        #   Double-layer current:
        #   idl = iel,e - iFar - iel,i
        i_dl_an = i_el_out - i_Far_an - i_el_in

        # This is for the electrolyte potential, which is equal to -dPhi_dl_an:
        resid[ptr.phi_an_io[i]] = dSV_dt[ptr.phi_an_io[i]] - i_dl_an / pars.C_dl_an

        # Algebraic equation for anode electronic phase:
        if i > 0:
            resid[ptr.phi_an_el[i]] = i_io_in - i_io_out + i_el_in - i_el_out

    # Read out the electrolyte potential in the anode at the electrolyte membrane
    #   interface. Load this as the "previous" potential for the first electrolyte
    #   volume:
    phi_elyte_o = SV[ptr.phi_an_io[-1]]

    """ NO CHANGE FOR HW 4 TO ANYTHING BELOW.  BUT THIS CODE MIGHT BE HELPFUL FOR YOUR ALGEBRAIC EQUATIONS TO BE IMPLEMENTED ABOVE."""
    for i in np.arange(pars.npts_elyte):

        # Read out electric potential of current node:
        phi_elyte_1 = SV[ptr.phi_elyte[i]]

        # Calculate ionic current into this node:
        i_io = pars.sigma_io * (phi_elyte_o - phi_elyte_1) / dy_elyte_cv

        # Algebraic equation: i_io should equal i_ext, for charge neutrality:
        resid[ptr.phi_elyte[i]] = i_io - pars.i_ext

        # Save current phi_elyte as the new 'previous' phi_elyte for the next iteration
        #   of the loop:
        phi_elyte_o = phi_elyte_1

    # Cathode double layer:
    #   Overpotential:
    eta = (SV[ptr.phi_ca[0]] - SV[ptr.phi_elyte[-1]]) - pars.E_ca

    #   Faradaic current:
    i_Far_ca = pars.i_o_ca * (exp(-pars.aF_RT_ca_fwd * eta) - exp(pars.aF_RT_ca_rev * eta))

    #   Double-layer current:
    i_dl_ca = pars.i_ext - i_Far_ca

    #   Cathode potential evolves at the rate of the local elyte, minus double layer:
    resid[ptr.phi_ca] = dSV_dt[ptr.phi_ca] - dSV_dt[ptr.phi_elyte[-1]] + i_dl_ca / pars.C_dl_ca