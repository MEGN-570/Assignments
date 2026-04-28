from math import exp
import numpy as np

"========= DEFINE RESIDUAL FUNCTION ========="
def residual(t, SV, dSV_dt, resid, input):
    pars, ptr = input
    # print(pars)

    # Initialize the derivative / residual:
    # resid = np.zeros_like(SV)

    # Anode potential does not change.  Currently set to zero, which assumes very high
    #   electrical conductivity.
    """ Change this, for HW 4: """
    for i in np.arange(pars.npts_an):
        resid[ptr.phi_an_el[i]] = SV[ptr.phi_an_el[i]]

    # For now, all charge transfer is in the last anode volume.  Assume that the
    #   electolyte phase electric potential does not change, for all other volumes.
    """ Change this, for HW 4: You can merge this with the loop above and give all Volumes a common loop."""
    for i in np.arange(pars.npts_an-1):
        resid[ptr.phi_an_io[i]] = dSV_dt[ptr.phi_an_io[i]]

    # Anode double layer
    #   Overpotential:
    """ For HW 4, all anode volumes have an i_Far and i_dl.  Move these inside the spatial loop, above, and add a new algebraic eq for phi_an_el"""
    eta = -SV[ptr.phi_an_io[-1]] - pars.E_an # Note phi_an = 0
    #   Faradaic current:
    i_Far_an = pars.i_o_an*(exp(-pars.aF_RT_an_fwd*eta) - exp(pars.aF_RT_an_rev*eta))
    #   Double-layer current:
    i_dl_an = -(pars.i_ext + i_Far_an)
    # This is for the electrolyte potential, which is equal to -dPhi_dl_an:
    resid[ptr.phi_an_io[-1]] = dSV_dt[ptr.phi_an_io[-1]] -  i_dl_an / pars.C_dl_an

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
