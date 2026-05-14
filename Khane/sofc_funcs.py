from math import exp
import numpy as np

"========= DEFINE RESIDUAL FUNCTION ========="
def residual(t, SV, dSV_dt, resid, input):
    pars, ptr = input

    # Anode potential does not change
    resid[ptr.phi_an_el[0]] = dSV_dt[ptr.phi_an_el[0]]

    # Loop over all anode volumes:
    for i in np.arange(pars.npts_an):

        phi_el = SV[ptr.phi_an_el[i]]
        phi_io = SV[ptr.phi_an_io[i]]

        # Ionic current IN :
        if i == 0:
            i_io_in = 0.0  
        else:
            i_io_in = pars.sigma_io_eff * (SV[ptr.phi_an_io[i-1]] - phi_io) / pars.dy_an_node

        # Ionic current OUT :
        if i == pars.npts_an - 1:
            i_io_out = pars.sigma_io_eff * (phi_io - SV[ptr.phi_elyte[0]]) / pars.dy_an_node
        else:
            i_io_out = pars.sigma_io_eff * (phi_io - SV[ptr.phi_an_io[i+1]]) / pars.dy_an_node

        # Electronic current IN:
        if i == 0:
            i_el_in = pars.i_ext  
        else:
            i_el_in = pars.sigma_el_eff * (SV[ptr.phi_an_el[i-1]] - phi_el) / pars.dy_an_node

        # Electronic current OUT :
        if i == pars.npts_an - 1:
            i_el_out = 0.0 
        else:
            i_el_out = pars.sigma_el_eff * (phi_el - SV[ptr.phi_an_el[i+1]]) / pars.dy_an_node

        # Butler-Volmer Faradaic current:
        eta = phi_el - phi_io - pars.E_an
        i_Far = pars.i_o_an * (exp(-pars.aF_RT_an_fwd * eta) - exp(pars.aF_RT_an_rev * eta))

        # Double-layer current :
        i_dl = i_el_out - i_Far - i_el_in

        # Algebraic eq for phi_an_el :
        if i == 0:
            pass 
        else:
            resid[ptr.phi_an_el[i]] = i_io_in - i_io_out + i_el_in - i_el_out

        # Differential eq for phi_an_io (double-layer capacitance):
        resid[ptr.phi_an_io[i]] = (dSV_dt[ptr.phi_an_io[i]]
                                   - dSV_dt[ptr.phi_an_el[i]]
                                   - i_dl / pars.C_dl_an)

    # Electrolyte :
    phi_elyte_o = SV[ptr.phi_an_io[-1]]

    for i in np.arange(pars.npts_elyte):
        phi_elyte_1 = SV[ptr.phi_elyte[i]]
        i_io = pars.sigma_io * (phi_elyte_o - phi_elyte_1) / pars.dy_elyte
        resid[ptr.phi_elyte[i]] = i_io - pars.i_ext
        phi_elyte_o = phi_elyte_1

    # Cathode:
    eta = (SV[ptr.phi_ca[0]] - SV[ptr.phi_elyte[-1]]) - pars.E_ca
    i_Far_ca = pars.i_o_ca * (exp(-pars.aF_RT_ca_fwd * eta) - exp(pars.aF_RT_ca_rev * eta))
    i_dl_ca = pars.i_ext - i_Far_ca
    resid[ptr.phi_ca] = (dSV_dt[ptr.phi_ca] - dSV_dt[ptr.phi_elyte[-1]]
                         + i_dl_ca / pars.C_dl_ca)