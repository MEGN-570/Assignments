from math import exp
import numpy as np

"========= DEFINE RESIDUAL FUNCTION ========="
def residual(t, SV, dSV_dt, resid, input):

    pars, ptr = input

    # Effective conductivities in the anode
    sigma_an_el_eff = (pars.eps_el_an ** (1 + pars.n_brugg)) * pars.sigma_el
    sigma_an_io_eff = (pars.eps_io_an ** (1 + pars.n_brugg)) * pars.sigma_io

    # Thickness of one anode finite volume
    dy_an_cv = pars.dy_an / pars.npts_an
    dy_elyte_cv = pars.dy_elyte / pars.npts_elyte


    # Anode Electronic Phase Potentail at Current Collector
    # Keep it fixed with d(phi_an_el,0)/dt = 0.
    resid[ptr.phi_an_el[0]] = dSV_dt[ptr.phi_an_el[0]]

    
    # Distributed Anode Electrochemistry
    # Every anode finite volume now has:
    #   1. electronic current in/out
    #   2. ionic current in/out
    #   3. local Butler-Volmer Faradaic current
    #   4. local double-layer current
    #   5. ionic potential differential equation
    #   6. electronic potential algebraic equation, except at current collector

    for i in np.arange(pars.npts_an):

    
        # Electronic current IN and OUT of Anode Volume i

        # Electronic current entering volume i
        if i == 0:
            # External current enters at the anode current collector
            i_el_i = pars.i_ext
        else:
            # Electronic current from previous anode volume to current volume
            phi_el_left = SV[ptr.phi_an_el[i - 1]]
            phi_el_here = SV[ptr.phi_an_el[i]]
            i_el_i = sigma_an_el_eff * (phi_el_left - phi_el_here) / dy_an_cv

        # Electronic current leaving volume i
        if i == pars.npts_an - 1:
            # No electronic conductor enters the electrolyte membrane
            i_el_e = 0.0
        else:
            # Electronic current from current anode volume to next volume
            phi_el_here = SV[ptr.phi_an_el[i]]
            phi_el_right = SV[ptr.phi_an_el[i + 1]]
            i_el_e = sigma_an_el_eff * (phi_el_here - phi_el_right) / dy_an_cv

       
        #  Ionic current IN and OUT of Anode Volume i
        # Ionic current entering volume i
        if i == 0:
            # No ionic current enters from the anode current collector side
            i_io_i = 0.0
        else:
            # Ionic current from previous anode volume to current volume
            phi_io_left = SV[ptr.phi_an_io[i - 1]]
            phi_io_here = SV[ptr.phi_an_io[i]]
            i_io_i = sigma_an_io_eff * (phi_io_left - phi_io_here) / dy_an_cv

        # Ionic current leaving volume i
        if i == pars.npts_an - 1:
            # Ionic current leaves anode and enters the dense electrolyte.
            # This couples the last anode ionic potential to the first electrolyte node.
            phi_io_here = SV[ptr.phi_an_io[i]]
            phi_elyte_first = SV[ptr.phi_elyte[0]]

            #i_io_e = pars.sigma_io * (phi_io_here - phi_elyte_first) / pars.dy_elyte
            i_io_e = pars.sigma_io * (phi_io_here - phi_elyte_first) / dy_elyte_cv
        else:
            # Ionic current from current anode volume to next anode volume
            phi_io_here = SV[ptr.phi_an_io[i]]
            phi_io_right = SV[ptr.phi_an_io[i + 1]]
            i_io_e = sigma_an_io_eff * (phi_io_here - phi_io_right) / dy_an_cv

        # ==============================================================
        # LOCAL BUTLER-VOLMER FARADAIC CURRENT
        # ==============================================================

        # Local anode overpotential:
        # eta = phi_an_el - phi_an_io - E_an
        eta_an = SV[ptr.phi_an_el[i]] - SV[ptr.phi_an_io[i]] - pars.E_an

        # Local anode Faradaic current
        i_Far_an = pars.i_o_an * (
            exp(-pars.aF_RT_an_fwd * eta_an)
            - exp(pars.aF_RT_an_rev * eta_an)
        )

        # Double Layer Current 
        # 0 = i_dl + i_Far + i_el,i - i_el,e
        # Therefore:
        # i_dl = i_el,e - i_Far - i_el,i
        i_dl_an = i_el_e - i_Far_an - i_el_i
        

        # Anode Ionic Potential Differential Equation
        # d(phi_an_io)/dt = i_dl / C_dl
        # Since phi_an_el is algebraic/fixed, d(phi_an_el)/dt is taken as zero here.
        resid[ptr.phi_an_io[i]] = dSV_dt[ptr.phi_an_io[i]] - i_dl_an / pars.C_dl_an



        # Anode Electronic Potential Algebraic Equation
        # i_io,i - i_io,e + i_el,i - i_el,e = 0
        if i > 0:
            resid[ptr.phi_an_el[i]] = i_io_i - i_io_e + i_el_i - i_el_e
            

    # Electrolyte Potentials 
    
    phi_elyte_o = SV[ptr.phi_an_io[-1]]

    for i in np.arange(pars.npts_elyte):

        # Read out electric potential of current electrolyte node
        phi_elyte_1 = SV[ptr.phi_elyte[i]]

        # Calculate ionic current into this node
       #i_io = pars.sigma_io * (phi_elyte_o - phi_elyte_1) / pars.dy_elyte
        i_io = pars.sigma_io * (phi_elyte_o - phi_elyte_1) / dy_elyte_cv

        # Algebraic equation: ionic current should equal external current
        resid[ptr.phi_elyte[i]] = i_io - pars.i_ext

        # Save current phi_elyte as previous potential for next electrolyte node
        phi_elyte_o = phi_elyte_1

    # Cathode Double Layer
    # Cathode overpotential
    eta_ca = (SV[ptr.phi_ca[0]] - SV[ptr.phi_elyte[-1]]) - pars.E_ca

    # Cathode Faradaic current
    i_Far_ca = pars.i_o_ca * (
        exp(-pars.aF_RT_ca_fwd * eta_ca)
        - exp(pars.aF_RT_ca_rev * eta_ca)
    )

    # Cathode double-layer current
    i_dl_ca = pars.i_ext - i_Far_ca

    # Cathode potential evolves at the rate of local electrolyte potential,
    # minus double-layer contribution
    resid[ptr.phi_ca] = (
        dSV_dt[ptr.phi_ca]
        - dSV_dt[ptr.phi_elyte[-1]]
        + i_dl_ca / pars.C_dl_ca
    )