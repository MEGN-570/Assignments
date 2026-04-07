# shekhar_sofc_model.py

"========= IMPORT MODULES ========="
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


"========= LOAD INPUTS AND OTHER PARAMETERS ========="
F = 96485.0
R = 8.314
T = 700.0 + 273.0   # K
n = 2.0
beta = 0.5

# Anode parameters
U_an = -0.4         
i0_an = 5e-2       
Cdl_an = 5e-6

# Cathode parameters
U_ca = 0.6          
i0_ca = 1e-2
Cdl_ca = 1e-4       

# Electrolyte area-specific resistance
R_elyte_asr = 0.09 


"========= INITIALIZE MODEL ========="
# State vector:
# SV[0] = dl_an = phi_an - phi_elyte_an
# SV[1] = dl_ca = phi_ca - phi_elyte_ca

phi_an_0 = 0.0
phi_elyte_an_0 = 0.6
phi_elyte_ca_0 = 0.6
phi_ca_0 = 1.1

dl_an_0 = phi_an_0 - phi_elyte_an_0
dl_ca_0 = phi_ca_0 - phi_elyte_ca_0

SV_0 = np.array([dl_an_0, dl_ca_0], dtype=float)


"========= DEFINE HELPER FUNCTION ========="
def butler_volmer_current(eta, i0):
    """
    Butler-Volmer current from the homework equation:
    iFar = i0 * [exp(-beta*nF*eta/RT) - exp((1-beta)*nF*eta/RT)]
    """
    term_1 = -beta * n * F * eta / (R * T)
    term_2 = (1.0 - beta) * n * F * eta / (R * T)

    # Keep exponentials numerically safe
    term_1 = np.clip(term_1, -100.0, 100.0)
    term_2 = np.clip(term_2, -100.0, 100.0)

    return i0 * (np.exp(term_1) - np.exp(term_2))


"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(t, SV, i_ext):
    dl_an, dl_ca = SV

    # Overpotentials
    eta_an = dl_an - U_an
    eta_ca = dl_ca - U_ca

    # Faradaic currents
    i_far_an = butler_volmer_current(eta_an, i0_an)
    i_far_ca = butler_volmer_current(eta_ca, i0_ca)

    # Double-layer currents using chosen external-current sign convention
    i_dl_an = -i_ext - i_far_an
    i_dl_ca = i_ext - i_far_ca

    # ODEs
    ddl_an_dt = -i_dl_an / Cdl_an
    ddl_ca_dt = -i_dl_ca / Cdl_ca

    return [ddl_an_dt, ddl_ca_dt]


"========= RUN / INTEGRATE MODEL ========="
def run_case(i_ext):
    t_start = 0.0
    t_end = 1e-3
    t_eval = np.linspace(t_start, t_end, 1000)

    solution = solve_ivp(
        fun=lambda t, y: derivative(t, y, i_ext),
        t_span=(t_start, t_end),
        y0=SV_0,
        method='Radau',
        t_eval=t_eval,
        max_step=1e-5,
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution


"========= PLOTTING AND POST-PROCESSING ========="
def recover_potentials(solution, i_ext):
    dl_an = solution.y[0, :]
    dl_ca = solution.y[1, :]

    # states phi_an = 0
    phi_an = 0.0

    # From dl_an = phi_an - phi_elyte_an
    phi_elyte_an = phi_an - dl_an

    # Add a simple electrolyte ohmic drop so the two electrolyte interface potentials are not identical
    phi_elyte_ca = phi_elyte_an - i_ext * R_elyte_asr

    # From dl_ca = phi_ca - phi_elyte_ca
    phi_ca = phi_elyte_ca + dl_ca

    return phi_elyte_an, phi_elyte_ca, phi_ca


def make_plot(solution, i_ext, file_name):
    phi_elyte_an, phi_elyte_ca, phi_ca = recover_potentials(solution, i_ext)
    time_ms = solution.t * 1e3

    plt.figure(figsize=(7, 5))
    plt.plot(time_ms, phi_elyte_an, label=r'$\phi_{elyte,an}$', linewidth=2)
    plt.plot(time_ms, phi_elyte_ca, label=r'$\phi_{elyte,ca}$', linewidth=2)
    plt.plot(time_ms, phi_ca, label=r'$\phi_{ca}$', linewidth=2)

    plt.xlabel('Time (ms)')
    plt.ylabel('Cell Potential (V)')
    plt.title(rf'SOFC charge-transfer response, $i_{{ext}}={i_ext}$ A/cm$^2$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Final values at i_ext = {i_ext} A/cm^2")
    print(f"phi_elyte_an = {phi_elyte_an[-1]:.6f} V")
    print(f"phi_elyte_ca = {phi_elyte_ca[-1]:.6f} V")
    print(f"phi_ca       = {phi_ca[-1]:.6f} V")


"========= MAIN ========="
# Case 1: open circuit
solution_oc = run_case(i_ext=0.0)
make_plot(solution_oc, i_ext=0.0, file_name='sofc_open_circuit.png')

# Case 2: loaded case
solution_load = run_case(i_ext=0.05)
make_plot(solution_load, i_ext=0.05, file_name='sofc_loaded_0p5_Acm2.png')