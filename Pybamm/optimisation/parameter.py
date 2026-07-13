
#%%
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pybop
#%%

model = pybamm.lithium_ion.DFN(options = {"surface form":"differential"})
parameter_values = pybamm.ParameterValues("Chayambuka2022")
parameter_values["Negative electrode double-layer capacity [F.m-2]"]= 0.05
parameter_values["Positive electrode double-layer capacity [F.m-2]"]= 0.05
frequencies = np.logspace(-4,4,200)
print(parameter_values)
#%%
simulator_syn = pybop.pybamm.EISSimulator(model=model,parameter_values= parameter_values, f_eval= np.asarray(frequencies))
solution= simulator_syn.solve()


Z =solution["Impedance"].data


#%%
chayam_positive = parameter_values[
    "Positive particle diffusivity [m2.s-1]"
]
chayam_negative = parameter_values[
    "Negative particle diffusivity [m2.s-1]"
]

sto = np.linspace(0.001, 0.999, 500)
T = 298.15

D_positive = np.array([
    parameter_values.evaluate(
        chayam_positive(pybamm.Scalar(s), pybamm.Scalar(T))
    )
    for s in sto
], dtype=float)

D_negative = np.array([
    parameter_values.evaluate(
        chayam_negative(pybamm.Scalar(s), pybamm.Scalar(T))
    )
    for s in sto
], dtype=float)

plt.figure(figsize=(7, 5))
plt.semilogy(sto, D_positive[:,0,0], label="Positive electrode — NVPF")
plt.semilogy(sto, D_negative[:,0,0], label="Negative electrode — hard carbon")
plt.xlabel("Particle stoichiometry")
plt.ylabel(r"Particle diffusivity [m$^2$ s$^{-1}$]")
plt.title("Chayambuka2022 particle diffusivities at 298.15 K")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("chayambukadiffusivity.pdf",bbox_inches = "tight")
plt.show()



dictionary_SPM = {
    "Upper voltage cut-off [V]": 4.2,
    "Negative electrode thickness [m]": 6.4e-05,
    "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Negative electrode Bruggeman coefficient (electrode)": 0,
    "Reference temperature [K]": 298.15,
    "Electrode height [m]": 0.000254,
    "Initial temperature [K]": 298.15,
    "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Positive electrode Bruggeman coefficient (electrode)": 0,
    "Separator thickness [m]": 2.5e-05,
    "Maximum concentration in negative electrode [mol.m-3]": 14540,
    "Initial concentration in electrolyte [mol.m-3]": 1000,
    "Electrode width [m]": 1,
    "Number of cells connected in series to make a battery": 1.0,
    "Number of electrodes connected in parallel to make a cell": 1.0,
    "Maximum concentration in positive electrode [mol.m-3]": 15320,
    "Separator Bruggeman coefficient (electrolyte)": 1.5,
    "Lower voltage cut-off [V]": 2.0,
    "Nominal cell capacity [A.h]": 0.003,
    "Positive electrode thickness [m]": 6.8e-05,

    "Initial concentration in negative electrode [mol.m-3]": 13520,
    "Negative particle radius [m]": 3.48e-06,
    "Negative electrode OCP [V]":None,
    "Negative electrode active material volume fraction": 0.489,
    "Negative electrode OCP entropic change [V.K-1]": 0,
    "Positive electrode exchange-current density [A.m-2]":
        None,

    # Not supplied by the Chayambuka parameter set
    "Negative electrode double-layer capacity [F.m-2]": None,

    "Ambient temperature [K]": 298.15,
    "Negative electrode exchange-current density [A.m-2]":
       j0_nmo,
    "Positive electrode OCP entropic change [V.K-1]": 0,
    "Positive particle diffusivity [m2.s-1]":
       None,
    "Negative particle diffusivity [m2.s-1]":
        None,
    "Current function [A]": 0.003,
    "Negative electrode porosity": 0.51,
    "Initial concentration in positive electrode [mol.m-3]": 3320,
    "Positive electrode porosity": 0.23,
    "Positive electrode OCP [V]": None,
    "Separator porosity": 0.55,

    # Not supplied by the Chayambuka parameter set
    "Positive electrode double-layer capacity [F.m-2]": None,

    "Positive particle radius [m]": 5.9e-07,
    "Positive electrode active material volume fraction": 0.55,
}

model = pybamm.lithium_ion.SPM(options = {"surface form":"differential"})
parameter_values = pybamm.ParameterValues("Chayambuka2022")
parameter_values.update({
    "Initial concentration in negative electrode [mol.m-3]":
        parameter_values[
            "Initial concentration in positive electrode [mol.m-3]"
        ],

    "Maximum concentration in negative electrode [mol.m-3]":
        parameter_values[
            "Maximum concentration in positive electrode [mol.m-3]"
        ],

    "Negative electrode Bruggeman coefficient (electrode)":
        parameter_values[
            "Positive electrode Bruggeman coefficient (electrode)"
        ],

    "Negative electrode Bruggeman coefficient (electrolyte)":
        parameter_values[
            "Positive electrode Bruggeman coefficient (electrolyte)"
        ],

    "Negative electrode OCP [V]":
        parameter_values["Positive electrode OCP [V]"],

    "Negative electrode OCP entropic change [V.K-1]":
        parameter_values[
            "Positive electrode OCP entropic change [V.K-1]"
        ],

    "Negative electrode active material volume fraction":
        parameter_values[
            "Positive electrode active material volume fraction"
        ],

    "Negative electrode charge transfer coefficient":
        parameter_values[
            "Positive electrode charge transfer coefficient"
        ],

    "Negative electrode conductivity [S.m-1]":
        parameter_values[
            "Positive electrode conductivity [S.m-1]"
        ],

    "Negative electrode exchange-current density [A.m-2]":
        parameter_values[
            "Positive electrode exchange-current density [A.m-2]"
        ],

    "Negative electrode porosity":
        parameter_values["Positive electrode porosity"],

    "Negative electrode thickness [m]":
        parameter_values["Positive electrode thickness [m]"],

    "Negative particle diffusivity [m2.s-1]":
        parameter_values["Positive particle diffusivity [m2.s-1]"],

    "Negative particle radius [m]":
        parameter_values["Positive particle radius [m]"],
})
parameter_values["Negative electrode double-layer capacity [F.m-2]"]=0.005
parameter_values["Positive electrode double-layer capacity [F.m-2]"]=0.05
#parameter_values["Positive particle diffusivity [m2.s-1]"]=0.02
#parameter_values["Negative particle diffusivity [m2.s-1]"]=parameter_values["Positive particle diffusivity [m2.s-1]"]


model = pybamm.lithium_ion.DFN(options = {"surface form":"differential"})
print(model.options)
eis_sim = pybamm.EISSimulation(model,parameter_values=parameter_values)
frequencies = np.logspace(-4,4,200)
result = eis_sim.solve(frequencies)
#print(list(result.data.keys())) 
Z_re = result["Z_re [Ohm]"]
Z_im = result["Z_im [Ohm]"]
freq= result["Frequency [Hz]"]
print(len(Z_im))
#plt.plot(Z_re,-Z_im)


# ---— Sensitivity screening: which parameters actually move the impedance?
# =============================================================================
 
def eis_sensitivity_screen(base_parameter_values, params_to_test, frequencies,
                           factor=2.0, model_options=None, plot=True):
    """
    One-at-a-time (OAT) sensitivity screen for EIS.
 
    For each parameter p, the impedance spectrum is recomputed with p*factor
    and p/factor (everything else held at its base value). The sensitivity
    index is the mean relative change of the complex impedance across the
    spectrum, normalised per decade of parameter change:
 
        S = mean_f( |Z_perturbed(f) - Z_base(f)| / |Z_base(f)| ) / |log10(factor)|
 
    S ~ 0        -> parameter is invisible to EIS, exclude from optimisation
    S >~ 0.05    -> parameter noticeably shapes the spectrum, worth fitting
 
    Parameters
    ----------
    base_parameter_values : pybamm.ParameterValues
        Your full baseline parameter set (plain floats, no pybop.Parameters).
    params_to_test : list[str]
        Names of the parameters to screen.
    frequencies : np.ndarray
        Frequency array [Hz].
    factor : float
        Multiplicative perturbation (default 2.0 -> tests p/2 and 2p).
    model_options : dict
        Options for the DFN model (default {"surface form": "differential"}).
    plot : bool
        If True, produce a bar chart of S and Nyquist overlays.
 
    Returns
    -------
    dict {parameter name: sensitivity index S}, sorted descending.
    """
    model_options = model_options or {"surface form": "differential","contact resistance":"true"}
    frequencies = np.asarray(frequencies, dtype=float)
 
    def run_eis(pvals):
        """Build a fresh model + simulator and return complex Z(f)."""
        m = pybamm.lithium_ion.DFN(options=model_options)
        sim = pybop.pybamm.EISSimulator(
            m, parameter_values=pvals, f_eval=frequencies
        )
        sol = sim.solve()
        return np.asarray(sol["Impedance"].data)
 
    # Baseline spectrum
    Z_base = run_eis(base_parameter_values.copy())
    absZ = np.abs(Z_base)
 
    sensitivities = {}
    spectra = {}  # keep perturbed spectra for the Nyquist overlay plot
 
    for name in params_to_test:
        base_val = base_parameter_values[name]
        S_vals = []
        spectra[name] = []
        for f in (factor, 1.0 / factor):
            pvals = base_parameter_values.copy()
            pvals[name] = base_val * f
            try:
                Z_pert = run_eis(pvals)
                mask = frequencies < 10.0
                rel_change = np.mean(np.abs(Z_pert[mask] - Z_base[mask]) / absZ[mask])
                S_vals.append(rel_change / abs(np.log10(factor)))
                spectra[name].append((f, Z_pert))
            except Exception as err:
                print(f"  [warn] {name} x{f:g} failed to solve: {err}")
        sensitivities[name] = float(np.mean(S_vals)) if S_vals else np.nan
        print(f"{name}: S = {sensitivities[name]:.4g}")
 
    # Sort descending
    sensitivities = dict(
        sorted(sensitivities.items(), key=lambda kv: -np.nan_to_num(kv[1]))
    )
 
    if plot:
        # --- Bar chart of sensitivity indices ---
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(sensitivities) + 1.5))
        names = list(sensitivities.keys())
        vals = [sensitivities[n] for n in names]
        ax.barh(names[::-1], vals[::-1])
        ax.set_xlabel(
            "Sensitivity index S (mean relative |Z| change per decade)"
        )
        ax.set_title(f"EIS parameter sensitivity (x{factor:g} perturbation)")
        ax.set_xscale("log")
        plt.tight_layout()
        plt.show()
 
        # --- Nyquist overlays for the top 4 most sensitive parameters ---
        top = [n for n in names if np.isfinite(sensitivities[n])][:4]
        if top:
            fig, axes = plt.subplots(
                1, len(top), figsize=(4.5 * len(top), 4), squeeze=False
            )
            for ax, name in zip(axes[0], top):
                ax.plot(Z_base.real, -Z_base.imag, "k-", lw=2, label="base")
                for f, Zp in spectra[name]:
                    ax.plot(Zp.real, -Zp.imag, "--", label=f"x{f:g}")
                ax.set_xlabel(r"$Z_{re}$ [$\Omega$]")
                ax.set_ylabel(r"$-Z_{im}$ [$\Omega$]")
                ax.set_title(name, fontsize=8)
                ax.legend(fontsize=7)
                ax.axis("equal")
            plt.tight_layout()
            
            plt.show()
 
    return sensitivities
 
 

  
# ---- Example usage ----------------------------------------------------------
parameter_values = pybamm.ParameterValues(dictionary)
parameter_values_tested = { 
       #PARTICLE
        #"Negative particle radius [m]": 3e-06,
        #"Positive particle radius [m]": 3e-06,
        "Negative particle diffusivity [m2.s-1]":1,
        "Positive particle diffusivity [m2.s-1]":1,
        #TEMPERATURE
        #"Initial temperature [K]":298.15,
        #"Ambient temperature [K]": 293.15,
        #"Reference temperature [K]":293.15,
        #ELECTRODE DATA
        "Initial concentration in negative electrode [mol.m-3]":c_init,
        "Initial concentration in positive electrode [mol.m-3]":c_init,
        #"Negative electrode conductivity [S.m-1]": 22.8,
        "Positive electrode conductivity [S.m-1]": 22.8,
        #"Positive electrode exchange-current density [A.m-2]":j0_nmo,
        #"Negative electrode exchange-current density [A.m-2]":j0_nmo,
        "Maximum concentration in negative electrode [mol.m-3]": 3.6e4,
        "Maximum concentration in positive electrode [mol.m-3]": 3.6e4,
        "Negative electrode porosity": 0.4,
        "Positive electrode porosity": 0.4,
        #"Negative electrode thickness [m]": 5e-05,
        #"Positive electrode thickness [m]": 5e-05,
        #"Negative electrode Bruggeman coefficient (electrode)": 0,
        #"Positive electrode Bruggeman coefficient (electrode)": 0,
        #"Negative electrode OCP entropic change [V.K-1]": 0,
        #"Positive electrode OCP entropic change [V.K-1]": 0,
        #"Negative electrode OCP [V]":func,
        #"Positive electrode OCP [V]":func,
        "Negative electrode thickness [m]": 5e-05,
        "Positive electrode thickness [m]": 5e-05,
        "Negative electrode active material volume fraction":0.6, 
        "Positive electrode active material volume fraction":0.6,
        "Electrode height [m]": 0.000254,
        "Electrode width [m]":0.001,
        "Electrode cross-sectional area [m2]": 1.5e-04,
       #ELECTROLYTE DATA
        "Electrolyte conductivity [S.m-1]":1e-8,
        "Electrolyte diffusivity [m2.s-1]":2e-10,
        "Initial concentration in electrolyte [mol.m-3]":1,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.875,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.875,
        #SEPARATOR
        "Separator porosity": 0.5,
        "Separator thickness [m]": 2.6e-04,
        "Separator Bruggeman coefficient (electrolyte)":1.5,
        #MISC
        "Thermodynamic factor":1,
        "Cation transference number": 0.4,
        #"Current function [A]":0.003,
        # "Nominal cell capacity [A.h]": 2.6e-03,
        #"Number of cells connected in series to make a battery":1,
        #"Number of electrodes connected in parallel to make a cell":1,
        #"Lower voltage cut-off [V]":2.3 ,
        #"Upper voltage cut-off [V]":3.6,
        #GUESSES
        "Negative electrode double-layer capacity [F.m-2]":10,
        "Positive electrode double-layer capacity [F.m-2]":10,
        "Contact resistance [Ohm]": 12,
        }
#
parameter_values = pybamm.ParameterValues(dictionary)
S = eis_sensitivity_screen(
     base_parameter_values=parameter_values,  # your baseline dict
     params_to_test=parameter_values_tested,
     frequencies=np.asarray(data_frequency),   # coarser grid = faster screening
     factor=2.0,
 )

 # Parameters worth optimising:
worth_fitting = [k for k, v in S.items() if v > 0.05]
print("Include in optimisation:", worth_fitting)
#settting up optimisation procedure and cost functio
#thoughts;
#----Electrolyte diffusivity: very sensitive for overall shape
#doulbe layer capacity: sensitive for the semicircle
# Electrolyte conductivity
 
# %%

print(problem.parameters.names)
param_names=  [""]
def results_to_latex(param_names, true_values, opt_values,
                     caption="Optimised parameter values against ground truth.",
                     label="tab:opt_results"):
    """
    Prints a LaTeX (booktabs) table: parameter | true value | optimised value
    | relative error [%]. Copy the printed output straight into Overleaf.
 
    Add \\usepackage{booktabs} and \\usepackage{siunitx} to your preamble.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Parameter & True value & Optimised value & Error [\%] \\",
        r"\midrule",
    ]
    for name, true, opt in zip(param_names, true_values, opt_values):
        err = 100.0 * abs(opt - true) / abs(true)
        # \num{} (siunitx) renders 2e-10 as 2 x 10^-10 automatically
        lines.append(
            rf"{name} & \num{{{true:.4g}}} & \num{{{opt:.4g}}} & {err} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)
    print(latex)
    return latex
 

#%%
print(problem.parameters.names)
# ---- Example usage (edit to match problem.parameters.names ordering!) ------
print(problem.parameters.names)   # <-- run this first, order must match
param_names = [
     r"Electrolyte conductivity $\sigma_e$ [\si{S.m^{-1}}]",
     r"Electrolyte diffusivity $D_e$ [\si{m^2.s^{-1}}]",
     r"Neg. double-layer capacity $C_{dl}^-$ [\si{F.m^{-2}}]",
          r"Pos. double-layer capacity $C_{dl}^+$ [\si{F.m^{-2}}]", ]
true_values = [0.88,2e-10,0.5,0.5]
opt_values  = result.x            # or list(result.x), same order as names
results_to_latex(param_names, true_values, opt_values)
 