#%% --preamble---#

import numpy as np
import pybamm
import pybop
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from myfuncs import func
import pprint
from scipy.interpolate import CubicSpline,make_interp_spline
import os
from pybop.costs.base_cost import BaseCost


#%%
#----contents of file--
#--running full cell for sodium ion cell---#
#----implementing Na-ion paramter values--#

def stoich(Q, data = None, Qmax = None, Qmin= None):
    if Qmax is None:
        Qmax = data["Q"].max()
    if Qmin is None:
        Qmin = data["Q"].min()
    return (Q - Qmin)/ (Qmax - Qmin)

#Data processing
data = pd.read_csv(r"C:\Users\sepps\OneDrive\Oxford\diss\Pybamm\data\NMO_eq_charge.csv")
Q = data["Q"]
V= data["Ewe"]
x= stoich(Q, data=data)
mask = ~np.isnan(x) & ~np.isnan(V)
x_clean = x[mask].to_numpy();V_clean = V[mask].to_numpy()
xc=np.asarray(x_clean,dtype= float).ravel()
xc_mapped = 0.472 * xc + 0.528 #specific mapping for the problem i am running: change based on galens exact 
Vc=np.asarray(V_clean,dtype= float).ravel()
#order = np.argsort(xc_mapped)
#sto = np.linspace(0.528,1,100)
#cubic = CubicSpline(xc_mapped, Vc[::-1])
#plt.plot(xc_mapped,Vc[::-1])
#plt.plot(sto,cubic(sto))
def func(sto):
     return pybamm.Interpolant(xc_mapped,Vc[::-1], sto, interpolator="cubic")
F= 96485.33212
def  j0_nmo(ce, cs_surf, cs_max, T):
    return  F * 3e-11 * np.sqrt(ce) * np.sqrt(cs_surf) * np.sqrt(cs_max - cs_surf)

def j0_nmo_rate(c_e, cs_surf, cs_max, T):
    k = pybamm.Parameter("NMO reaction rate constant [m2.5.mol-0.5.s-1]")
    return F * k * pybamm.sqrt(c_e) * pybamm.sqrt(cs_surf) *pybamm.sqrt(cs_max-cs_surf)
#%%
print(os.listdir(r"C:\Users\sepps\OneDrive\Oxford\diss\Pybamm\data"))
data_EIS  =pd.read_csv(r"C:\Users\sepps\OneDrive\Oxford\diss\Code\Pybamm\data\NMO_symmetrical.csv")
print(data_EIS.head())
print(data_EIS.columns.to_list())
data_frequency  = data_EIS["frequency (Hz)"]
data_Z_re = data_EIS["Z'"]
data_Z_im = data_EIS["Z''"]
idx = np.argsort(data_frequency)
data_frequency = np.asarray(data_frequency)[idx]
data_Z_re      = np.asarray(data_Z_re)[idx]
data_Z_im      = np.asarray(data_Z_im)[idx]
print(data_frequency.max(), data_frequency.min())
print(data_frequency)
plt.plot(data_Z_re[:],data_Z_im[:], "o-")
#%%
#--dictionary of parameter values. Prioritisation in descending order:
# Galen,Fiyanshu,Chayambuka
x_mid = (1+0.528)/2
c_init = x_mid * 3.6e4
print(c_init)
c_min  =0.528 * 3.6e4
c_max = 1 * 3.6e4
dictionary = { 
       #PARTICLE
        "Negative particle radius [m]": 3e-06,
        "Positive particle radius [m]": 3e-06,
        "Negative particle diffusivity [m2.s-1]":1e-10,
        "Positive particle diffusivity [m2.s-1]":1e-10,
        #TEMPERATURE
        "Initial temperature [K]":298.15,
        "Ambient temperature [K]": 298,#reference 293.15,change to 298
        "Reference temperature [K]":293.15,
        #ELECTRODE DATA
        "Initial concentration in negative electrode [mol.m-3]":c_init,
        "Initial concentration in positive electrode [mol.m-3]":c_init,
        "Negative electrode conductivity [S.m-1]": 22.8,
        "Positive electrode conductivity [S.m-1]": 22.8,
        "Positive electrode exchange-current density [A.m-2]":j0_nmo_rate,
        "Negative electrode exchange-current density [A.m-2]":j0_nmo_rate,

        "Maximum concentration in negative electrode [mol.m-3]": 3.6e4,
        "Maximum concentration in positive electrode [mol.m-3]": 3.6e4,
        "Negative electrode porosity": 0.4,
        "Positive electrode porosity": 0.4,
        "Negative electrode thickness [m]": 5e-05,
        "Positive electrode thickness [m]": 5e-05,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode OCP entropic change [V.K-1]": 0,
        "Positive electrode OCP entropic change [V.K-1]": 0,
        "Negative electrode OCP [V]":func,
        "Positive electrode OCP [V]":func,
        "Negative electrode thickness [m]": 5e-05,
        "Positive electrode thickness [m]": 5e-05,
        "Negative electrode active material volume fraction":0.6, 
        "Positive electrode active material volume fraction":0.6,
        "Electrode height [m]":1,
        "Electrode width [m]":1.5e-04,#only the product is computed...so mathcing it to galens data
        #"Electrode cross-sectional area [m2]": 1.5e-04,
       #ELECTROLYTE DATA
        "Electrolyte conductivity [S.m-1]":0.88,
        "Electrolyte diffusivity [m2.s-1]":2e-10,#2e-10
        "Initial concentration in electrolyte [mol.m-3]":1000,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.875,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.875,
        #SEPARATOR
        "Separator porosity": 0.5,
        "Separator thickness [m]": 2.6e-04,
        "Separator Bruggeman coefficient (electrolyte)":1.5,
        #MISC
        "Thermodynamic factor":1,
        "Cation transference number": 0.4,
        #"Current function [A]":0.03,
         "Nominal cell capacity [A.h]": 2.6e-03,
        "Number of cells connected in series to make a battery":1,
        "Number of electrodes connected in parallel to make a cell":1,
        "Lower voltage cut-off [V]":2.15 ,
        "Upper voltage cut-off [V]":3.5,
        #GUESSES
        "Negative electrode double-layer capacity [F.m-2]":0.02,
        "Positive electrode double-layer capacity [F.m-2]":0.02,
        "Contact resistance [Ohm]": 35,
        #ADDED PARAMETER: RATE CONSTANT
        "NMO reaction rate constant [m2.5.mol-0.5.s-1]": 5e-10
        }
parameter_values = pybamm.ParameterValues(dictionary)
#frequencies = np.logspace(-2,5,200)
#%%
for x in [0.528, x_mid, 1.0]:
    U = parameter_values.evaluate(
        pybamm.FunctionParameter(
            "Positive electrode OCP [V]",
            {"stoichiometry": pybamm.Scalar(x)},
        )
    )
    print(U)

dx = 1e-4
dUdx = (parameter_values.evaluate(pybamm.FunctionParameter("Positive electrode OCP [V]",
            {"stoichiometry": pybamm.Scalar(x_mid + dx)}))
      - parameter_values.evaluate(pybamm.FunctionParameter("Positive electrode OCP [V]",
            {"stoichiometry": pybamm.Scalar(x_mid - dx)}))) / (2 * dx)
print("dU/dx at operating point:", dUdx)

#%%--setting up synthetic data---
model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential","contact resistance": "true"})
simulator_syn = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(data_frequency))
solution= simulator_syn.solve()
Z =solution["Impedance"].data
Z_real = np.real(Z)
Z_imaginary = np.imag(Z)
absZ=  np.abs(Z)
fig,ax = plt.subplots(1,2, tight_layout = True)
ax[0].plot(Z_real,-Z_imaginary, 'o-',color= 'g')
ax[0].set_xlabel(r"$Z_r(\omega_k) [\text{m}\Omega]$")
ax[0].set_ylabel(r"$Z_j(\omega_k) [\text{m}\Omega]$")
ax[1].plot(np.asarray(data_frequency),absZ,)
ax[1].set_yscale('log')
ax[1].set_xlabel("frequency [Hz]")
ax[1].set_ylabel(r"$|Z(\omega_k)| [\text{m}\Omega]$")
ax[1].set_xscale('log')
#ax[0].plot(data_Z_re,data_Z_im)
#ax[1].set_xlim([1e-4,])

#%% Running Pybop 
#%% contents: building synethic data to test on parameter values
Impedance_data =   np.asarray(data_Z_re,dtype=float ) - 1j * np.asarray(data_Z_im, dtype= float)
#synthetic_Impedance =np.asarray(Z_real,dtype=float ) - 1j * np.asarray(Z_imaginary, dtype= float)
print("Z'' column at semicircle:", data_Z_im[np.argmax(-np.real(1j*Impedance_data))])
print("Im(Impedance_data) should be NEGATIVE over the arc:",
      np.median(np.imag(Impedance_data)))
dataset = pybop.Dataset({"Frequency [Hz]": np.asarray(data_frequency),
                         "Impedance": Impedance_data},domain = "Frequency [Hz]")
#synthetic_dataset = pybop.Dataset({"Frequency [Hz]": np.asarray(frequencies),"Impedance": Z}, domain = "Frequency [Hz]")
lo = data_frequency < 1.0
slope = np.polyfit(np.log10(data_frequency[lo]), np.log10(np.abs(Impedance_data[lo])), 1)[0]
print("low-f slope of |Z|:", slope)
# section contents: parameter updating for optimisation. For electrode specific parameter values, 
# the positive electrode parameter is updated with a pybop parameter,
#and the corresponding negative electrode assigned a pybamm.Parameter("positive counterpart"). Ensures that at each forwad
# run, the cell's parameter values are symmetric. 

parameter_values.update({
    # short times(frequencies 1 Hz < f< 1kHz)
    "Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(0.5,0.25,truncated_at=[1e-4,2]),initial_value= 0.5),
   "Initial concentration in positive electrode [mol.m-3]":pybop.Parameter(pybop.Gaussian(c_init,1e3, truncated_at= [c_min,c_max]),initial_value= c_init),
    "Electrolyte diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-10,5e-11,truncated_at = [1e-15,1e-4]),
                                                       transformation = pybop.LogTransformation(),
                                                    initial_value = 2e-10),
   "Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(1,0.25,truncated_at = [0.01,10]),
                                                      transformation = pybop.LogTransformation(),
                                                        initial_value = 0.88),
    "Positive particle diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-10,5e-11,truncated_at = [1e-16,1e-8]),
                                                      transformation = pybop.LogTransformation(),
                                                        initial_value = 1e-10),
    "Contact resistance [Ohm]":pybop.Parameter(pybop.Gaussian(40,10,truncated_at = [5,80]),
                                                        initial_value = 40),
    "Cation transference number":pybop.Parameter(pybop.Gaussian(0.5,0.3,truncated_at = [1e-15,0.7]),
                                                        initial_value = 0.4),
    #"Electrode height [m]":pybop.Parameter(pybop.Gaussian(0.003,0.0065,truncated_at = [1e-18,1e-1]),transformation = pybop.LogTransformation(),
                                                        #initial_value = 0.0007),
    #"Electrode width [m]":pybop.Parameter(pybop.Gaussian(0.1,0.05,truncated_at = [1e-6,1.5]),transformation = pybop.LogTransformation(),
                                                        #initial_value = 0.1),
    #"Thermodynamic factor":pybop.Parameter(pybop.Gaussian(0.5,0.3,truncated_at = [1e-15,0.999999]),
                                                        #initial_value = 0.4),
    })
#need to make sure it does not check to see if parameter exists as it is new
parameter_values.update(
    {
        "NMO reaction rate constant [m2.5.mol-0.5.s-1]": pybop.Parameter(
            pybop.Gaussian(5e-11, 9e-12, truncated_at=[1e-13, 1e-5]),
            transformation=pybop.LogTransformation(),
            initial_value=5e-11),
    },
    check_already_exists=False,   # needed the first time, since it's a new key
)

#setting the negative and positive electrode parameters to be the same.
parameter_values["Negative electrode double-layer capacity [F.m-2]"]= pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]")
parameter_values["Initial concentration in negative electrode [mol.m-3]"]=pybamm.Parameter("Initial concentration in positive electrode [mol.m-3]")
parameter_values["Negative particle diffusivity [m2.s-1]"]=pybamm.Parameter("Positive particle diffusivity [m2.s-1]")
#contents: building pybamm model that includes contact resistnace(R0) and surface form differential(introduces double layer capacitance dependence)
model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})
simulator = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(data_frequency))
#print([p for p in dir(pybop) if "cost" in p.lower() or "error" in p.lower() or "impedance" in p.lower() or "eis" in p.lower()])
#%% contents:
#  Build cost function, weighting over the frequences, check if this overrides the domain based weighting included in pybop.
cost= pybop.SumSquaredError(dataset=dataset,target="Impedance",weighting = "domain")
cost.weighting = cost.weighting / np.abs(Impedance_data) **2 
problem = pybop.Problem(simulator,cost)
problem.set_target("Impedance")
pybop.plot.nyquist(problem)
#%%
print(problem.parameters.names)
x_true = [1e-8, 2e-10,10,10]
print("cost at truth", problem(x_true))
#%% contents: runs optimiser, finds result. 
optim = pybop.CMAES(problem)
optim.set_max_iterations(100)
optim.set_max_unchanged_iterations(40)
result = optim.run()

#%% #implementing a second optimiser
problem.parameters.update(initial_values=result.x) 
second_optim = pybop.NelderMead(problem)
second_optim.set_max_iterations(50)
second_optim.set_max_unchanged_iterations(20)
final_result = second_optim.run()
print(final_result.best_inputs)
#%%

pybop.plot.nyquist(problem=problem, inputs = final_result.best_inputs)
#%%
print(result)
print(result.best_inputs)
pybop.plot.nyquist(problem=problem, inputs = result.best_inputs)
print(problem.parameters.names)
x0 = problem.parameters.get_initial_values()
print("cost at optimised parameter:", problem(result.best_inputs))

print("cost at initial guess:", problem(x0))

#%% contents: running optimised values in a new simulation 
result.best_inputs
model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})
parameter_values.update(final_result.best_inputs)
simulator = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(data_frequency))
sol = simulator.solve()
imp = sol["Impedance"].data
plt.plot(np.real(imp),-np.imag(imp))
plt.plot(data_Z_re,data_Z_im)
#pybop.plot.problem(problem, inputs=result.best_inputs, title = "Optimised Comparison")

neg_val = parameter_values.evaluate(
    pybamm.Parameter("Negative electrode double-layer capacity [F.m-2]")
)


pos_val = parameter_values.evaluate(
    pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]")
)
print(f"Negative: {neg_val}")
print(f"Positive: {pos_val}")
print(f"Equal: {neg_val == pos_val}")

#%%
x0 = problem.parameters.get_initial_values()
sol = problem.simulate(problem.parameters.to_inputs_list(x0)[0]) if hasattr(problem, "simulate") else None
# simpler: rebuild once
Zm = simulator.solve_batch(inputs=[dict(zip(problem.parameters.names, x0))])[0]["Impedance"].data

fig, ax = plt.subplots(1, 2, figsize=(11, 4), tight_layout=True)
ax[0].loglog(data_frequency, np.abs(Impedance_data), 'o-', label='data')
ax[0].loglog(data_frequency, np.abs(Zm), 's-', label='model')
ax[0].set_xlabel('f [Hz]'); ax[0].set_ylabel('|Z| [Ω]'); ax[0].legend()

ax[1].semilogx(data_frequency, -np.imag(Impedance_data), 'o-', label='data')
ax[1].semilogx(data_frequency, -np.imag(Zm), 's-', label='model')
ax[1].set_xlabel('f [Hz]'); ax[1].set_ylabel('-Im(Z) [Ω]'); ax[1].legend()
f_apex_data  = data_frequency[np.argmax(data_Z_im)]          # your Z'' is already -Im
f_apex_model = data_frequency[np.argmax(-np.imag(Zm))]
print(f"data apex: {f_apex_data:.3g} Hz, model apex: {f_apex_model:.3g} Hz, "
      f"ratio: {f_apex_model/f_apex_data:.3g}")
#%% SENSITIVITY ANALYSIS

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
 