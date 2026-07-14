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

def func(sto):
     return pybamm.Interpolant(xc_mapped,Vc[::-1], sto, interpolator="cubic")
F= 96485.33212
def  j0_nmo(ce, cs_surf, cs_max, T):
    return  F * 3e-11 * np.sqrt(ce) * np.sqrt(cs_surf) * np.sqrt(cs_max - cs_surf)

def j0_nmo_rate(c_e, cs_surf, cs_max, T):
    k = pybamm.Parameter("NMO reaction rate constant [m2.5.mol-0.5.s-1]")
    return F * k * pybamm.sqrt(c_e) * pybamm.sqrt(cs_surf) *pybamm.sqrt(cs_max-cs_surf)
#%%
#print(os.listdir(r"C:\Users\sepps\OneDrive\Oxford\diss\Pybamm\data"))
data_EIS  =pd.read_csv(r"C:\Users\sepps\OneDrive\Oxford\diss\Code\Pybamm\data\NMO_symmetrical.csv")
#print(data_EIS.head())
#print(data_EIS.columns.to_list())
data_frequency  = data_EIS["frequency (Hz)"]
data_Z_re = data_EIS["Z'"]
data_Z_im = data_EIS["Z''"]
idx = np.argsort(data_frequency)
data_frequency = np.asarray(data_frequency)[idx]
data_Z_re      = np.asarray(data_Z_re)[idx]
data_Z_im      = np.asarray(data_Z_im)[idx]
print(np.shape(data_frequency))
#print(data_frequency.max(), data_frequency.min())
#print(data_frequency)
plt.plot(data_Z_re,data_Z_im, "o-")
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
        "Positive electrode active material volume fraction":0.6,#open to changes
        "Electrode height [m]":1,
        "Electrode width [m]":1.5e-04,#only the product is computed...so mathcing it to galens data
        #"Electrode cross-sectional area [m2]": 1.5e-04,
       #ELECTROLYTE DATA
        "Electrolyte conductivity [S.m-1]":0.05,#sdecrease 0.88
        "Electrolyte diffusivity [m2.s-1]":2e-10,#2e-10 #decrease 
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
        "Negative electrode double-layer capacity [F.m-2]":0.5,
        "Positive electrode double-layer capacity [F.m-2]":0.5,
        "Contact resistance [Ohm]": 40,
        #ADDED PARAMETER: RATE CONSTANT
        "NMO reaction rate constant [m2.5.mol-0.5.s-1]": 5e-11}
parameter_values = pybamm.ParameterValues(dictionary)
#frequencies = np.logspace(-2,5,200)

#%%--setting up synthetic data---
frequencies = np.logspace(1e-4,1e-1,141)
model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential","contact resistance": "true"})
simulator_syn = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(frequencies))
solution= simulator_syn.solve()
Z =solution["Impedance"].data
Z_real = np.real(Z)
Z_imaginary = np.imag(Z)
absZ=  np.abs(Z)
fig,ax = plt.subplots(1,2, tight_layout = True)
ax[0].plot(Z_real,-Z_imaginary, 'o-',color= 'g')
ax[0].set_xlabel(r"$Z_r(\omega_k) [\text{m}\Omega]$")
ax[0].set_ylabel(r"$Z_j(\omega_k) [\text{m}\Omega]$")
ax[1].plot(np.asarray(frequencies),absZ,)
ax[1].set_yscale('log')
ax[1].set_xlabel("frequency [Hz]")
ax[1].set_ylabel(r"$|Z(\omega_k)| [\text{m}\Omega]$")
ax[1].set_xscale('log')
#ax[0].plot(data_Z_re,data_Z_im)
#ax[1].set_xlim([1e-4,])


#%% Running Pybop
# Constructing the optimisation into sections
# Section 1: high frequency semicircle 
f_low,f_high = 1e3,1e5
freq = np.logspace(3,5,41)
mask = (data_frequency >= f_low)
data_frequency=  data_frequency[mask]
data_Z_re = data_Z_re[mask]
data_Z_im = data_Z_im[mask]
plt.plot(data_Z_re,data_Z_im)
print(np.shape(data_frequency))
print(np.sum(np.abs(freq[i]-data_frequency[i])for  i in range(len(freq))))
# & (data_frequency <= f_high)
#Scaling Real data

#%% contents: building synethic data to test on parameter values

Impedance_data =   np.asarray(data_Z_re,dtype=float ) - 1j * np.asarray(data_Z_im, dtype= float)
dataset = pybop.Dataset({"Frequency [Hz]": np.asarray(data_frequency),
                         "Impedance": Impedance_data},domain = "Frequency [Hz]")

#Series resistance and semicircle
parameter_values.update({
    "Contact resistance [Ohm]":pybop.Parameter(pybop.Gaussian(40,10,truncated_at = [5,80]),
                                                        initial_value = 40),
   "Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(0.1,0.05,truncated_at = [0.001,10]), initial_value= 0.05),
                                                     
    "Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(0.5,0.25,truncated_at=[1e-4,2]),initial_value= 0.5),
     })

parameter_values["Negative electrode double-layer capacity [F.m-2]"]= pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]")
parameter_values.update(
    {
        "NMO reaction rate constant [m2.5.mol-0.5.s-1]": pybop.Parameter(
            pybop.Gaussian(5e-11, 9e-12, truncated_at=[1e-13, 1e-5]),
            transformation=pybop.LogTransformation(),
            initial_value=5e-11),
    },
    check_already_exists=False,   # needed the first time, since it's a new key
)

# section contents: parameter updating for optimisation. For electrode specific parameter values, 
# the positive electrode parameter is updated with a pybop parameter,
#and the corresponding negative electrode assigned a pybamm.Parameter("positive counterpart"). Ensures that at each forwad
# run, the cell's parameter values are symmetric. 
#%%
"""
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
"""
#contents: building pybamm model that includes contact resistnace(R0) and surface form differential(introduces double layer capacitance dependence)
model_pybop = pybamm.lithium_ion.SPM(options = {"surface form":"differential", "contact resistance" : "true"})
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
optim.set_max_iterations(200)
optim.set_max_unchanged_iterations(60)
result = optim.run()
print(result)

#%% #implementing a second optimiser
problem.parameters.update(initial_values=result.x) 
second_optim = pybop.NelderMead(problem)
second_optim.set_max_iterations(50)
second_optim.set_max_unchanged_iterations(40)
final_result = second_optim.run()
print(final_result.best_inputs)
print(final_result)
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

def section_based_optimisation(frequencies = None, f_bounds = [], parameter_values= None, optim_params= None, Z_real= None, Z_imag = None):
    #runs section wise CMAES,NelderMed 
    #pass into pybopparameters wither there relevant distributions

    f = np.asarray(frequencies)
    if f_bounds[0] >= f_bounds[1]:
        return ValueError("Lower bound must be strictly less than upper")
    f_low,f_high = f_bounds[0],f_bounds[1]
    mask = (frequencies >= f_low) & (frequencies <= f_high)
    Z_real =Z_real[mask]
    Z_imag = Z_imag[mask]
    frequencies = frequencies[mask]
    params = pybamm.ParameterValues(parameter_values)
    params.update(optim_params,check_already_exists=False)
    Impedance_data =   np.asarray(Z_real,dtype=float ) - 1j * np.asarray(Z_imag, dtype= float)
    dataset = pybop.Dataset({"Frequency [Hz]": np.asarray(frequencies),
                         "Impedance": Impedance_data},domain = "Frequency [Hz]")
    model_pybop = pybamm.lithium_ion.SPMe(options = {"surface form":"differential", "contact resistance" : "true"})  
    simulator = pybop.pybamm.EISSimulator(model_pybop,parameter_values= params, f_eval= np.asarray(frequencies))
    cost= pybop.SumSquaredError(dataset=dataset,target="Impedance",weighting = "domain")
    cost.weighting = cost.weighting / np.abs(Impedance_data) **2 
    problem = pybop.Problem(simulator,cost);problem.set_target("Impedance")
    optim = pybop.CMAES(problem);optim.set_max_iterations(200);optim.set_max_unchanged_iterations(60)
    result = optim.run()
    problem.parameters.update(initial_values=result.x) 
    second_optim = pybop.NelderMead(problem);second_optim.set_max_iterations(50);second_optim.set_max_unchanged_iterations(20)
    final_result = second_optim.run()
    print(final_result.best_inputs)
    print(final_result)
    pybop.plot.nyquist(problem=problem,inputs=final_result.best_inputs)
    return [final_result,problem]

#%%

# & (data_frequency <= f_high)
#Scaling Real data
# #f_low,f_high = 1e3,1e5
#freq = np.logspace(3,5,41)
#mask = (data_frequency >= f_low)
#data_frequency=  data_frequency[mask]
#data_Z_re = data_Z_re[mask]
#data_Z_im = data_Z_im[mask]


#%%
parameter_values= pybamm.ParameterValues(dictionary)
data_frequency  = data_EIS["frequency (Hz)"]
data_Z_re = data_EIS["Z'"]
data_Z_im = data_EIS["Z''"]
idx = np.argsort(data_frequency)
data_frequency = np.asarray(data_frequency)[idx]
data_Z_re      = np.asarray(data_Z_re)[idx]
data_Z_im      = np.asarray(data_Z_im)[idx]
trim =(data_frequency <= 5e4)
data_frequency,data_Z_im,data_Z_re = data_frequency[trim], data_Z_im[trim],data_Z_re[trim]
Impedance_data_galen =   np.asarray(data_Z_re,dtype=float ) - 1j * np.asarray(data_Z_im, dtype= float)
def frequency_plot(fbounds ,data_frequency,data_Z_re,data_Z_im):
    mask = (data_frequency >= fbounds[0]) & (data_frequency <= fbounds[1])
    data_frequency=  data_frequency[mask]
    data_Z_re = data_Z_re[mask]
    data_Z_im = data_Z_im[mask]
    plt.plot(data_Z_re,data_Z_im)
    return None

#getting frequency bounds
freq_bounds_semicircle = [5e3,5e4]
freq_bounds_warburg=[9e-1,1e2]
freq_bounds_diffusion_tail= [data_frequency.min(), 0.5e-1]
fbounds = freq_bounds_semicircle
frequency_plot(fbounds=fbounds, data_frequency=data_frequency,data_Z_re=data_Z_re ,data_Z_im=data_Z_im)
#%% SEMICIRCLE: HIGH FREQUENCY
optim_params= {
    "Contact resistance [Ohm]":pybop.Parameter(pybop.Gaussian(40,10,truncated_at = [5,80]),initial_value = 40),
   "Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(0.1,0.05,truncated_at = [0.001,10]), initial_value= 0.05),
    "Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(0.5,0.25,truncated_at=[1e-4,2]),initial_value= 0.5),
       "NMO reaction rate constant [m2.5.mol-0.5.s-1]": pybop.Parameter(
            pybop.Gaussian(5e-11, 9e-12, truncated_at=[1e-13, 1e-5]),
            transformation=pybop.LogTransformation(),
            initial_value=5e-11),
    "Negative electrode double-layer capacity [F.m-2]":pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]"),
    }
semi_circle = section_based_optimisation(frequencies=data_frequency, f_bounds=freq_bounds_semicircle,
                                         parameter_values=parameter_values,optim_params=optim_params,Z_real = data_Z_re,Z_imag= data_Z_im)

#%%

#%% updating parameter values with the best values from the previous optimisation
print(semi_circle[0])
parameter_values.update(semi_circle[0].best_inputs)
parameter_values["Negative electrode double-layer capacity [F.m-2]"]=parameter_values["Positive electrode double-layer capacity [F.m-2]"]
print(parameter_values["Negative electrode double-layer capacity [F.m-2]"])
sim_check = pybop.pybamm.EISSimulator(
    pybamm.lithium_ion.SPMe(options={"surface form": "differential", "contact resistance": "true"}),
    parameter_values=parameter_values, f_eval=data_frequency)
Zm = sim_check.solve()["Impedance"].data

hf = data_frequency > 5e3    # everything above the next window
rel_err = np.abs(Zm[hf] - Impedance_data_galen[hf]) / np.abs(Impedance_data_galen[hf])
print(f"HF relative misfit: mean {rel_err.mean():.2%}, max {rel_err.max():.2%}")
#%% Electrolyte Center:
optim_params_warburg = {"Electrolyte diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-10,5e-11,truncated_at = [1e-15,1e-4]),
                                                       transformation = pybop.LogTransformation(),
                                                    initial_value = 2e-10),
   "Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(1,0.25,truncated_at = [0.01,10]),
                                                      transformation = pybop.LogTransformation(),
                                                        initial_value = 0.88),
}
warburg = section_based_optimisation(frequencies=data_frequency, f_bounds=freq_bounds_warburg,
                                         parameter_values=parameter_values,optim_params=optim_params_warburg,Z_real = data_Z_re,Z_imag= data_Z_im)


#%%
parameter_values.update(warburg[0].best_inputs)

#DIFFUSION TAIL
optim_params_diffusion_tail = {
    
"Positive particle diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-10,5e-11,truncated_at = [1e-16,1e-8]),
                                                      transformation = pybop.LogTransformation(),
                                                        initial_value = 1e-10),
    "Negative particle diffusivity [m2.s-1]" : pybamm.Parameter("Positive particle diffusivity [m2.s-1]"),
    "Positive particle radius [m]":pybop.Parameter(pybop.Gaussian(3e-6,6.5e-7,truncated_at = [1e-8,1e-4]),
                                                        initial_value = 3e-6),
    "Negative particle radius [m]":pybamm.Parameter("Positive particle radius [m]"),
    "Initial concentration in positive electrode [mol.m-3]":pybop.Parameter(pybop.Gaussian(c_init,1e3, truncated_at= [c_min,c_max]),initial_value= c_init),
     "Initial concentration in negative electrode [mol.m-3]":pybamm.Parameter("Initial concentration in positive electrode [mol.m-3]"),
}
freq_bounds_diffusion_tail = [data_frequency.min(),1] #Hz
diffusion_tail = section_based_optimisation(frequencies=data_frequency,f_bounds=freq_bounds_diffusion_tail,parameter_values=parameter_values,optim_params=optim_params_diffusion_tail, Z_real=data_Z_re,Z_imag=data_Z_im)

#%% plotting section-wise results
#updating parameter values
parameter_values.update(diffusion_tail[0].best_inputs)
parameter_values["Initial concentration in negative electrode [mol.m-3]"]=parameter_values["Initial concentration in positive electrode [mol.m-3]"]
parameter_values["Negative particle radius [m]"]=parameter_values["Positive particle radius [m]"]
parameter_values["Negative particle diffusivity [m2.s-1]"] =parameter_values["Positive particle diffusivity [m2.s-1]"]
print(parameter_values)


#%%
#%% total optimisaiton










"""
parameter_values.update({
    # short times(frequencies 1 Hz < f< 1kHz)
    Initial concentration in positive electrode [mol.m-3]":pybop.Parameter(pybop.Gaussian(c_init,1e3, truncated_at= [c_min,c_max]),initial_value= c_init),
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
"""
# %%
