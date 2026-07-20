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
from scipy.optimize import least_squares
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


def section_based_optimisation(model = "DFN" ,frequencies = None, f_bounds = [], parameter_values= None, optim_params= None, Z_real= None, Z_imag = None):
    #runs section wise CMAES,NelderMed 
    #pass into pybopparameters wither there relevant distributions

    f = np.asarray(frequencies)
    if f_bounds[0] >= f_bounds[1]:
        raise ValueError("Lower bound must be strictly less than upper")
    f_low,f_high = f_bounds[0],f_bounds[1]
    mask = (frequencies >= f_low) & (frequencies <= f_high)
    Z_real =Z_real[mask]
    Z_imag = Z_imag[mask]
    frequencies = frequencies[mask]
    parameter_values.update(optim_params,check_already_exists=False)
    Impedance_data =   np.asarray(Z_real,dtype=float ) - 1j * np.asarray(Z_imag, dtype= float)
    dataset = pybop.Dataset({"Frequency [Hz]": np.asarray(frequencies),
                         "Impedance": Impedance_data},domain = "Frequency [Hz]")
    if model == "DFN":
        model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})  
    if model == "SPMe":
        model_pybop = pybamm.lithium_ion.SPMe(options = {"surface form":"differential", "contact resistance" : "true"})  
    elif model == "SPM":
        model_pybop = pybamm.lithium_ion.SPM(options = {"surface form":"differential", "contact resistance" : "true"})  
    simulator = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(frequencies))
    cost= pybop.SumSquaredError(dataset=dataset,target="Impedance",weighting = "domain")
    cost.weighting = cost.weighting / np.abs(Impedance_data) **2 
    problem = pybop.Problem(simulator,cost);problem.set_target("Impedance")
    pybop.plot.nyquist(problem=problem)
    optim = pybop.CMAES(problem);optim.set_max_iterations(150);optim.set_max_unchanged_iterations(75)
    result = optim.run()
    print(parameter_values["Negative electrode double-layer capacity [F.m-2]"])
    pybop.plot.nyquist(problem=problem,inputs=result.best_inputs)
    return result.best_inputs,result.best_cost

#%%


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
plt.plot(xc_mapped,Vc)
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
idx = (data_frequency - 0.5).abs().idxmin()
print(f"f = {data_frequency[idx]} Hz")
print(f"Z' = {data_Z_re[idx]}, Z'' = {data_Z_im[idx]}")
print(f"(Zr, Zi) = ({data_Z_re[idx]}, {data_Z_im[idx]})")

#idx = np.argsort(data_frequency)
#data_frequency = np.asarray(data_frequency)[idx]
#data_Z_re      = np.asarray(data_Z_re)[idx]
#data_Z_im      = np.asarray(data_Z_im)[idx]
#print(np.shape(data_frequency))

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
        "Negative particle diffusivity [m2.s-1]":1e-12,
        "Positive particle diffusivity [m2.s-1]":1e-12,
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
        "Negative electrode thickness [m]": 5e-05,#50 change to 100
        "Positive electrode thickness [m]": 5e-05,
        "Negative electrode active material volume fraction":0.6, 
        "Positive electrode active material volume fraction":0.6,#open to changes
        "Electrode height [m]":pybamm.sqrt(1.5e-04),
        "Electrode width [m]":pybamm.sqrt(1.5e-04),#only the product is computed...so mathcing it to galens data
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
#%%

#%%
from scipy.optimize import least_squares

target = np.array([150.0, 50.0])           # [Re(Z), -Im(Z)] at f_target — check your units (Ohm vs mOhm)!
f_target = np.array([0.5])
opts = {"surface form": "differential", "contact resistance": "true"}

def build_params(logDs):
    pv = pybamm.ParameterValues(dictionary)   # includes your OCVs and j0
    Ds = 10.0 ** logDs
    pv.update({
        "Positive particle diffusivity [m2.s-1]": Ds,
        "Negative particle diffusivity [m2.s-1]": Ds,      # symmetric
        "Contact resistance [Ohm]": -10,
        "Positive particle radius [m]":3e-6,
        "Negative particle radius [m]":3e-6,
        "Initial concentration in negative electrode [mol.m-3]":c_min,
        "Initial concentration in positive electrode [mol.m-3]":c_min,
        "Positive electrode active material volume fraction": 0.044,
        "Negative electrode active material volume fraction":0.044

    }, check_already_exists=False)
    return pv

def residual(x):
    logDs = float(np.asarray(x).ravel()[0])
    pv = build_params(logDs)
    model = pybamm.lithium_ion.SPMe(options=opts)
    Z = pybop.pybamm.EISSimulator(model, parameter_values=pv,
                                  f_eval=f_target).solve()["Impedance"].data[0]
    return np.array([np.real(Z), -np.imag(Z)]) - target

x0 = np.log10(1e-12)                                   # start from your current values
sol = least_squares(residual, x0, diff_step=[0.05])
print(sol.x, residual(sol.x) + target)        # fitted knobs and achieved Z
#%%
synthetic_params = build_params(*sol.x)        # <- the dict you asked for
for eps in [0.6, 0.4, 0.2, 0.1,0.05,0.02,0.004]:
    pv = build_params(-13)
    pv["Positive electrode active material volume fraction"] = eps
    pv["Negative electrode active material volume fraction"] = eps
    Z = pybop.pybamm.EISSimulator(pybamm.lithium_ion.SPMe(options=opts),
                                  parameter_values=pv, f_eval=f_target).solve()["Impedance"].data[0]
    print(f"eps={eps}: Re={Z.real:.1f}, -Im={-Z.imag:.1f}")
#%%

target = np.array([166.0, 52.0])           # [Re(Z), -Im(Z)] at f_target — check your units (Ohm vs mOhm)!
f_target = np.array([0.5])
opts = {"surface form": "differential", "contact resistance": "true"}
def build_params(logDs, eps, Rc):
    pv = pybamm.ParameterValues(dictionary)
    pv.update({
        "Positive particle diffusivity [m2.s-1]": 10.0**logDs,
        "Negative particle diffusivity [m2.s-1]": 10.0**logDs,
        "Positive particle radius [m]": 3e-6, "Negative particle radius [m]": 3e-6,
        "Initial concentration in negative electrode [mol.m-3]": c_min,
        "Initial concentration in positive electrode [mol.m-3]": c_min,
        "Positive electrode active material volume fraction": eps,
        "Negative electrode active material volume fraction": eps,
        "Contact resistance [Ohm]": Rc,
    }, check_already_exists=False)
    return pv

def residual(x):
    logDs, eps, Rc = np.asarray(x, float)
    Z = pybop.pybamm.EISSimulator(pybamm.lithium_ion.SPMe(options=opts),
                                  parameter_values=build_params(logDs, eps, Rc),
                                  f_eval=f_target).solve()["Impedance"].data[0]
    return np.array([Z.real, -Z.imag]) - target

sol = least_squares(residual, [-12.0, 0.044, -10.0],
                    diff_step=[0.1, 0.005, 1.0],
                    bounds=([-15, 1e-3, -500], [-9, 0.75, 500]))
#%%
print(sol.x[0], residual(sol.x) + target) 
D = 10**sol.x[0]
eps0 = sol.x[1]
Rc0 = sol.x[2]
#%%--setting up synthetic data sandbox---

freq_bounds_semicircle = [5e3,5e4]
freq_bounds_warburg=[1e0,1e1]
freq_bounds_diffusion_tail= [data_frequency.min(),5e-1]
def frequency_plot(fbounds ,data_frequency,data_Z_re,data_Z_im):
    mask = (data_frequency >= fbounds[0]) & (data_frequency <= fbounds[1])
    data_frequency=  data_frequency[mask]
    data_Z_re = data_Z_re[mask]
    data_Z_im = data_Z_im[mask]
    plt.plot(data_Z_re,data_Z_im)
    return None
fbounds = freq_bounds_diffusion_tail
frequency_plot(fbounds=fbounds, data_frequency=data_frequency,data_Z_re=data_Z_re ,data_Z_im=data_Z_im)
parameter_values.update({
        "Positive particle diffusivity [m2.s-1]":D,
        "Negative particle diffusivity [m2.s-1]":D,      # symmetric
        "Contact resistance [Ohm]": Rc0,
        "Positive particle radius [m]":3e-6,
        "Negative particle radius [m]":3e-6,
        "Initial concentration in negative electrode [mol.m-3]":c_min,
        "Initial concentration in positive electrode [mol.m-3]":c_min,
        "Positive electrode active material volume fraction": eps,
        "Negative electrode active material volume fraction":eps }, check_already_exists=False)
mask = (data_frequency >= freq_bounds_diffusion_tail[0]) &(data_frequency <= freq_bounds_diffusion_tail[1]) 
data_frequency=  data_frequency[mask]
data_Z_re = data_Z_re[mask]
data_Z_im = data_Z_im[mask]
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
ax[0].plot(data_Z_re,data_Z_im)
#ax[1].set_xlim([1e-4,])

#%%

optim_params_diffusion_tail = {
#"Positive particle diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-12,2e-10,truncated_at = [1e-16,1e-6]),
                                                     # transformation = pybop.LogTransformation(),
                                                       # ),
    #"Negative particle diffusivity [m2.s-1]" : pybamm.Parameter("Positive particle diffusivity [m2.s-1]"),
    #"Positive particle radius [m]":pybop.Parameter(pybop.Gaussian(3e-5,6.5e-3,truncated_at = [1e-8,1e-1]),
                                                        #initial_value = 3e-5),
   # "Negative particle radius [m]":pybamm.Parameter("Positive particle radius [m]"),
   # "Initial concentration in positive electrode [mol.m-3]":pybop.Parameter(pybop.Gaussian(c_max*.74,c_max/2, truncated_at= [c_min/2,c_max*2]),transformation = pybop.LogTransformation(),initial_value= c_max*.8),
    # "Initial concentration in negative electrode [mol.m-3]":pybamm.Parameter("Initial concentration in positive electrode [mol.m-3]"),
    # "Electrolyte diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-10,5e-11,truncated_at = [1e-15,1e-4]),
                                                       #transformation = pybop.LogTransformation(),
                                                    #initial_value = 2e-10),
   #"Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(1,0.25,truncated_at = [0.01,10]),
                                                      #transformation = pybop.LogTransformation(),
                                                       # initial_value = 0.88),
     #'Electrolyte conductivity [S.m-1]': pybop.Parameter(pybop.Gaussian(semi_circle['Electrolyte conductivity [S.m-1]'], 1e-4), initial_value = semi_circle['Electrolyte conductivity [S.m-1]']),
    #"Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(semi_circle["Positive electrode double-layer capacity [F.m-2]"],semi_circle["Positive electrode double-layer capacity [F.m-2]"]*0.1 ), initial_value = semi_circle["Positive electrode double-layer capacity [F.m-2]"]),
       #"NMO reaction rate constant [m2.5.mol-0.5.s-1]": pybop.Parameter(pybop.Gaussian(semi_circle["NMO reaction rate constant [m2.5.mol-0.5.s-1]"],semi_circle["NMO reaction rate constant [m2.5.mol-0.5.s-1]"]*0.1 ), initial_value = semi_circle["NMO reaction rate constant [m2.5.mol-0.5.s-1]"]),
    #"Negative electrode double-layer capacity [F.m-2]":pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]")
    
    "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
        pybop.Gaussian(D, 0.5*D, truncated_at=[D/2,2*D]),
        transformation=pybop.LogTransformation(),
        initial_value=D),
    "Negative particle diffusivity [m2.s-1]":
        pybamm.Parameter("Positive particle diffusivity [m2.s-1]"),
    "Positive electrode active material volume fraction": pybop.Parameter(
        pybop.Gaussian(eps0, 0.15*eps0, truncated_at=[0.5*eps0, 2*eps0]),
        initial_value=eps0),
    "Negative electrode active material volume fraction":
        pybamm.Parameter("Positive electrode active material volume fraction"),
    "Contact resistance [Ohm]": pybop.Parameter(
        pybop.Gaussian(Rc0, 5, truncated_at=[Rc0-20, Rc0+20]),
        initial_value=Rc0),
    }                                     
 #Hz
data_frequency  = data_EIS["frequency (Hz)"] 
data_Z_re = data_EIS["Z'"]
data_Z_im = data_EIS["Z''"]
idx = np.argsort(data_frequency)
data_frequency = np.asarray(data_frequency)[idx]
data_Z_re      = np.asarray(data_Z_re)[idx]
data_Z_im      = np.asarray(data_Z_im)[idx]
freq_bounds_all = [data_frequency.min(), 5e4]
freq_bounds_diffusion_tail= [data_frequency.min(),5e-1]
diffusion_tail = section_based_optimisation(model = "SPMe",frequencies=data_frequency,f_bounds=freq_bounds_diffusion_tail,parameter_values=parameter_values,optim_params=optim_params_diffusion_tail, Z_real=data_Z_re,Z_imag=data_Z_im)
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
    "Negative electrode double-layer capacity [F.m-2]": pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]"),
     })

#parameter_values["Negative electrode double-layer capacity [F.m-2]"]= pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]")
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
optim.set_max_iterations(40)
optim.set_max_unchanged_iterations(30)
result = optim.run()
print(result)
print(parameter_values["Negative electrode double-layer capacity [F.m-2]"])
print(result.best_inputs)

#%% #implementing a second optimiser
problem.parameters.update(initial_values=result.x) 
print(parameter_values)
print(parameter_values["Negative electrode double-layer capacity [F.m-2]"])
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


#%%
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

#getting frequency bounds
freq_bounds_semicircle = [5e2,5e4]
freq_bounds_warburg=[1e0,1e1]
freq_bounds_diffusion_tail= [data_frequency.min(), 5e3]
fbounds = freq_bounds_semicircle
frequency_plot(fbounds=fbounds, data_frequency=data_frequency,data_Z_re=data_Z_re ,data_Z_im=data_Z_im)
#%% SEMICIRCLE: HIGH FREQUENCY
optim_params= {
    "Contact resistance [Ohm]":pybop.Parameter(pybop.Gaussian(40,10,truncated_at = [5,80]),initial_value = 40),
   "Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(0.1,0.05,truncated_at = [0.001,10]), initial_value= 0.05),
    "Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(0.5,0.25,truncated_at=[1e-4,2]),initial_value= 0.5),
       "NMO reaction rate constant [m2.5.mol-0.5.s-1]": pybop.Parameter(
            pybop.Gaussian(5e-12,2e-12, truncated_at=[1e-15, 1e-10]),
            transformation=pybop.LogTransformation(),
            initial_value=5e-12),
   "Negative electrode double-layer capacity [F.m-2]":pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]"),
    }
semi_circle,semi_circle_cost = section_based_optimisation(model = "SPMe",frequencies=data_frequency, f_bounds=freq_bounds_semicircle,
                                         parameter_values=parameter_values,optim_params=optim_params,Z_real = data_Z_re,Z_imag= data_Z_im)

#%% CASE STUDY: COMPARING MODELS
freq_bounds_semicircle = [5e2,5e4]
fig, ax = plt.subplots(1,2, figsize = (12,5),constrained_layout= True
                        )
mask = (data_frequency >= freq_bounds_semicircle[0]) & (data_frequency <= freq_bounds_semicircle[1])
f_tail = data_frequency[mask]
Zre_tail = data_Z_re[mask]
Zim_tail = data_Z_im[mask]
Z_data = np.asarray(Zre_tail, dtype=float) - 1j * np.asarray(Zim_tail, dtype=float)
print(np.shape(Z_data))
model_classes = {"DFN": pybamm.lithium_ion.DFN,
                 "SPMe": pybamm.lithium_ion.SPMe,
                 "SPM": pybamm.lithium_ion.SPM}
#model_classes = {"SPM":pybamm.lithium_ion.SPM}
opts = {"surface form": "differential", "contact resistance": "true"}
ax[0].plot(Zre_tail,Zim_tail, "k.", label="data")
ax[1].plot(f_tail,np.abs(Z_data),"k.", label = "data")
impedances={}
results = {}
for model in ["DFN","SPMe","SPM"]:
    optim_params= {
    "Contact resistance [Ohm]":pybop.Parameter(pybop.Gaussian(40,10,truncated_at = [5,80]),initial_value = 40),
   "Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(0.1,0.05,truncated_at = [0.001,10]), initial_value= 0.05),
    "Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(0.5,0.25,truncated_at=[1e-4,2]),initial_value= 0.5),
       "NMO reaction rate constant [m2.5.mol-0.5.s-1]": pybop.Parameter(
            pybop.Gaussian(5e-12,2e-12, truncated_at=[1e-15, 1e-10]),
            transformation=pybop.LogTransformation(),
            initial_value=5e-12),
   "Negative electrode double-layer capacity [F.m-2]":pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]"),
    }
    parameter_values.update(optim_params,check_already_exists=False)
    result,cost = section_based_optimisation(
        model=model, frequencies=data_frequency,  # note: full data, not f_tail
        f_bounds=freq_bounds_semicircle,
        parameter_values=parameter_values, optim_params=optim_params,
        Z_real=data_Z_re, Z_imag=data_Z_im)
    parameter_values.update(result, check_already_exists=False)
    mask = (data_frequency >= freq_bounds_semicircle[0]) & (data_frequency <= freq_bounds_semicircle[1])
    f_tail = data_frequency[mask]
    model_pybop = model_classes[model](options=opts)
    imp = pybop.pybamm.EISSimulator(model_pybop, parameter_values=parameter_values,
                                    f_eval=np.asarray(f_tail)).solve()["Impedance"].data
    impedances[model]=imp
    results[model] = result
    rel_err = np.abs(imp - Z_data) / np.abs(Z_data)
    print(f"{model}: FE[%] = {rel_err.mean()*100:.3e}, max = {rel_err.max():.3e}")
    ax[0].plot(np.real(imp), -np.imag(imp), label=model)
    ax[1].plot(f_tail,np.abs(imp),label = model)
ax[0].set_xlabel(r"Re(Z($\omega$)) [Ohm]"); ax[0].set_ylabel(r"-Im(Z($\omega$)) [Ohm]")
ax[1].set_ylabel(r"|Z($\omega$)| [Ohm]")
ax[1].set_xscale("log"),ax[1].set_yscale("log")
ax[1].set_xlabel("Frequency [Hz]")
ax[1].legend()
ax[0].legend()
plt.savefig("semicircle_semicircle_again.pdf", bbox_inches = "tight")
#%%
print(results)
#%% updating parameter values with the best values from the previous optimisation
#running same thing but doing at different SOC values
def FE(imp,real_imp):
    return 100 * np.abs(imp - real_imp) / np.abs(real_imp)
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for model, imp in impedances.items():
    resid = imp - Z_data
    resid_real = np.real(imp)-np.real(Zre_tail)
    resid_im = np.imag(imp)- np.imag(Zim_tail)
    ax2[0].plot(f_tail, np.abs(resid) / np.abs(Z_data) * 100, label=model)   # relative, %
    ax2[1].plot(f_tail, np.abs(resid_real)/ np.abs(Zre_tail) * 100, label=f"{model} Re")
    ax2[1].plot(f_tail, np.abs(resid_im) / np.abs(Zim_tail)*100, "--", label=f"{model} Im")
ax2[0].set_xscale("log"); ax2[1].set_xscale("log")
ax2[0].set_ylabel(r"|$\Delta$Z| [%]"); ax2[1].set_ylabel(r"$\Delta$Z [%]")
ax2[1].set_yscale("log")
for a in ax2: a.set_xlabel("Frequency [Hz]"); a.legend()
plt.savefig("residual_semicircle_case_again.pdf", bbox_inches = "tight")
#%%

print(semi_circle)
print(semi_circle["Electrolyte conductivity [S.m-1]"])
#run1 = semi_circle[0].best_inputs#run 1 savedwith good inputs

parameter_values.update(semi_circle)
parameter_values["Negative electrode double-layer capacity [F.m-2]"]=parameter_values["Positive electrode double-layer capacity [F.m-2]"]
print(parameter_values["Negative electrode double-layer capacity [F.m-2]"])
sim_check = pybop.pybamm.EISSimulator(
    pybamm.lithium_ion.SPMe(options={"surface form": "differential", "contact resistance": "true"}),
    parameter_values=parameter_values, f_eval=data_frequency)
Zm = sim_check.solve()["Impedance"].data

plt.plot(np.real(Zm),-np.imag(Zm))
plt.plot(data_Z_re,data_Z_im)
plt.savefig("semicircle_optimisation_full_image.pdf", bbox_inches = "tight")
hf = data_frequency > 5e3    # everything above the next window
rel_err = np.abs(Zm[hf] - Impedance_data_galen[hf]) / np.abs(Impedance_data_galen[hf])
print(f"HF relative misfit: mean {rel_err.mean():.2%}, max {rel_err.max():.2%}")
#%%
#%%
#parameter_values.update(warburg[0].best_inputs)
#DIFFUSION TAIL
optim_params_diffusion_tail = {
"Positive particle diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-12,2e-10,truncated_at = [1e-16,1e-6]),
                                                      transformation = pybop.LogTransformation(),
                                                        ),
    "Negative particle diffusivity [m2.s-1]" : pybamm.Parameter("Positive particle diffusivity [m2.s-1]"),
    "Positive particle radius [m]":pybop.Parameter(pybop.Gaussian(3e-5,6.5e-3,truncated_at = [1e-8,1e-1]),
                                                        initial_value = 3e-5),
   # "Negative particle radius [m]":pybamm.Parameter("Positive particle radius [m]"),
    "Initial concentration in positive electrode [mol.m-3]":pybop.Parameter(pybop.Gaussian(c_max*.74,c_max/2, truncated_at= [c_min/2,c_max*2]),transformation = pybop.LogTransformation(),initial_value= c_max*.8),
     "Initial concentration in negative electrode [mol.m-3]":pybamm.Parameter("Initial concentration in positive electrode [mol.m-3]"),
     #"Electrolyte diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-10,5e-11,truncated_at = [1e-15,1e-4]),
                                                       #transformation = pybop.LogTransformation(),
                                                    #initial_value = 2e-10),
   #"Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(1,0.25,truncated_at = [0.01,10]),
                                                      #transformation = pybop.LogTransformation(),
                                                       # initial_value = 0.88),
     #'Electrolyte conductivity [S.m-1]': pybop.Parameter(pybop.Gaussian(semi_circle['Electrolyte conductivity [S.m-1]'], 1e-4), initial_value = semi_circle['Electrolyte conductivity [S.m-1]']),
    #"Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(semi_circle["Positive electrode double-layer capacity [F.m-2]"],semi_circle["Positive electrode double-layer capacity [F.m-2]"]*0.1 ), initial_value = semi_circle["Positive electrode double-layer capacity [F.m-2]"]),
       #"NMO reaction rate constant [m2.5.mol-0.5.s-1]": pybop.Parameter(pybop.Gaussian(semi_circle["NMO reaction rate constant [m2.5.mol-0.5.s-1]"],semi_circle["NMO reaction rate constant [m2.5.mol-0.5.s-1]"]*0.1 ), initial_value = semi_circle["NMO reaction rate constant [m2.5.mol-0.5.s-1]"]),
    #"Negative electrode double-layer capacity [F.m-2]":pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]")
    }                                         
 #Hz
data_frequency  = data_EIS["frequency (Hz)"] 
data_Z_re = data_EIS["Z'"]
data_Z_im = data_EIS["Z''"]
idx = np.argsort(data_frequency)
data_frequency = np.asarray(data_frequency)[idx]
data_Z_re      = np.asarray(data_Z_re)[idx]
data_Z_im      = np.asarray(data_Z_im)[idx]
freq_bounds_all = [data_frequency.min(), 5e4]
diffusion_tail = section_based_optimisation(model = "SPMe",frequencies=data_frequency,f_bounds=freq_bounds_all,parameter_values=parameter_values,optim_params=optim_params_diffusion_tail, Z_real=data_Z_re,Z_imag=data_Z_im)
#%%
print(diffusion_tail[0])
#%%

R_ct = 12.6   # from your diagnostic, or recompute
C_tot = parameter_values["Positive electrode double-layer capacity [F.m-2]"] * (3*0.6/3e-6) * 5e-5 * 1.5e-4
print("apex at start:", 1/(2*np.pi*R_ct*C_tot), "Hz  (want ~5.8e3)")
print("knee at start:", 1e-13/(2*np.pi*(3e-6)**2), "Hz  (want ≲1e-2)")
#%% plotting section-wise results
#updating parameter values
parameter_values.update(diffusion_tail[0].best_inputs)
parameter_values["Initial concentration in negative electrode [mol.m-3]"]=parameter_values["Initial concentration in positive electrode [mol.m-3]"]
parameter_values["Negative particle radius [m]"]=parameter_values["Positive particle radius [m]"]
parameter_values["Negative particle diffusivity [m2.s-1]"] =parameter_values["Positive particle diffusivity [m2.s-1]"]
print(parameter_values)

