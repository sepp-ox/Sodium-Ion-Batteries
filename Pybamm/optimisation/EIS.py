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
xc_mapped = 0.5 * xc + 0.5 
Vc=np.asarray(V_clean,dtype= float).ravel()
order = np.argsort(xc_mapped)
xc_mapped= xc[order]
Vc = Vc[order]

def func(sto):
     return pybamm.Interpolant(xc_mapped,Vc[::-1], sto, interpolator="cubic")

F= 98485.33212
def  j0_nmo(ce, cs_surf, cs_max, T):
    return  F * 3e-11 * np.sqrt(ce) * np.sqrt(cs_surf) * np.sqrt(cs_max - cs_surf)


#%%
print(os.listdir(r"C:\Users\sepps\OneDrive\Oxford\diss\Pybamm\data"))
data_EIS  =pd.read_csv(r"C:\Users\sepps\OneDrive\Oxford\diss\Code\Pybamm\data\NMO_symmetrical.csv")
print(data_EIS.head())
print(data_EIS.columns.to_list())
data_frequency  = data_EIS["frequency (Hz)"]
data_Z_re = data_EIS["Z'"]
data_Z_im = data_EIS["Z''"]

#%%
#---full cell due to symettric EIS--#
'''
Positive electrode active material volume fraction: 0.6
Positive electrode porosity: 0.4
Positive electrode thickness: 5e-05 m 
Positive particle radius: 3e-06 m
Mode of operation: Half cell
Ambient temperature: 298 K
Electrode cross-sectional area: 1.5e-04 m2
Initial concentration in the electrolyte: 1 mol/m3
Nominal cell capacity: 2.6e-03 Ah
Separator porosity: 0.5


Separator thickness: 2.6e-04 m
 Exchange-current density for negative metal electrode [A.m-2]: 1.0e8

- Negative electrode conductivity [S.m-1]: 1.0e8

- Separator porosity: 0.5

- Separator Bruggeman coefficient (electrolyte): 1.5

- Initial concentration in electrolyte [mol.m-3]: 1000

- Cation transference number: 0.4

- Thermodynamic factor: 1.0

- Electrolyte diffusivity [m2.s-1]: 2e-10

- Electrolyte conductivity [S.m-1]: 0.88

- Initial temperature [K]: 293.15

- Ambient temperature [K]: 293.15

- Reference temperature [K]: 293.15

- Positive particle diffusivity [m2.s-1]: 1e-14

- Positive electrode reaction rate constant [m2.5 mol-0.5 s-1]: 3e-11

- Positive electrode conductivity [S.m-1]: 22.8

- Positive electrode Bruggeman coefficient (electrode): 0.0

- Positive electrode Bruggeman coefficient (electrolyte): 1.875

- Maximum concentration in positive electrode [mol.m-3]: 3.6e4

- Initial concentration in positive electrode [mol.m-3]: 1.9e4

- For charge initialization, use the near-fully-lithiated NMO value:
  Initial concentration in positive electrode [mol.m-3] = 0.994761 * 3.6e4 = 35811.396

- NMO stoichiometry window:
  x at 2.3 V: 1.000000
  x at 3.6 V: 0.527778
  Delta x: 0.472222

- Positive electrode OCP entropic change [V.K-1]: 0.0

- Positive electrode exchange-current density [A.m-2]: j0_nmo

- j0_nmo function:
  j0_nmo(ce, cs_surf, cs_max, T) = F * 3e-11 * sqrt(ce) * sqrt(cs_surf) * sqrt(cs_max - cs_surf)

- Lower voltage cut-off [V]: 2.3

- Upper voltage cut-off [V]: 3.6

- Contact resistance [Ohm]: 12
"Initial concentration in negative electrode [mol.m-3]":35811.396,
        "Initial concentration in positive electrode [mol.m-3]":35811.396,
        "Negative electrode conductivity [S.m-1]": 22.8,
        "Positive electrode conductivity [S.m-1]": 22.8,
        "Positive electrode exchange-current density [A.m-2]":j0_nmo,
        "Negative electrode exchange-current density [A.m-2]":j0_nmo,
        "Maximum concentration in negative electrode [mol.m-3]": 3.6e4,
        "Maximum concentration in positive electrode [mol.m-3]": 3.6e4,

chayam_positive= parameter_values["Positive electrode OCP [V]"]
chayam_negative = parameter_values["Negative electrode OCP [V]"]
chayam_parameter = parameter_values["Positive electrode exchange-current density [A.m-2]"]
sto = np.linspace(0,1,100)
t=1000
m=10
v= np.array([float(chayam_positive(pybamm.Scalar(s)).evaluate()) for s in sto])
v2 = np.array([float(chayam_parameter(pybamm.Scalar(s),pybamm.Scalar(m),pybamm.Scalar(t),pybamm.Scalar(298.15)).evaluate()) for s in sto])
v1= np.array([float(chayam_negative(pybamm.Scalar(s)).evaluate()) for s in sto])
#plt.plot(sto,v,label = "positive")
#plt.plot(sto,v1,label = "negative")
plt.plot(sto,v2)
plt.legend()
'''
x_mid = (1+0.528)/2
c_init = x_mid * 3.6e4
dict = { 
       #PARTICLE
        "Negative particle radius [m]": 3e-06,
        "Positive particle radius [m]": 3e-06,
        "Negative particle diffusivity [m2.s-1]":1,
        "Positive particle diffusivity [m2.s-1]":1,
        #TEMPERATURE
        "Initial temperature [K]":298.15,
        "Ambient temperature [K]": 293.15,
        "Reference temperature [K]":293.15,
        #ELECTRODE DATA
        "Initial concentration in negative electrode [mol.m-3]":c_init,
        "Initial concentration in positive electrode [mol.m-3]":c_init,
        "Negative electrode conductivity [S.m-1]": 22.8,
        "Positive electrode conductivity [S.m-1]": 22.8,
        "Positive electrode exchange-current density [A.m-2]":j0_nmo,
        "Negative electrode exchange-current density [A.m-2]":j0_nmo,
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
        "Current function [A]":0.003,
         "Nominal cell capacity [A.h]": 2.6e-03,
        "Number of cells connected in series to make a battery":1,
        "Number of electrodes connected in parallel to make a cell":1,
        "Lower voltage cut-off [V]":2.3 ,
        "Upper voltage cut-off [V]":3.6,
        #GUESSES
        "Negative electrode double-layer capacity [F.m-2]":10,
        "Positive electrode double-layer capacity [F.m-2]":10,
        "Contact resistance [Ohm]": 12,
        #
        }
model = pybamm.lithium_ion.DFN(options = {"surface form":"differential"})
parameter_values= pybamm.ParameterValues(dict)
eis_sim = pybamm.EISSimulation(model,parameter_values=parameter_values)
frequencies = np.logspace(-2,5,141)

result = eis_sim.solve(frequencies)
Z_re = result["Z_re [Ohm]"]
Z_im = result["Z_im [Ohm]"]
print(Z_re,Z_im)
plt.plot(Z_re,-Z_im)


#%% Running Pybop 
#%%
dataset = pybop.Dataset({"Frequency [Hz]": np.asarray(data_frequency),"Z_re [Ohm]": np.asarray(data_Z_re),"Z_im [Ohm]":np.asarray(data_Z_im)},domain = "Frequency [Hz]")
'''parameter_values.update({
    "Negative electrode double-layer capacity [F.m-2]":pybop.Parameter(pybop.Gaussian(5,2.5,truncated_at = [0.1,20])),
    "Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(5,2.5,truncated_at=[0.1,20])),
    "Electrolyte diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(np.log(1e-12),1,truncated_at = [1e-15,1e-4]),transformation = pybop.LogTransformation())})'''
simulator = pybop.pybamm.EISSimulator(model,parameter_values= parameter_values, f_eval= data_frequency)
print(dir(simulator)) 
#print([p for p in dir(pybop) if "cost" in p.lower() or "error" in p.lower() or "impedance" in p.lower() or "eis" in p.lower()])
#setting up the cost function#
# build the problem with your EIS simulator, parameters, and dataset
# use the custom EIS cost instead of SumSquaredError
# gradient-free optimiser (EIS has no gradients)

#%%


#%%
cost= pybop.SumSquaredError(dataset,target=["Z_re [Ohm]", "Z_im [Ohm]"],weighting = "domain")
problem = pybop.Problem(simulator,cost)
optim = pybop.CMAES(problem)
optim.set_max_iterations(150)
result = optim.run()
'''High frequency (kHz range)

Ohmic resistance R₀ — electrolyte conductivity, separator, contact resistance
Electrolyte conductivity [S.m-1]
Separator thickness [m], Separator porosity

Mid frequency (Hz range — semicircle)

Charge transfer resistance — Butler-Volmer kinetics
Positive/Negative electrode exchange-current density [A.m-2]
Double layer capacitance
Positive/Negative electrode double-layer capacity [F.m-2]

Low frequency (mHz range — Warburg tail)

Solid-state diffusion
Positive/Negative particle diffusivity [m2.s-1]
Electrolyte diffusion
Electrolyte diffusivity [m2.s-1]'''               

#settting up optimisation procedure and cost functio

#thoughts;
#----Electrolyte diffusivity: very sensitive for overall shape
#doulbe layer capacity: sensitive for the semicircle
# Electrolyte conductivity
# 


# %%
