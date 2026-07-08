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
'''
#---THE COMMENTED OUT VALUES ARE TAKEN FROM CHAYAMBUKA2022--# OTHERWISE IT IS GALENS HALF CELL DATA APPLIED TO BOTH ELECTRODES---outdated
dict = {"Positive electrode active material volume fraction":0.6, 
        "Positive electrode porosity": 0.4,
        "Positive electrode thickness [m]": 5e-05,
        "Positive particle radius [m]": 3e-06,
        "Positive electrode OCP [V]":func,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode OCP entropic change [V.K-1]": 0,
        #"Positive electrode charge transfer coefficient ":0.5,
        #"Positive electrode exchange-current density [A.m-2]":1,
        "Positive electrode conductivity [S.m-1]": 50,
        "Positive particle diffusivity [m2.s-1]":1,
        #"Positive particle electrode diffusivity [m2.s-1]":func,
        "Maximum concentration in positive electrode [mol.m-3]": 1, 
        "Electrode cross-sectional area [m2]": 1.5e-04,
        "Initial concentration in electrolyte [mol.m-3]":1,
        "Nominal cell capacity [A.h]": 2.6e-03,
        "Separator porosity": 0.5,
        "Separator thickness [m]": 2.6e-04,
        "Ambient temperature [K]": 298,
       
        "Negative electrode active material volume fraction":0.6, 
        "Negative electrode porosity": 0.4,
        "Negative electrode thickness [m]": 5e-05,
        "Negative particle radius [m]": 3e-06,
        "Negative electrode OCP [V]":func,
        
        "Initial temperature [K]":298.15,
        
        "Initial concentration in negative electrode [mol.m-3]":13520,
        "Initial concentration in positive electrode [mol.m-3]":3320,

        
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode OCP entropic change [V.K-1]": 0,
        #"Initial concentration in negative electrode [mol.m-3]":0, 
        #"Negative electrode charge transfer coefficient ":0.5,
        #"Negative electrode exchange-current density [A.m-2]":func,
        "Negative electrode conductivity [S.m-1]": 50,
        "Negative particle diffusivity [m2.s-1]":1,
        #"Negative particle electrode diffusivity [m2.s-1]":func,
        "Maximum concentration in negative electrode [mol.m-3]": 1,
        
        "Electrode height [m]": 0.000254,
        "Electrode width [m]":0.001,
         
        "Reference temperature [K]": 298.15,
        "Number of cells connected in series to make a battery":1,
        "Number of electrodes connected in parallel to make a cell":1,
        "Separator Bruggeman coefficient (electrolyte)":1.5,
        "Lower voltage cut-off [V]":2 ,
        "Upper voltage cut-off [V]":4,
        "Thermodynamic factor":1,
        "Cation transference number": 0.5,
        "Electrolyte conductivity [S.m-1]":100,
        "Current function [A]":0.003,
         "Electrolyte diffusivity [m2.s-1]":1,
         "Positive electrode exchange-current density [A.m-2]":1,
         "Negative electrode exchange-current density [A.m-2]":1,
        }

experiment=pybamm.Experiment(["Charge at 0.03 A for 1 hour"])
model = pybamm.lithium_ion.DFN()
model.print_parameter_info()
#parameter_values = pybamm.ParameterValues("Chayambuka2022")
parameter_values = pybamm.ParameterValues(dict)
#parameter_values.update(dict)
#print(parameter_values)
sim = pybamm.Simulation(model,parameter_values=parameter_values)
sol = sim.solve([0,3600])
output_variables = ["Voltage [V]", "Electrolyte concentration [mol.m-3]"]
pybamm.dynamic_plot(sol,output_variables=output_variables)
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
#%%

esi_sim = pybamm.EISSimulation(model,parameter_values=parameter_values)

frequencies = np.logspace(-4,4,30)





# %%



model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chayambuka2022")
sim = pybamm.Simulation(model = model, parameter_values=parameter_values)
sol =sim.solve([0,3600])
pybamm.dynamic_plot(sol, output_variables=output_variables)


