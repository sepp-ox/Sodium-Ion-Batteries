#%% --preamble---#

import numpy as np
import pybamm
import pybop
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from myfuncs import func
#%%
#----contents of file--
#--running full cell for sodium ion cell---#
#----implementing Na-ion paramter values--#
#%%
#---full cell due to symettric EIS--#

#---THE COMMENTED OUT VALUES ARE TAKEN FROM CHAYAMBUKA2022--# OTHERWISE IT IS GALENS HALF CELL DATA APPLIED TO BOTH ELECTRODES
dict = {"Positive electrode active material volume fraction":0.6, 
        "Positive electrode porosity": 0.4,
        "Positive electrode thickness [m]": 5e-05,
        "Positive particle radius [m]": 3e-06,
        "Positive electrode OCP [V]":func,
        #"Positive electrode Bruggeman coefficent (electrolyte)": 1.5,
        #"Positive electrode Bruggeman coefficent (electrode)": 0,
        #"Positive electrode OCP entropic change [V.K-1]": 0,
        #"Initial concentration in positive electrode [mol.m_3]":0, 
        #"Positive electrode charge transfer coefficient ":0.5,
        #"Positive electrode exchange-current density [A.m-2]":func,
        #"Positive electrode conductivity [S.m-1]": 50,
        #"Particle particle diffusivity [m2.s-1] ":func,
        #"Positive particle electrode diffusivity [m2.s-1]":func,
        
        # "Maximum concentration in positive electrode [mol.m-3]": 15320, 
        "Electrode cross-sectional area [m2]": 1.5e-04,
        "Initial concentration in the electrolyte [mol.m-3]":1,
        "Nominal cell capacity [Ah]": 2.6e-03,
        "Separator porosity": 0.5,
        "Separator thickness [m]": 2.6e-04,
        "Ambient temperature [K]": 298,
       
       
        "Negative electrode active material volume fraction":0.6, 
        "Negative electrode porosity": 0.4,
        "Negative electrode thickness [m]": 5e-05,
        "Negative particle radius [m]": 3e-06,
        "Negative electrode OCP [V]":func,
        
        #"Initial temperature [K]":298.15,
        
        
        #"Negative electrode Bruggeman coefficent (electrolyte)": 1.5,
        #"Negative electrode Bruggeman coefficent (electrode)": 0,
        #"Negative electrode OCP entropic change [V.K-1]": 0,
        #"Initial concentration in negative electrode [mol.m_3]":0, 
        #"Negative electrode charge transfer coefficient ":0.5,
        #"Negative electrode exchange-current density [A.m-2]":func,
        #"Negative electrode conductivity [S.m-1]": 50,
        #"Negative particle diffusivity [m2.s-1] ":func,
        #"Negative particle electrode diffusivity [m2.s-1]":func,
        #"Maximum concentration in negative electrode [mol.m-3]": 15320,
        #
        # 
        # "Reference temperature [K]": 298.15,
        #"Number of cells connected in series to make a battery":parameter
        #"Number of cells connected in parallel to make a battery":parameter
        #"Separator Bruggemant coefficent (electrolyte)":"
        #"Lower voltage cut-off [V]":paramter,
        #"Upper voltage cut-off [V]":paramter,
        #"Thermodynamic factor":function,
        #"Cation transference number": ,
        #"Electrolyte conductivity [S.m-1]"
        #"Current function [A]":function,
        # "Electrolyte Diffusivity":function,

        "Negative electrode double-layer capacity [F.m-2]":100,
        "Positive electrode double-layer capacity [F.m-2]":100,
        }
options = {"surface form":"differential"}
model = pybamm.lithium_ion.DFN(options=options )
parameter_values = pybamm.ParameterValues("Chayambuka2022")
parameter_values.update(dict)
print(parameter_values)
sim = pybamm.Simulation(model,parameter_values=parameter_values)
sol = sim.solve([0,3600])
output_variables = ["Voltage [V]", "Electrolyte concentration [mol.m-3]"]
pybamm.dynamic_plot(sol,output_variables=output_variables)

#%%

esi_sim = pybamm.EISSimulation(model,parameter_values=parameter_values)

frequencies = np.logspace(-4,4,30)





# %%
