#%%
# ---preamble
import numpy as np
import pybamm
import scipy as sc
import math
import matplotlib.pyplot as plt
from myfuncs import cubic,x_clean,V_clean,Y,y
from scipy.interpolate import CubicSpline
#%%
# file contents: 
#- using OCV from data_polishing in pybamm with SPMe
#-Getting a sense of the objective functoin
#%%
#--using pybamm cubic interpolant so it can be easily translated into an expression tree--#
xc=np.asarray(x_clean,dtype= float).ravel()
xc_mapped = 0.5 * xc + 0.5 
Vc=np.asarray(V_clean,dtype= float).ravel()
t= np.linspace(0,1,100)
order = np.argsort(x_clean)
xc= x_clean[order]
Vc = V_clean[order]
#plt.plot(xc_mapped[::-1],Vc)
def func(sto):
    return pybamm.Interpolant(xc,Vc, sto, interpolator="cubic")
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")
hero = parameter_values["Positive electrode OCP [V]"] 
#%%
model = pybamm.lithium_ion.SPMe()
parameter_values= pybamm.ParameterValues("Chayambuka2022")
model.print_parameter_info()
print(parameter_values)
#hero = parameter_values["Negative electrode OCP [V]"]
sto = np.linspace(0,1,100)
def positive_electrode_func(sto):
    return func(sto) + hero(sto)
#parameter_values["Positive electrode OCP [V]"] = positive_electrode_func
parameter_values["Lower voltage cut-off [V]"]= 1.8
parameter_values["Upper voltage cut-off [V]"]= 4.2
v = np.array([float(hero(pybamm.Scalar(s)).evaluate()) for s in sto])
#%%
simulation = pybamm.Simulation(model,parameter_values=parameter_values)
#%%
t_evals = np.linspace(0,10,100)
output_variables = ["Terminal voltage [V]","Current [A]", "Voltage [V]","Electrolyte concentration [mol.m-3]" ]
#%%
solution = simulation.solve([0,3600])
print(solution["Time [s]"].entries[[0, -1]])   # start and end time
print(solution.termination) 
solution = simulation.solve([0, 3600])
print("t_end =", solution["Time [s]"].entries[-1])
I = solution["Current [A]"].entries
print("current range:", I.min(), I.max())  
#%%
pybamm.dynamic_plot(solutions=solution,output_variables=output_variables)
type(solution["Terminal voltage [V]"])