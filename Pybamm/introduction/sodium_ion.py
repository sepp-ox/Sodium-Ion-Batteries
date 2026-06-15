
#%%
#---preamble---#
import numpy as np
import scipy as sc
import pybamm
import pybop
import matplotlib as plt
import os

#%% 
#---sodium ion application--#
model = pybamm.sodium_ion.BasicDFN()
model.print_parameter_info()
params=  model.default_parameter_values
#print(params)
model2 = pybamm.lithium_ion.DFN()
model2.print_parameter_info()
params2 = model2.default_parameter_values
print("SOID BATTERY",params)
print("LITHIUM BATTERY",params2)
#parameter_values["Positive particle radius"] = "[input]"
#%% 
#---restructure for sodium ion batteries using lithium DFN as i can add volume expansion options--#
options = {""}
model = pybamm.lithium_ion.DFN()
parameter_values = model.default_parameter_values
print(parameter_values)

# %%
#---Ch2022 paramters--#
parameter_values = pybamm.ParameterValues("Chayambuka2022")
print(parameter_values)

#%% 
#---Sodium-ion specifics---#
#incorporate (volume expansion in composite), 

options= {"particle mechanics":"swelling only"}
model = pybamm.lithium_ion.DFN(options=options)
parameter_values = pybamm.ParameterValues("Chayambuka2022")
sim = pybamm.Simulation(model, parameter_values=parameter_values)
#sim.solve([0,3600])
#output_variables = ["Voltage [V]", "Electrolyte concentration [mol.m-3]"]

#%% 
#---parameter difference: "particle mechanics": "swelling only" from base lithium_ion model
base = {p.name for p in pybamm.lithium_ion.DFN().parameters}
with_mech = {p.name for p in pybamm.lithium_ion.DFN(
    options={"particle mechanics": "swelling only"}
).parameters}

print("Added by swelling option:")
for p in sorted(with_mech - base):
    print(p)
#%%
#---SODIUM ION OPTIMISATION PROBLEM
#---gathering synethic data---#
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")
t_eval = np.arange(0,900,2)
solution = pybamm.Simulation(model).solve(t_eval=t_eval)
current = solution["Current [A]"](t_eval)
voltage = solution["Voltage [V]"](t_eval)

#---guassian noise---#
sigma = 0.01 #1mV
dataset = pybop.import_pybamm_solution({

    "Time [s]": t_eval,
    "Current [A]": current,
    "Voltage [V]": pybop.add_noise(voltage,sigma)
})
#---IDENTIFYING OPTIMISATION PARMETERS---adjusting some parameters with initial guesses---##
parameter_values.update({
"Negative electrode active material volume fraction":pybop.Parameter(pybop.Gaussian(0.6,0.02,truncated_at=[0.5,0.8])),
"Positive electrode active material volume fraction":pybop.Parameter(pybop.Gaussian(0.48,0.02,truncated_at=[0.4,0.7])),
})

#---Generating Simulator, optimisation procedure----#

simulator = pybop.pybamm.Simulator(model, parameter_values=parameter_values, protocol=dataset)
#---constructing cost function---#
cost= pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator,cost)
#---solving the optimisation problem---#
optim= pybop.AdamW(problem)
optim.set_max_unchanged_iterations(40)
optim.set_max_iterations(150)

#--adjustments to optimiser---#
optim.optimiser.b1 = 0.75
optim.optimiser.b2 = 0.75
#%%
#---Running optimisation, quick plot---#
result=  optim.run()
pybop.plot.problem(problem,inputs=result.best_inputs,title ="Optimised Comparison")
# %%
