#%%
#---preamble---
import numpy as np
import scipy as sc
import pybamm
import pybop
import matplotlib.pyplot as plt
import json
from pybamm.input.parameters.ecm.example_set import ocv_data
#%% 
#---setting model---#
pybop.plot.PlotlyManager().pio.renderers.default="notebook_connected"
np.random.seed(8)
model= pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")
sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
t_eval = np.arange(0,900,2)
solution = sim.solve(t_eval=t_eval)
current = solution["Current [A]"](t_eval)
voltage = solution["Voltage [V]"](t_eval)
#%%
sigma = 0.01
dataset=  pybop.Dataset({
    "Time [s]": t_eval,
    "Current [A]": current,
    "Voltage [V]": np.random.normal(sigma,voltage)})
#%% #---parameter values updates___
parameter_values.update({"Negative electrode active material volume fraction": pybop.Parameter(pybop.Gaussian(0.6,0.02,truncated_at=[0.5,0.8])),
                         "Positive electrode active material volume fraction": pybop.Parameter(pybop.Gaussian(0.48,0.02,truncated_at=[0.4,0.7]))})
#%% 
simulator = pybop.pybamm.Simulator(model,parameter_values=parameter_values, protocol=dataset)
cost= pybop.SumSquaredError(dataset=dataset)
problem = pybop.Problem(simulator,cost)
optim= pybop.AdamW(problem=problem)
optim.set_max_unchanged_iterations(40)
optim.set_max_iterations(150)
optim.optimiser.b1=.75
optim.optimiser.b2=0.75
#%%
#---running result---#
result = optim.run()
result.best_inputs
#%%
#--plotting results--#
pybop.plot.problem(problem=problem,inputs= result.best_inputs,title ="Optimised Comparison")
result.plot_convergence()
result.plot_parameters()
#%%
#--new section: interacting with optimiser interface---#
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 1})
with open(r"C:\Users\sepps\OneDrive\Oxford\diss\PyBOP\examples\parameters\initial_ecm_parameters.json") as file: 
    parameter_values = pybamm.ParameterValues(json.load(file))
parameter_values.update({
    "Open-circuit voltage [V]": model.default_parameter_values["Open-circuit voltage [V]"]
})
t_eval = np.arange(0,900,2)
solution =pybamm.Simulation(model, parameter_values=parameter_values).solve(t_eval=t_eval,t_interp =t_eval)
dataset = pybop.import_pybamm_solution(solution=solution, variables=["Time [s]", "Current [A]", "Voltage [V]"])
#%%
#---defining parameter to fit: here i am defining to fit Rho---#
parameter_values.update({"R0 [Ohm]": pybop.Parameter(pybop.Gaussian(0.0002,0.0001,truncated_at=[1e-4,1e-2]))})
#building cost:
simulator  = pybop.pybamm.Simulator(model,parameter_values=parameter_values, protocol=dataset)
cost  = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)
parameter_values.search("Open-circuit voltage [V]")
#load model_define simulated data__makesoltuion____dataset_____define cost with pybamm.Simulator(model, parameter=paramet..)
# %% 
#---interacting with optimisers---#
options = pybop.PintsOptions(max_iterations=50, max_unchanged_iterations=25, absolute_tolerance=1e-12)
optimiser_one = pybop.XNES(problem=problem, options=options)
optimiser_one.set_max_iterations(50)
result = optimiser_one.run()
print(result)
bounds = np.asarray([[1e-6,0.2], [0,0.2]])
result.plot_surface(bounds=bounds)
parameter_values.search("R0 [Ohm]")
#%% 
#---taking a closer look at OCV--#
name, (x,y) = ocv_data
x = np.array(x).flatten()
plt.plot(x,y)
plt.xlabel("SOC")
plt.ylabel("Voltage[V]")
