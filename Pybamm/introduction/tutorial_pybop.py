#%%
# ---preamble---#
import numpy as np
import pybamm
import pybop
import scipy as sc
import matplotlib as plt
import os
#%%------implementing pybop-----
pybop.plot.PlotlyManager().pio.renderers.default = "notebook_connected"
np.random.seed(8)

#%%
# ----setting up model----
model = pybamm.lithium_ion.SPM()
parameter_values= pybamm.ParameterValues("Chen2020")
t_eval = np.linspace(0,10,100)
solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(t_eval = t_eval, t_interp=t_eval)
dataset= pybop.import_pybamm_solution(solution)
#---updating parameters----
parameter_values.update({
"Negative electrode active material volume fraction ": pybop.Parameter(initial_value=0.6),
"Positive electrode active material volume fraction": pybop.Parameter(initial_value=0.6),
})
#---using dataset, solution and parameter values(defined above)---#
output_variables = ["Voltage [V]"]
simulator = pybop.pybamm.Simulator(model,
            parameter_values=parameter_values, protocol=dataset, output_variables=output_variables)
inputs = simulator.parameters.to_dict([0.5,0.5])
solution =simulator.solve(inputs=inputs)
#---Defining cost---#
cost=pybop.SumOfPower(dataset)
cost.evaluate(solution,inputs=inputs)
#%% 
# --- Problem Object appraoch--
problem = pybop.Problem(simulator,cost)
evaluation = problem.evaluate(inputs)
evaluation.get_values()
#%% 
# --- creating custom loss function--with guassian noise --# 
def my_cost(inputs):
    y= simulator.solve(inputs)
    solution = pybop.Solution(inputs)
    solution.set_solution_variable("Voltage [V]", data= pybop.add_noise(y["Voltage [V]"].data, 0.003),)
    return cost.evaluate(solution,inputs)
my_cost(inputs)
# %%
