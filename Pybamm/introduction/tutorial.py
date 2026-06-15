
#--preamble--#
import numpy as np
import pybamm 
import matplotlib.pyplot as plt
import os
import pybop
import scipy as sc


#%%
#--comparing parameters to sodium ion---#
model = pybamm.lithium_ion.DFN()
model.print_parameter_info()
#%% --tutorial 1--
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.solve([0,3600])
models= [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.SPMe(),pybamm.lithium_ion.DFN()]
sims= []
for model in models:
    sim = pybamm.Simulation(model)
    sim.solve([0,3600])
    sims.append(sim)
#%%
#pybamm.dynamic_plot(sims)
model.variables.search("electrolyte")
#plotting only certain things
output_variables= ["Electrolyte concentration [mol.m-3]","Voltage [V]"]
sim.plot(output_variables=output_variables)
sim.plot(
    [
        ["Electrode current density [A.m-2]","Electrolyte current density [A.m-2]"],"Voltage [V]",
    ]
)

sim.plot_voltage_components()
sim.plot_voltage_components(split_by_electrode = True)
pybamm.print_citations()
#%% 
#--tutorial 2--checking out changing parameters 
parameter_values=  pybamm.ParameterValues("Chen2020")
parameter_values =pybamm.ParameterValues("Chen2020")
parameter_values["Electrode width [m]"]=10
parameter_values.search("electrolyte")
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, parameter_values= parameter_values)
sim.solve([0,3600])
#plotting only one things
output_variables = ["Electrolyte concentration [mol.m-3]","Voltage [V]"]
help(sim.plot)
sim.plot(output_variables =output_variables)
model.print_parameter_info()
#Example of wanting a function as an input 
#to set information:
# --tutorial 3--say i want to change Current function to be a function_initially
#initial setup...
def curr(t):
    return  pybamm.sin(t)
parameter_values =pybamm.ParameterValues("Chen2020")
parameter_values["Current function [A]"]= curr
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model,parameter_values=parameter_values)
t_val = np.arange(0,121,1)
sim.solve(t_eval = t_val)
output_variables = ["Electrolyte concentration [mol.m-3]","Voltage [V]"]
sim.plot(output_variables=output_variables)
model.print_parameter_info()
#%%
#--tutorial 4-- putting in a parameter sweep. ise "[input]" then sweep through inputs in the solve format
parameter_values= pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.DFN()
parameter_values["Current function [A]"] ="[input]"
sim = pybamm.Simulation(model, parameter_values= parameter_values)
sols = []
for c in [0.1,0.2,0.3]:
    soln = sim.solve([0,3600],inputs={"Current function [A]": c})
    plt.plot(soln["Time [s]"].entries, soln["Voltage [V]"].entries, label = f"{c} A")
    sols.append(soln["Terminal voltage [V]"].entries)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.legend(loc= "upper right")
plt.show()

#%%
#build model
# get parameters
# simulate with parameters
# initialise sweep
# solve/plot in loop
# plot

#%%

#Defining new parameter set:
def cube(t):
    return t**3
parameter_values = pybamm.ParameterValues({"Current function [A]":cube})
#issue..not enough parameter values

#%%
#--tutorial five-- running experiments using experimental setups
#create experiemnts in lists
#run it in the simulation step of the process
#for instance
#plotting multiple things
experiment =pybamm.Experiment(["Charge at 0.5 C for 2 minutes","Rest for 2 minutes","Discharge at 0.5 C for 2 minutes"])
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Separator porosity"] = 0.2
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment,parameter_values = parameter_values)
output_variables = ["Electrolyte concentration [mol.m-3]","Voltage [V]"]
sim.solve()
sim.plot(output_variables=output_variables)
#if i wanted to plot mulitpled things
models  = [pybamm.lithium_ion.DFN(),pybamm.lithium_ion.SPM()]
sols = []
for model in models:
    sim = pybamm.Simulation(model,experiment = experiment,parameter_values= parameter_values)
    soln = sim.solve()
    plt.plot(soln["Time [s]"].entries, soln["Voltage [V]"].entries,label = f"{model}")
    sols.append(soln["Terminal voltage [V]"].entries)
plt.legend()
plt.show()
#%%
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values

#%%                            
#building a cyclic experiment
model  = pybamm.lithium_ion.DFN()
experiment = pybamm.Experiment(
[("Charge at 0.5 C for 2 minutes", 
  "Rest for 1 second", 
  "Discharge at 1 C for 2 minutes",
)] *3 + ["Rest for 2 minutes",]
)
parameter_values = pybamm.ParameterValues("Chen2020")
output_variables = ["Terminal voltage [V]","Electrolyte concentration [mol.m-3]"]
model.print_parameter_info()
sim = pybamm.Simulation(model, experiment= experiment,parameter_values= parameter_values)
sim.solve()
sim.solution.cycles[1].plot(output_variables = output_variables) 

#%%
#Doing a parameter sweep in the cases on two differnet models
models= [pybamm.lithium_ion.DFN(),pybamm.lithium_ion.SPM()]
parameter_values["Initial concentration in electrolyte [mol.m-3]"] = "[input]"
sol = []
for model in models:
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values = parameter_values)
    for c in [1000,2000,3000]:
        soln = sim.solve(inputs= {"Initial concentration in electrolyte [mol.m-3]":c}) 
        plt.plot(soln["Time [s]"].entries,soln["Voltage [V]"].entries, label = f"{c} [mol.m-3]")
        sol.append(soln["Terminal voltage [V]"].entries)
    plt.legend()
    plt.show()

# %%
pybamm.step.string("Discharge at 1A for 2 minutes or until 2.5V")
#initialising power
time = np.linspace(0,10,10)
sin_t = np.sin(time *np.pi)
drive = np.column_stack([time,sin_t])
experiment = pybamm.Experiment([pybamm.step.power(drive)])
sim = pybamm.Simulation(model, experiment= experiment)
sim.solve()
output_variables = ["Terminal voltage [V]", "Voltage [V]"]
sim.plot(output_variables = output_variables)


#more experiment stuff

#Tutorial 6: Updating simulation outputs for 
#setup simple 
model = pybamm.lithium_ion.SPMe()
sim = pybamm.Simulation(model)
sim.solve([0,3600])
solution = sim.solution 
V= solution["Voltage [V]"]
t =solution["Time [s]"]
V.entries
#t.entries
model.variables
conductance_symbol = model.variables["Current [A]"] / model.variables["Voltage [V]"]
conductance = solution.observe(conductance_symbol)
conductance
#plotting conductance
time = np.linspace(0,solution.t[-1],1000)
plt.plot(time, conductance(time))
plt.xlabel("time")
plt.ylabel("conductance")
sim.save("SPMe.pkl")
sim2 =pybamm.load("SPMe.pkl")
sim2.plot()
sol = sim.solution
sol.save("SPMe_sol.pkl")
sol.save_data("sol_data.pkl", ["Current [A]","Voltage [V]"])
os.remove("SPMe_sol.pkl")
os.remove("sol_data.pkl")
#extracting voltage 


#%% Tutorial Seven: Model Options
#Pass it through the model = pybamm.lithium_ion.SPM(options=options)
options = {"thermal":"lumped"}
model = pybamm.lithium_ion.SPM(options=options)
sim = pybamm.Simulation(model)
sim.solve([0,3600])
output_variables = ["Cell temperature [K]", "Total heating [W.m-3]", "Current [A]", "Voltage [V]"]
sim.plot(output_variables=output_variables)
pybamm.print_citations()    

#Review Model Options(wanting to add thermal components)
options= {"thermal": "lumped"}
model = pybamm.lithium_ion.DFN(options= options)
sim = pybamm.Simulation(model,parameter_values=model.default_parameter_values)
sim.solve([0,3600])
sim.plot(["Cell temperature [K]","Total heating [W.m-3]","Current [A]", "Voltage [V]"])

# %%
#Tutorial 8: Solver Options
#This is for when i want to change resoltuion or accuracy of a model
model = pybamm.lithium_ion.DFN()
params = model.default_parameter_values
params["Lower voltage cut-off [V]"] = 3.6

#now define the two solvers
solver_baseline = pybamm.IDAKLUSolver(rtol = 1e-4,atol =1e-6)
solver_high_accuracy = pybamm.IDAKLUSolver(rtol = 1e-8, atol = 1e-12)

#use the two solvers to simulate
sim_baseline = pybamm.Simulation(model, parameter_values=params, solver= solver_baseline)
sim_high_accuracy = pybamm.Simulation(model, parameter_values=params,solver= solver_high_accuracy)

#solve them
sim_baseline.solve([0,3600])
print(f"Baseline mode solve time: {sim_baseline.solution.solve_time}")
sim_high_accuracy.solve([0,3600])
print(f"High accuracy solve time: {sim_high_accuracy.solution.solve_time}")
pybamm.dynamic_plot([sim_baseline,sim_high_accuracy],labels = ["Baseline","High accuracy"])
pybamm.print_citations()

#changing solver
experiment = [("Charge for 2 minutes", "Rest for 1 minute", 'Discharge until 3.0 V')]*3 + ["Rest for 2 minutes"]
solver = pybamm.IDAKLUSolver(rtol = 1e-6,atol=1e-10)
sim = pybamm.Simulation(model,solver=solver, experiment=experiment)
# %%
#Tutorial number 9: Changing the mesh
##Doing a mesh test
n_pts = [4,8,16,32,64]

model = pybamm.lithium_ion.SPMe()   
parameter_values = pybamm.ParameterValues("Ecker2015")
solutions = []
for n in n_pts:
    var_pts = {
    "x_n":n,
    "x_s":n,
    "x_p":n,
    "r_p": n,
    "r_n":n,
    }
    sim = pybamm.Simulation(model, var_pts=var_pts,parameter_values=parameter_values)
    sim.solve([0,3600],)
    solutions.append(sim.solution)
print(len(solutions))
print(len(n_pts))
pybamm.dynamic_plot(solutions,["Voltage [V]"],labels = n_pts)


