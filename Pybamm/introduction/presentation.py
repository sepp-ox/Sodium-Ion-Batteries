#%%
#---preamble---#
import numpy as np
import scipy
import pybamm
import pybop
import matplotlib as plt

#%%
#---quick model---#
experiment = pybamm.Experiment([("Discharge at 0.2C for 1 minutes","Rest for 0.5 minutes","Charge at 0.2C for 1 minutes") *2 
                                ])
model = pybamm.lithium_ion.DFN()
parameter_values= pybamm.ParameterValues("Chayambuka2022")
sim = pybamm.Simulation(model, parameter_values=parameter_values,experiment=experiment)
t_eval = [0,3600]
sim.solve()
output_variables= ["Current [A]","Negative electrode potential [V]"]
fig1 = sim.plot(output_variables=output_variables, show_plot= False)
fig1.fig.savefig("Voltage_electrolyteconcentration.png", dpi=300, bbox_inches= "tight")
plot_v,ax = sim.plot_voltage_components(show_plot=False)
plot_v.savefig("voltage_components.png",dpi=300, bbox_inches = "tight")
print(parameter_values)
#%%

import pybamm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# parameter ranges to sweep
D_s_values = np.linspace(1e-1, 0.9, 10)   # charge transfer at negative electrode
k0_values  = np.linspace(1e-1, 0.9, 10)   # reaction rate constant

# store a scalar cost (e.g. terminal voltage at end of discharge)
landscape = np.zeros((len(D_s_values), len(k0_values)))

model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chayambuka2022")

for i, C_s in enumerate(D_s_values):
    for j, k0 in enumerate(k0_values):
            pv = parameter_values.copy()
            pv["Negative electrode charge transfer coefficent "] = C_s
            pv["Negative electrode porosity"] = k0
            experiment = pybamm.Experiment(["Discharge at 0.1C until 1.0V"])
            sim = pybamm.Simulation(model, parameter_values=pv)
            sim.solve([0,1400])
            # scalar output — terminal voltage at end
            V = sim.solution["Terminal voltage [V]"].entries[-1]
            landscape[i, j] = V

# --- Plot as heatmap ---


# Heatmap
#%%
landscape
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im = axes[0].contourf(
    np.log10(k0_values), np.log10(D_s_values), landscape,
    levels=20, cmap="viridis"
)
plt.colorbar(im, ax=axes[0], label="Maximum voltage [V]")
axes[0].set_xlabel("log$_{10}$(k$_0$)")
axes[0].set_ylabel("log$_{10}$(D$_s$)")
axes[0].set_title("Optimisation Landscape (heatmap)")

# 3D surface

fig, ax3d = plt.subplots(1,1,figsize = (5,5))

fig.add_subplot(122, projection="3d")
K, D = np.meshgrid(np.log10(k0_values), np.log10(D_s_values))
ax3d.plot_surface(K, D, landscape, cmap="viridis", alpha=0.85)
ax3d.set_xlabel("log$_{10}$(k$_0$)")
ax3d.set_ylabel("log$_{10}$(D$_s$)")
ax3d.set_zlabel("Terminal voltage [V]")
ax3d.set_title("Optimisation Landscape (3D)")
plt.tight_layout()
plt.savefig("optimisation_landscape.png", dpi=300, bbox_inches="tight")
plt.show()


