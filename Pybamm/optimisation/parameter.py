
#%%
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pybop
#%%

model = pybamm.lithium_ion.DFN(options = {"surface form":"differential"})
parameter_values = pybamm.ParameterValues("Chayambuka2022")
parameter_values["Negative electrode double-layer capacity [F.m-2]"]= 0.05
parameter_values["Positive electrode double-layer capacity [F.m-2]"]= 0.05
frequencies = np.logspace(-4,4,200)
print(parameter_values)
#%%
simulator_syn = pybop.pybamm.EISSimulator(model=model,parameter_values= parameter_values, f_eval= np.asarray(frequencies))
solution= simulator_syn.solve()


Z =solution["Impedance"].data


#%%
chayam_positive = parameter_values[
    "Positive particle diffusivity [m2.s-1]"
]
chayam_negative = parameter_values[
    "Negative particle diffusivity [m2.s-1]"
]

sto = np.linspace(0.001, 0.999, 500)
T = 298.15

D_positive = np.array([
    parameter_values.evaluate(
        chayam_positive(pybamm.Scalar(s), pybamm.Scalar(T))
    )
    for s in sto
], dtype=float)

D_negative = np.array([
    parameter_values.evaluate(
        chayam_negative(pybamm.Scalar(s), pybamm.Scalar(T))
    )
    for s in sto
], dtype=float)

plt.figure(figsize=(7, 5))
plt.semilogy(sto, D_positive[:,0,0], label="Positive electrode — NVPF")
plt.semilogy(sto, D_negative[:,0,0], label="Negative electrode — hard carbon")
plt.xlabel("Particle stoichiometry")
plt.ylabel(r"Particle diffusivity [m$^2$ s$^{-1}$]")
plt.title("Chayambuka2022 particle diffusivities at 298.15 K")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("chayambukadiffusivity.pdf",bbox_inches = "tight")
plt.show()



dictionary_SPM = {
    "Upper voltage cut-off [V]": 4.2,
    "Negative electrode thickness [m]": 6.4e-05,
    "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Negative electrode Bruggeman coefficient (electrode)": 0,
    "Reference temperature [K]": 298.15,
    "Electrode height [m]": 0.000254,
    "Initial temperature [K]": 298.15,
    "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Positive electrode Bruggeman coefficient (electrode)": 0,
    "Separator thickness [m]": 2.5e-05,
    "Maximum concentration in negative electrode [mol.m-3]": 14540,
    "Initial concentration in electrolyte [mol.m-3]": 1000,
    "Electrode width [m]": 1,
    "Number of cells connected in series to make a battery": 1.0,
    "Number of electrodes connected in parallel to make a cell": 1.0,
    "Maximum concentration in positive electrode [mol.m-3]": 15320,
    "Separator Bruggeman coefficient (electrolyte)": 1.5,
    "Lower voltage cut-off [V]": 2.0,
    "Nominal cell capacity [A.h]": 0.003,
    "Positive electrode thickness [m]": 6.8e-05,

    "Initial concentration in negative electrode [mol.m-3]": 13520,
    "Negative particle radius [m]": 3.48e-06,
    "Negative electrode OCP [V]":None,
    "Negative electrode active material volume fraction": 0.489,
    "Negative electrode OCP entropic change [V.K-1]": 0,
    "Positive electrode exchange-current density [A.m-2]":
        None,

    # Not supplied by the Chayambuka parameter set
    "Negative electrode double-layer capacity [F.m-2]": None,

    "Ambient temperature [K]": 298.15,
    "Negative electrode exchange-current density [A.m-2]":
       j0_nmo,
    "Positive electrode OCP entropic change [V.K-1]": 0,
    "Positive particle diffusivity [m2.s-1]":
       None,
    "Negative particle diffusivity [m2.s-1]":
        None,
    "Current function [A]": 0.003,
    "Negative electrode porosity": 0.51,
    "Initial concentration in positive electrode [mol.m-3]": 3320,
    "Positive electrode porosity": 0.23,
    "Positive electrode OCP [V]": None,
    "Separator porosity": 0.55,

    # Not supplied by the Chayambuka parameter set
    "Positive electrode double-layer capacity [F.m-2]": None,

    "Positive particle radius [m]": 5.9e-07,
    "Positive electrode active material volume fraction": 0.55,
}

model = pybamm.lithium_ion.SPM(options = {"surface form":"differential"})
parameter_values = pybamm.ParameterValues("Chayambuka2022")
parameter_values.update({
    "Initial concentration in negative electrode [mol.m-3]":
        parameter_values[
            "Initial concentration in positive electrode [mol.m-3]"
        ],

    "Maximum concentration in negative electrode [mol.m-3]":
        parameter_values[
            "Maximum concentration in positive electrode [mol.m-3]"
        ],

    "Negative electrode Bruggeman coefficient (electrode)":
        parameter_values[
            "Positive electrode Bruggeman coefficient (electrode)"
        ],

    "Negative electrode Bruggeman coefficient (electrolyte)":
        parameter_values[
            "Positive electrode Bruggeman coefficient (electrolyte)"
        ],

    "Negative electrode OCP [V]":
        parameter_values["Positive electrode OCP [V]"],

    "Negative electrode OCP entropic change [V.K-1]":
        parameter_values[
            "Positive electrode OCP entropic change [V.K-1]"
        ],

    "Negative electrode active material volume fraction":
        parameter_values[
            "Positive electrode active material volume fraction"
        ],

    "Negative electrode charge transfer coefficient":
        parameter_values[
            "Positive electrode charge transfer coefficient"
        ],

    "Negative electrode conductivity [S.m-1]":
        parameter_values[
            "Positive electrode conductivity [S.m-1]"
        ],

    "Negative electrode exchange-current density [A.m-2]":
        parameter_values[
            "Positive electrode exchange-current density [A.m-2]"
        ],

    "Negative electrode porosity":
        parameter_values["Positive electrode porosity"],

    "Negative electrode thickness [m]":
        parameter_values["Positive electrode thickness [m]"],

    "Negative particle diffusivity [m2.s-1]":
        parameter_values["Positive particle diffusivity [m2.s-1]"],

    "Negative particle radius [m]":
        parameter_values["Positive particle radius [m]"],
})
parameter_values["Negative electrode double-layer capacity [F.m-2]"]=0.005
parameter_values["Positive electrode double-layer capacity [F.m-2]"]=0.05
#parameter_values["Positive particle diffusivity [m2.s-1]"]=0.02
#parameter_values["Negative particle diffusivity [m2.s-1]"]=parameter_values["Positive particle diffusivity [m2.s-1]"]


model = pybamm.lithium_ion.DFN(options = {"surface form":"differential"})
print(model.options)
eis_sim = pybamm.EISSimulation(model,parameter_values=parameter_values)
frequencies = np.logspace(-4,4,200)
result = eis_sim.solve(frequencies)
#print(list(result.data.keys())) 
Z_re = result["Z_re [Ohm]"]
Z_im = result["Z_im [Ohm]"]
freq= result["Frequency [Hz]"]
print(len(Z_im))
#plt.plot(Z_re,-Z_im)