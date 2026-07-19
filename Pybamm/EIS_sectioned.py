#%%
###----preamble----###
import numpy as np
import scipy as sc
import math
import matplotlib.pyplot as plt
import pybamm
import pybop
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy import stats
#%% 
'''
------
contents
Goals: One parameter set fitting an experimental data set
Sections of script:
1) function definitions used in file
2) importing data and cleaning
3) iniital parameter set (dictionary)
4)sending the time constants to diffusion out of the inverval
5) optimising the semicircle and creating a new updated parameter set with those values
6)
------
'''
#%%1) function definitions and defining constants

F= 96485.33212
def stoich(Q, data = None, Qmax = None, Qmin= None):
    if Qmax is None:
        Qmax = data["Q"].max()
    if Qmin is None:
        Qmin = data["Q"].min()
    return (Q - Qmin)/ (Qmax - Qmin)
def section_based_optimisation(model = "DFN" ,frequencies = None, f_bounds = [], parameter_values= None, optim_params= None, Z_real= None, Z_imag = None):
    #runs section wise CMAES,NelderMed 
    #pass into pybopparameters wither there relevant distributions

    f = np.asarray(frequencies)
    if f_bounds[0] >= f_bounds[1]:
        raise ValueError("Lower bound must be strictly less than upper")
    f_low,f_high = f_bounds[0],f_bounds[1]
    mask = (frequencies >= f_low) & (frequencies <= f_high)
    Z_real =Z_real[mask]
    Z_imag = Z_imag[mask]
    frequencies = frequencies[mask]
    parameter_values.update(optim_params,check_already_exists=False)
    Impedance_data =   np.asarray(Z_real,dtype=float ) - 1j * np.asarray(Z_imag, dtype= float)
    dataset = pybop.Dataset({"Frequency [Hz]": np.asarray(frequencies),
                         "Impedance": Impedance_data},domain = "Frequency [Hz]")
    if model == "GroupedSPMe":
        model_pybop  = pybop.models.lithium_ion.GroupedSPMe(options ={"surface form":"differential", "contact resistance" : "true"})
    elif model == "DFN":
        model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})  
    elif model == "SPMe":
        model_pybop = pybamm.lithium_ion.SPMe(options = {"surface form":"differential", "contact resistance" : "true"})  
    elif model == "SPM":
        model_pybop = pybamm.lithium_ion.SPM(options = {"surface form":"differential", "contact resistance" : "true"})  
    simulator = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(frequencies))
    cost= pybop.SumSquaredError(dataset=dataset,target="Impedance",weighting = "domain")
    cost.weighting = cost.weighting / np.abs(Impedance_data) **2 
    problem = pybop.Problem(simulator,cost);problem.set_target("Impedance")
    pybop.plot.nyquist(problem=problem)
    optim = pybop.CMAES(problem);optim.set_max_iterations(300);optim.set_max_unchanged_iterations(150)
    result = optim.run()
    pybop.plot.nyquist(problem=problem,inputs=result.best_inputs)
    return result.best_inputs,result.best_cost
def frequency_plot(fbounds ,data_frequency,data_Z_re,data_Z_im):
    mask = (data_frequency >= fbounds[0]) & (data_frequency <= fbounds[1])
    data_frequency=  data_frequency[mask]
    data_Z_re = data_Z_re[mask]
    data_Z_im = data_Z_im[mask]
    plt.plot(data_Z_re,data_Z_im)
    return None
def func(sto):
     return pybamm.Interpolant(xc_mapped,Vc[::-1], sto, interpolator="cubic")
def  j0_nmo(ce, cs_surf, cs_max, T):
    return  F * 3e-11 * np.sqrt(ce) * np.sqrt(cs_surf) * np.sqrt(cs_max - cs_surf)
def j0_nmo_rate(c_e, cs_surf, cs_max, T):
    '''
    runs a pybamm EIS forward model for input frequency data
    '''
    k = pybamm.Parameter("NMO reaction rate constant [m2.5.mol-0.5.s-1]")
    return F * k * pybamm.sqrt(c_e) * pybamm.sqrt(cs_surf) *pybamm.sqrt(cs_max-cs_surf)
def run(model = "DFN" ,data_frequency = None,parameter_values= None,data_Z_re= None,data_Z_im = None):
    if model == "GroupedSPMe":
        model_pybop  = pybop.models.lithium_ion.GroupedSPMe(options ={"surface form":"differential", "contact resistance" : "true"})
    elif model == "DFN":
        model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})  
    elif model == "SPMe":
        model_pybop = pybamm.lithium_ion.SPMe(options = {"surface form":"differential", "contact resistance" : "true"})  
    elif model == "SPM":
        model_pybop = pybamm.lithium_ion.SPM(options = {"surface form":"differential", "contact resistance" : "true"})  
    simulator_syn = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(data_frequency))
    solution= simulator_syn.solve()
    Z =solution["Impedance"].data
    Z_real = np.real(Z)
    Z_imag = np.imag(Z)
    absZ=  np.abs(Z)
    fig,ax = plt.subplots(1,2, tight_layout = True)
    ax[0].plot(Z_real,-Z_imag, 'o-',color= 'g')
    ax[0].set_xlabel(r"$Z_r(\omega_k) [\text{m}\Omega]$")
    ax[0].set_ylabel(r"$Z_j(\omega_k) [\text{m}\Omega]$")
    ax[1].plot(np.asarray(data_frequency),absZ,)
    ax[1].set_yscale('log')
    ax[1].set_xlabel("frequency [Hz]")
    ax[1].set_ylabel(r"$|Z(\omega_k)| [\text{m}\Omega]$")
    ax[1].set_xscale('log')
    if data_Z_im is not None and  data_Z_re is not None: 
        ax[0].plot(data_Z_re,data_Z_im)
    return Z,Z_real,Z_imag
def time_constants(parameter_values = None):
    pv = parameter_values
    R_p  = pv["Positive particle radius [m]"]
    D_s  = pv["Positive particle diffusivity [m2.s-1]"]
    D_e  = pv["Electrolyte diffusivity [m2.s-1]"]
    eps  = pv["Positive electrode porosity"]
    eps_sep = pv["Separator porosity"]
    b    = pv["Positive electrode Bruggeman coefficient (electrolyte)"]
    b_sep = pv["Separator Bruggeman coefficient (electrolyte)"]
    L    = (pv["Negative electrode thickness [m]"] + pv["Separator thickness [m]"]
            + pv["Positive electrode thickness [m]"])
    k    = pv["NMO reaction rate constant [m2.5.mol-0.5.s-1]"]
    ce0  = pv["Initial concentration in electrolyte [mol.m-3]"]
    tau_d   = R_p**2 / D_s
    tau_e   = eps_sep * L**2 / (eps**b * D_e)          # paper's convention
    tau_sep = L**2 / (eps_sep**(b_sep - 1) * D_e)
    tau_ct  = R_p / (k * np.sqrt(ce0)) 
                    # = F R / (m sqrt(ce0)), m = F k
    T=298.15
    tau_ct = F * R_p / (np.sqrt(ce0)* F * k * np.sqrt(ce0) * np.sqrt(c_init) * np.sqrt(c_max - c_init))
    for name, tau in [("particle diffusion", tau_d), ("electrolyte (electrode)", tau_e),
                    ("electrolyte (separator)", tau_sep), ("charge transfer", tau_ct)]:
        print(f"{name:26s} tau = {tau:10.4g} s   ->  f = {1/(2*np.pi*tau):10.4g} Hz")
    return tau_d,tau_e,tau_sep,tau_ct

#%%

'''
data: importing, cleaning, and plotting to check
'''

data_EIS  =pd.read_csv(r"C:\Users\sepps\OneDrive\Oxford\diss\Code\Pybamm\data\NMO_symmetrical.csv")

data_frequency  = data_EIS["frequency (Hz)"] 
data_Z_re = data_EIS["Z'"]
data_Z_im = data_EIS["Z''"]
idx = np.argsort(data_frequency)
data_frequency = np.asarray(data_frequency)[idx]
data_Z_re      = np.asarray(data_Z_re)[idx]
data_Z_im      = np.asarray(data_Z_im)[idx]

data = pd.read_csv(r"C:\Users\sepps\OneDrive\Oxford\diss\Pybamm\data\NMO_eq_charge.csv")
Q = data["Q"]
V= data["Ewe"]
x= stoich(Q, data=data)
mask = ~np.isnan(x) & ~np.isnan(V)
x_clean = x[mask].to_numpy();V_clean = V[mask].to_numpy()
xc=np.asarray(x_clean,dtype= float).ravel()
xc_mapped = 0.472 * xc + 0.528 #specific mapping for the problem i am running: change based on galens exact 
Vc=np.asarray(V_clean,dtype= float).ravel()
fig,ax = plt.subplots(1,2, figsize = (10,10), constrained_layout = True)
ax[0].plot(xc,Vc[::-1])
ax[1].set_ylabel(r"$Z_{\text{im}}$")
ax[1].set_xlabel(r"$Z_{\text{re}}$")
ax[1].plot(data_Z_re,data_Z_im)
ax[0].set_ylabel("OCV [V]")
ax[0].set_xlabel("SOC [.]")
#%%
'''
parameter dictionary
calculating time constants 

'''
x_mid = (1+0.528)/2
c_init = x_mid * 3.6e4
c_min  =0.528 * 3.6e4
c_max = 1 * 3.6e4
dictionary = { 
       #PARTICLE
        "Negative particle radius [m]": 3e-3,
        "Positive particle radius [m]": 3e-3,
        "Negative particle diffusivity [m2.s-1]":1e-10,
        "Positive particle diffusivity [m2.s-1]":1e-10,
        #TEMPERATURE
        "Initial temperature [K]":298.15,
        "Ambient temperature [K]": 298,#reference 293.15,change to 298
        "Reference temperature [K]":293.15,
        #ELECTRODE DATA
        "Initial concentration in negative electrode [mol.m-3]":c_max*0.95,
        "Initial concentration in positive electrode [mol.m-3]":c_max*0.95,
        "Negative electrode conductivity [S.m-1]": 22.8,
        "Positive electrode conductivity [S.m-1]": 22.8,
        "Positive electrode exchange-current density [A.m-2]":j0_nmo_rate,
        "Negative electrode exchange-current density [A.m-2]":j0_nmo_rate,

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
        "Negative electrode thickness [m]": 5e-5,#50 change to 100
        "Positive electrode thickness [m]": 5e-05,
        "Negative electrode active material volume fraction":0.6, 
        "Positive electrode active material volume fraction":0.6,#open to changes
        "Electrode height [m]":float(np.sqrt(1.5e-04)),
        "Electrode width [m]":float(np.sqrt(1.5e-04)),#only the product is computed...so matching to galens data
       #ELECTROLYTE DATA
        "Electrolyte conductivity [S.m-1]":0.370,#sdecrease 0.88
        "Electrolyte diffusivity [m2.s-1]":2e-10,#2e-10 #decrease 
        "Initial concentration in electrolyte [mol.m-3]":1000,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.875,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.875,
        #SEPARATOR
        "Separator porosity": 0.5,
        "Separator thickness [m]": 2.6e-04,
        "Separator Bruggeman coefficient (electrolyte)":1.5,
        #MISC
        "Thermodynamic factor":1,
        "Cation transference number": 0.4,
        #"Current function [A]":0.03,
         "Nominal cell capacity [A.h]": 2.6e-03,
        "Number of cells connected in series to make a battery":1,
        "Number of electrodes connected in parallel to make a cell":1,
        "Lower voltage cut-off [V]":2.15 ,
        "Upper voltage cut-off [V]":3.5,
        #GUESSES
        "Negative electrode double-layer capacity [F.m-2]":0.5,
        "Positive electrode double-layer capacity [F.m-2]":0.5,
        "Contact resistance [Ohm]": 39.46,
        #ADDED PARAMETER: RATE CONSTANT
        "NMO reaction rate constant [m2.5.mol-0.5.s-1]": 7.3e-12}
parameter_values = pybamm.ParameterValues(dictionary)
parameter_values.update({'Positive electrode double-layer capacity [F.m-2]': np.float64(0.0016260742139396185), 'Contact resistance [Ohm]': np.float64(37.08512597190219), 'NMO reaction rate constant [m2.5.mol-0.5.s-1]': np.float64(4.569684612602907e-11)}
)
parameter_values["Negative electrode double-layer capacity [F.m-2]"]= parameter_values["Positive electrode double-layer capacity [F.m-2]"]
#%% 
'''
diagnostics:
calculating time coefficents and calculating the corresponding frqeuencies they are active in 

'''
parameter_values["Negative particle diffusivity [m2.s-1]"]=1e-15
parameter_values["Positive particle diffusivity [m2.s-1]"]=parameter_values["Negative particle diffusivity [m2.s-1]"]
run(model="SPMe", data_frequency=data_frequency, parameter_values=parameter_values,data_Z_re=data_Z_re,data_Z_im=data_Z_im )
time_constants(parameter_values)
#Diagnostics

#%% 
'''optimising the semiciricle:
1)trimming frequency range
2) choosing parameters to optimise in this region
3) running optimiser
'''
freq_bounds_semicircle = [5e2,5e4]
freq_bounds_electrolyte=[5e-1,5e2]
freq_bounds_diffusion= [data_frequency.min(),5e-1]
freq_bounds= freq_bounds_electrolyte
frequency_plot(fbounds=freq_bounds,data_frequency=data_frequency,data_Z_im=data_Z_im,data_Z_re=data_Z_re)
#%%
optim_params= {
    "Contact resistance [Ohm]":pybop.Parameter(pybop.Gaussian(40,10,truncated_at = [5,80]),initial_value = 40),
   #"Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(0.1,0.05,truncated_at = [0.001,10]), initial_value= 0.05),
    "Positive electrode double-layer capacity [F.m-2]": pybop.Parameter(pybop.Gaussian(0.5,0.25,truncated_at=[1e-4,2]),initial_value= 0.5),
       "NMO reaction rate constant [m2.5.mol-0.5.s-1]": pybop.Parameter(
            pybop.Gaussian(5e-12,5e-12, truncated_at=[1e-15, 1e-10]),
            transformation=pybop.LogTransformation(),
            initial_value=5e-12),
   "Negative electrode double-layer capacity [F.m-2]":pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]"),
   #"Positive particle radius [m]":pybop.Parameter(pybop.Gaussian(3e-6,6.1e-6,truncated_at = [1e-8,1e-1]),
                                                        #initial_value = 3e-6),
    #"Negative particle radius [m]":pybamm.Parameter("Positive particle radius [m]"),
    }
semi_circle,semi_circle_cost = section_based_optimisation(model = "SPMe",frequencies=data_frequency, f_bounds=freq_bounds_semicircle,
                                         parameter_values=parameter_values,optim_params=optim_params,Z_real = data_Z_re,Z_imag= data_Z_im)

#%% 
'''
new dictionary with updated parameters from semicircle optimisation
'''

parameter_values.update(semi_circle)
print(semi_circle)
mask = (data_frequency >= freq_bounds_electrolyte[0]) & (data_frequency <= freq_bounds_electrolyte[1])
data_masked = data_frequency[mask]
parameter_values["Electrolyte diffusivity [m2.s-1]"] = 3e-3
#parameter_values["Electrolyte conductivity [S.m-1]"] = 0.3
#parameter_values["Negative particle diffusivity [m2.s-1]"]= 1e-12
#parameter_values["Positive particle diffusivity [m2.s-1]"]= 1e-12
#parameter_values["Positive particle radius [m]"] = 3e-6
#parameter_values["Negative particle radius [m]"]=3e-6
#parameter_values["Thermodynamic factor"]= 2
print(parameter_values)
run(model="SPMe", data_frequency=data_frequency, parameter_values=parameter_values,data_Z_re=data_Z_re,data_Z_im=data_Z_im )
time_constants(parameter_values)

#%%
print(parameter_values)
#%%

"""
testing electrolyte section
"""
optim_params_electrolyte = {"Electrolyte diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(2e-7,2e-5,truncated_at = [1e-12,1e-1]),
                                                       transformation = pybop.LogTransformation(),
                                                    initial_value = 2e-7),     
"Electrolyte conductivity [S.m-1]":pybop.Parameter(pybop.Gaussian(0.4,0.4,truncated_at = [01e-5,1e4]),
                                                      transformation = pybop.LogTransformation(),
                                                        initial_value = 0.4),
#"Positive particle diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-4,2e-3,truncated_at = [1e-16,1e-2]), initial_value=1e-4,
                                                     # transformation = pybop.LogTransformation(),
                                                       # ),
#"Negative particle diffusivity [m2.s-1]" : pybamm.Parameter("Positive particle diffusivity [m2.s-1]"),
#"Positive particle radius [m]":pybop.Parameter(pybop.Gaussian(3e-5,6.5e-3,truncated_at = [1e-8,1e-1]),
                                                        #initial_value = 3e-5),
#"Negative particle radius [m]": pybamm.Parameter("Positive particle radius [m]")
}
electrolyte,electrolyte_cost = section_based_optimisation(model = "DFN",frequencies=data_frequency, f_bounds=freq_bounds_electrolyte,
                                         parameter_values=parameter_values,optim_params=optim_params_electrolyte,Z_real = data_Z_re,Z_imag= data_Z_im)

#%%
print(electrolyte)
parameter_values.update(electrolyte)
print(parameter_values)
run(model="DFN", data_frequency=data_frequency, parameter_values=parameter_values,data_Z_re=data_Z_re,data_Z_im=data_Z_im )
time_constants(parameter_values)
mask = (data_frequency >= freq_bounds_electrolyte[0]) & (data_frequency <= freq_bounds_electrolyte[1])
data_masked = data_frequency[mask]
print(data_masked)

#%%
'''
particle diffusion time scale
''' 

optim_params_diffusion = {"Positive particle diffusivity [m2.s-1]":pybop.Parameter(pybop.Gaussian(1e-12,2e-10,truncated_at = [1e-16,1e-6]),
                                                      transformation = pybop.LogTransformation(),initial_value=1e-12
                                                        ),
    "Negative particle diffusivity [m2.s-1]" : pybamm.Parameter("Positive particle diffusivity [m2.s-1]"),
    "Positive particle radius [m]":pybop.Parameter(pybop.Gaussian(9e-6,6.5e-3,truncated_at = [1e-8,1e-1]),
                                                        initial_value = 9e-6),
    "Negative particle radius [m]":pybamm.Parameter("Positive particle radius [m]"),}
diffusion, diffusion_cost = section_based_optimisation(model = "DFN",frequencies=data_frequency, f_bounds=freq_bounds_electrolyte,
                                         parameter_values=parameter_values,optim_params=optim_params_electrolyte,Z_real = data_Z_re,Z_imag= data_Z_im)

#%%
def time_constants(pv, F=96485.33, Rgas=8.314, T=298.15):
    R_p  = pv["Positive particle radius [m]"]
    D_s  = pv["Positive particle diffusivity [m2.s-1]"]
    D_e  = pv["Electrolyte diffusivity [m2.s-1]"]
    eps  = pv["Positive electrode porosity"]
    eps_sep = pv["Separator porosity"]
    b    = pv["Positive electrode Bruggeman coefficient (electrolyte)"]
    b_sep = pv["Separator Bruggeman coefficient (electrolyte)"]
    L_pos = pv["Positive electrode thickness [m]"]
    L    = pv["Negative electrode thickness [m]"] + pv["Separator thickness [m]"] + L_pos
    k    = pv["NMO reaction rate constant [m2.5.mol-0.5.s-1]"]
    ce0  = pv["Initial concentration in electrolyte [mol.m-3]"]
    c_init = pv["Initial concentration in positive electrode [mol.m-3]"]
    c_max  = pv["Maximum concentration in positive electrode [mol.m-3]"]
    Cdl    = pv["Positive electrode double-layer capacity [F.m-2]"]
    eps_am = pv["Positive electrode active material volume fraction"]
    kappa  = pv["Electrolyte conductivity [S.m-1]"]   # if callable: kappa = kappa(ce0)

    tau_d   = R_p**2 / D_s
    tau_e   = eps_sep * L**2 / (eps**b * D_e)
    tau_sep = L**2 / (eps_sep**(b_sep - 1) * D_e)

    j0 = F * k * np.sqrt(ce0 * c_init * (c_max - c_init))
    tau_rc = (Rgas * T / (F * j0)) * Cdl                      # semicircle RC

    a = 3 * eps_am / R_p    
    tau_pore = (L_pos**2 / (eps**b * kappa)) * a * Cdl        # de Levie / blocking

    for name, tau in [("particle diffusion", tau_d),
                      ("electrolyte (electrode)", tau_e),
                      ("electrolyte (separator)", tau_sep),
                      ("charge transfer RC", tau_rc),
                      ("pore / de Levie", tau_pore)]:
        print(f"{name:26s} tau = {tau:10.4g} s   ->  f = {1/(2*np.pi*tau):10.4g} Hz")
    return tau_d, tau_e, tau_sep, tau_rc, tau_pore
#%%
'''checking with a chayamuka dataset
'''
pv = pybamm.ParameterValues("Chayambuka2022")
chay_semi,chay_semi = section_based_optimisation(model = "DFN",frequencies=data_frequency, f_bounds=freq_bounds_semicircle,
                                         parameter_values=pv,optim_params=optim_params,Z_real = data_Z_re,Z_imag= data_Z_im)
print(chay_semi)

#%%
model= pybop.models.lithium_ion.GroupedSPMe()
model.print_parameter_info()

#%%
'''
section devoted to grouping parameters in SPMe
'''
parameter_values['Open-circuit voltage at 100% SOC [V]'] = parameter_values['Upper voltage cut-off [V]']
parameter_values['Open-circuit voltage at 0% SOC [V]'] = parameter_values['Lower voltage cut-off [V]']
parameter_values['Current function [A]'] = 0.003
grouped_parameter_values = pybop.models.lithium_ion.GroupedSPMe.create_grouped_parameters(parameter_values=parameter_values)
m_nmo = 96485 * 7.3e-12          # ≈ 7.04e-7 (A/m2)(m3/mol)^1.5
tau_ct = 96485 * 9e-6 / (m_nmo * np.sqrt(1000))   # ≈ 3.9e4 s
grouped_parameter_values["Positive electrode charge transfer time scale [s]"] = tau_ct
grouped_parameter_values["Negative electrode charge transfer time scale [s]"] = tau_ct
grouped_parameter_values["Minimum negative stoichiometry"] = 0.90   # x at 0% SOC
grouped_parameter_values["Maximum negative stoichiometry"] = 1.00   # x at 100% SOC
grouped_parameter_values["Minimum positive stoichiometry"] = 0.90   # y at 100% SOC
grouped_parameter_values["Maximum positive stoichiometry"] = 1.00   # y at 0% SOC
grouped_parameter_values["Initial SoC"] = 0.5 
Q_th = 96485 * 0.6 * 3.6e4 * 5e-5 * 1.5e-4    # F·α·c_max·L·A ≈ 15.6 A·s
grouped_parameter_values["Measured cell capacity [A.s]"] = 0.1 * Q_th 
grouped_parameter_values["Negative electrode charge transfer time scale [s]"] =pybamm.Parameter("Positive electrode charge transfer time scale [s]")
grouped_parameter_values["Negative electrode capacitance [F]"] =pybamm.Parameter("Positive electrode capacitance [F]")

print(grouped_parameter_values)
run(model="GroupedSPMe", data_frequency=data_frequency, parameter_values=grouped_parameter_values,data_Z_re=data_Z_re,data_Z_im=data_Z_im )

#%%
'''optimising semicircle using GROUPED PARAMETER VALUES
'''

optim_params_grouped= {
   "Positive electrode charge transfer time scale [s]":pybop.Parameter(pybop.Gaussian(38986,1e4,truncated_at=[1e-1,1e8]),transformation = pybop.LogTransformation()),
    "Series resistance [Ohm]": pybop.Parameter(pybop.Gaussian(53,5,truncated_at=[20,100]),initial_value=53),
    "Positive electrode capacitance [F]": pybop.Parameter(pybop.Gaussian(7e-6,1e-6, truncated_at=[1e-8,1e-2]),transformation=pybop.LogTransformation(),initial_value= 6e-6)
}
semi_circle,semi_circle_cost = section_based_optimisation(model = "GroupedSPMe",frequencies=data_frequency, f_bounds=freq_bounds_semicircle,
                                         parameter_values=grouped_parameter_values,optim_params=optim_params_grouped,Z_real = data_Z_re,Z_imag= data_Z_im)

#%%
f = 1/(2*np.pi * grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"])
#print(grouped_parameter_values["Positive electrode charge transfer time scale [s]"])
print("f",f)
frequencies =np.logspace(-4,4,200)
Z_fake = np.linspace(0,100,10)
#print(semi_circle)
grouped_parameter_values.update(semi_circle)
#print(grouped_parameter_values)
grouped_parameter_values["Negative electrode electrolyte diffusion time scale [s]"]=pybamm.Parameter("Positive electrode electrolyte diffusion time scale [s]")
grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"] = 60
grouped_parameter_values["Negative particle diffusion time scale [s]"] =pybamm.Parameter("Positive particle diffusion time scale [s]")
grouped_parameter_values["Positive particle diffusion time scale [s]"] = 0.0001
grouped_parameter_values["Separator electrolyte diffusion time scale [s]"]=60
print(1/(2 * np.pi * 25* grouped_parameter_values["Positive particle diffusion time scale [s]"])) 
run(model="GroupedSPMe", data_frequency=data_frequency, parameter_values=grouped_parameter_values,data_Z_re=data_Z_re,data_Z_im=data_Z_im )
cubic_interpolated = CubicSpline(data_Z_re[::-1],data_Z_im[::-1])
#plt.plot(Z_fake,cubic_interpolated(Z_fake))
#%%

tau_e_pos = pybamm.Parameter("Positive electrode electrolyte diffusion time scale [s]")
grouped_parameter_values["Negative electrode electrolyte diffusion time scale [s]"] = tau_e_pos
grouped_parameter_values["Separator electrolyte diffusion time scale [s]"] = 0.5074 * tau_e_pos
grouped_parameter_values["Negative particle diffusion time scale [s]"] = \
    pybamm.Parameter("Positive particle diffusion time scale [s]")
grouped_parameter_values["Negative electrode charge transfer time scale [s]"] = \
    pybamm.Parameter("Positive electrode charge transfer time scale [s]")
grouped_parameter_values["Negative electrode capacitance [F]"] = \
    pybamm.Parameter("Positive electrode capacitance [F]")
print(grouped_parameter_values)
optim_params_diffusion_grouped = {
    # particle diffusion: tail onset 1/(2π·τ_d); wide but excludes the semicircle band
    "Positive particle diffusion time scale [s]": pybop.Parameter(
        stats.loguniform(1e1, 1e7),          # f_onset between ~2e-8 and 1.6e-2 Hz... adjust to your f_min
        transformation=pybop.LogTransformation(),
        initial_value=1e4,                    # from the |ΔZ| inversion estimate
    ),
    # electrolyte diffusion: arc position 1/(2π·τ_e)
    "Positive electrode electrolyte diffusion time scale [s]": pybop.Parameter(
        stats.loguniform(1e0, 1e6),
        transformation=pybop.LogTransformation(),
        initial_value=1.8e3,                  # the physically-converted value
    ),
    # arc magnitude ∝ (1−t⁺)²; bound away from 1 so it can't zero-out the feature
    "Cation transference number": pybop.Parameter(
        pybop.Gaussian(0.4, 0.2, truncated_at=[0.05, 0.95]),
    ),
}
half_decade = np.log(10) / 2   # ≈ 1.15 in ln units
diffusion_grouped, diffusion_grouped = section_based_optimisation(model = "GroupedSPMe",frequencies=data_frequency, f_bounds=freq_bounds_electrolyte,
                                                                           parameter_values=grouped_parameter_values,optim_params=optim_params_diffusion_grouped,Z_real = data_Z_re,Z_imag= data_Z_im)

#%%
optim_params_full = {
    **optim_params_diffusion_grouped,
    "Positive electrode charge transfer time scale [s]": pybop.Parameter(
        stats.lognorm(s=half_decade, scale=6228),      # median = section fit
        transformation=pybop.LogTransformation(),
        initial_value=6228,
    ),
    "Positive electrode capacitance [F]": pybop.Parameter(
        stats.lognorm(s=half_decade, scale=2.44e-6),
        transformation=pybop.LogTransformation(),
        initial_value=2.44e-6,
    ),
    "Series resistance [Ohm]": pybop.Parameter(
        pybop.Gaussian(53.7, 3, truncated_at=[30, 90]),
        initial_value=53.7,
    ),
}

#%%
tau_e_prior = stats.lognorm(s=np.log(10), scale=4)
optim_params_electrolyte_grouped = {
"Positive electrode electrolyte diffusion time scale [s]":pybop.Parameter(tau_e_prior,transformation = pybop.LogTransformation()),
"Cation transference number ": pybop.Parameter(pybop.Gaussian(0.1,0.2,truncated_at=[0.01,1]))
}
electrolyte_grouped, electrolyte_cost_grouped = section_based_optimisation(model = "GroupedSPMe",frequencies=data_frequency, f_bounds=[data_frequency.min(),5e4],
                                         parameter_values=grouped_parameter_values,optim_params=optim_params_full,Z_real = data_Z_re,Z_imag= data_Z_im)

#%%
print(electrolyte_grouped)