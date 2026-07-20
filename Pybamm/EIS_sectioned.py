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
    parameter_values = parameter_values.copy()
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
    exp_r = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh,
    submesh_params={"side": "right", "stretch": 6.0})
    submesh_types = model_pybop.default_submesh_types.copy()
    submesh_types["negative particle"] = exp_r
    submesh_types["positive particle"] = exp_r
    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 100, "r_p": 100} 
    simulator =pybop.pybamm.EISSimulator(
    model_pybop, parameter_values=parameter_values,
    f_eval=np.asarray(frequencies),
    var_pts=var_pts, submesh_types=submesh_types) 
    cost= pybop.SumSquaredError(dataset=dataset,target="Impedance",weighting = "domain")
    cost.weighting = cost.weighting / np.abs(Impedance_data) **2 
    problem = pybop.Problem(simulator,cost);problem.set_target("Impedance")
    pybop.plot.nyquist(problem=problem)
    optim = pybop.NelderMead(problem);optim.set_max_iterations(500);optim.set_max_unchanged_iterations(150)
    result = optim.run()
    pybop.plot.nyquist(problem=problem,inputs=result.best_inputs)
    print(result)
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
        exp_r = pybamm.MeshGenerator(
        pybamm.Exponential1DSubMesh,
        submesh_params={"side": "right", "stretch": 6.0})
        submesh_types = model_pybop.default_submesh_types.copy()
        submesh_types["negative particle"] = exp_r
        submesh_types["positive particle"] = exp_r
        var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 100, "r_p": 100}
        simulator_syn = pybop.pybamm.EISSimulator(
        model_pybop, parameter_values=parameter_values,
        f_eval=np.asarray(data_frequency),
        var_pts=var_pts, submesh_types=submesh_types) 
    elif model == "DFN":
        model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})
        simulator_syn = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(data_frequency))
  
    elif model == "SPMe":
        model_pybop = pybamm.lithium_ion.SPMe(options = {"surface form":"differential", "contact resistance" : "true"})  
        simulator_syn = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(data_frequency))

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
        ax[1].plot(data_frequency,np.abs(np.asarray(data_Z_re, dtype= float)- 1j * np.asarray(data_Z_im,dtype= float)))
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
        "Electrolyte conductivity [S.m-1]":0.0001,#sdecrease 0.88
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


#%% 
'''optimising the semiciricle:
1)trimming frequency range
2) choosing parameters to optimise in this region
3) running optimiser
'''
freq_bounds_semicircle = [5e0,5e4]
freq_bounds_electrolyte=[data_frequency.min(),5e1]
print(data_frequency.min())
freq_bounds_diffusion= [data_frequency.min(),0.06]
freq_bounds= freq_bounds_semicircle
frequency_plot(fbounds=freq_bounds,data_frequency=data_frequency,data_Z_im=data_Z_im,data_Z_re=data_Z_re)

#%%
'''
section devoted to grouping parameters in SPMe. 
Tying negative and positive parameters so i only need to optimise positive counterpart
first need to add required values. 
'''

parameter_values['Open-circuit voltage at 100% SOC [V]'] = parameter_values['Upper voltage cut-off [V]']
parameter_values['Open-circuit voltage at 0% SOC [V]'] = parameter_values['Lower voltage cut-off [V]']
parameter_values['Current function [A]'] = 0.003 #unused but required
grouped_parameter_values = pybop.models.lithium_ion.GroupedSPMe.create_grouped_parameters(parameter_values=parameter_values)
m_nmo = 96485 * 7.3e-12          # approx 7.04e-7 (A/m2)(m3/mol)^1.5
tau_ct = 96485 * 9e-6 / (m_nmo * np.sqrt(1000))   # approx 3.9e4 s
grouped_parameter_values["Positive electrode charge transfer time scale [s]"] = tau_ct
grouped_parameter_values["Negative electrode charge transfer time scale [s]"] = tau_ct

grouped_parameter_values["Minimum negative stoichiometry"] = 0.4   # x at 0% SOC
grouped_parameter_values["Maximum negative stoichiometry"] = 1.00   # x at 100% SOC
grouped_parameter_values["Minimum positive stoichiometry"] = 0.4   # y at 100% SOC
grouped_parameter_values["Maximum positive stoichiometry"] = 1.00   # y at 0% SOC
grouped_parameter_values["Initial SoC"] = 0.5
Q_th = 96485 * 0.6 * 3.6e4 * 5e-5 * 1.5e-4    # F·α·c_max·L·A approx  15.6 A·s
grouped_parameter_values["Measured cell capacity [A.s]"] = 0.1 * Q_th #starting value 


'''tying negative electrode to positive'''

grouped_parameter_values["Negative electrode charge transfer time scale [s]"] = pybamm.Parameter("Positive electrode charge transfer time scale [s]")
grouped_parameter_values["Negative electrode charge transfer time scale [s]"] =pybamm.Parameter("Positive electrode charge transfer time scale [s]")
grouped_parameter_values["Negative electrode capacitance [F]"] =pybamm.Parameter("Positive electrode capacitance [F]")
grouped_parameter_values["Negative particle diffusion time scale [s]"] = pybamm.Parameter("Positive particle diffusion time scale [s]")
grouped_parameter_values["Negative electrode electrolyte diffusion time scale [s]"] = pybamm.Parameter("Positive electrode electrolyte diffusion time scale [s]")

#%%
'''
setting time scales to be in the correct frequency range
'''
grouped_parameter_values["Positive particle diffusion time scale [s]"] = 70
grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"] = 0.43
f = 1/(2*np.pi * grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"])
print("f",f)
print("f_diff",1/(2 * np.pi* grouped_parameter_values["Positive particle diffusion time scale [s]"])) 
grouped_parameter_values["Separator electrolyte diffusion time scale [s]"] = 0.507 * grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"] 
grouped_parameter_values["Cation transference number"] = 0.4
grouped_parameter_values["Measured cell capacity [A.s]"] = 0.00685975 #1.56 initially.dropping to increase impedance
print((func(0.955).evaluate()-func(.945).evaluate())/0.01)
grouped_parameter_values["Separator electrolyte diffusion time scale [s]"] = (
    0.507 * pybamm.Parameter("Positive electrode electrolyte diffusion time scale [s]"))

run(model="GroupedSPMe", data_frequency=data_frequency, parameter_values=grouped_parameter_values,data_Z_re=data_Z_re,data_Z_im=data_Z_im );
grouped_parameter_values.update({'Positive electrode charge transfer time scale [s]': np.float64(26.767014971554683), 'Positive electrode capacitance [F]': np.float64(2.40963741500189e-06), 'Series resistance [Ohm]': np.float64(53.65688976571483)}
)

#%%
print(grouped_parameter_values)
optim_params_grouped= {
   "Positive electrode charge transfer time scale [s]":pybop.Parameter(stats.loguniform(1e1, 1e5),transformation = pybop.LogTransformation(),initial_value=36),
    "Series resistance [Ohm]": pybop.Parameter(pybop.Gaussian(53,25,truncated_at=[1,100]),initial_value=53),
    "Positive electrode capacitance [F]": pybop.Parameter(pybop.Gaussian(7e-6,1e-6, truncated_at=[1e-8,1e-2]),transformation=pybop.LogTransformation(),initial_value= 2e-6),
    "Measured cell capacity [A.s]": pybop.Parameter(
        stats.loguniform(1e-4, 1),
        transformation=pybop.LogTransformation(), initial_value=0.19),
    "Positive electrode electrolyte diffusion time scale [s]": pybop.Parameter(
        stats.loguniform(1e-2, 1e5),
       transformation=pybop.LogTransformation(), initial_value=2.5),
    "Positive particle diffusion time scale [s]": pybop.Parameter(
    stats.loguniform(1e1, 5e4),
    transformation=pybop.LogTransformation(), initial_value=85)
}
semi_circle_2,semi_circle_cost_1 = section_based_optimisation(model = "GroupedSPMe",frequencies=data_frequency, f_bounds=[data_frequency.min(),5e4],
                                         parameter_values=grouped_parameter_values,optim_params=optim_params_grouped,Z_real = data_Z_re,Z_imag= data_Z_im)

#%%
print(semi_circle_2)

#grouped_parameter_values.update({'Measured cell capacity [A.s]': np.float64(0.01965890193874428), 'Positive electrode electrolyte diffusion time scale [s]': np.float64(0.010015649293352493), 'Positive electrode charge transfer time scale [s]': np.float64(21.316085210672018), 'Positive electrode capacitance [F]': np.float64(2.1180051993279087e-06), 'Series resistance [Ohm]': np.float64(53.261847744996395)})
#%%
'''looking at results. seeing if anything clamped to a bound'''

#print({'Positive electrode charge transfer time scale [s]': np.float64(26.767014971554683), 'Positive electrode capacitance [F]': np.float64(2.40963741500189e-06), 'Series resistance [Ohm]': np.float64(53.65688976571483)}
)
#print(grouped_parameter_values) #ensure that the negative electrode is tied.
#grouped_parameter_values["Separator electrolyte diffusion time scale [s]"] = (
   # 0.507 * pybamm.Parameter("Positive electrode electrolyte diffusion time scale [s]"))
#%%
#print(semi_circle_2)
grouped_parameter_values.update(semi_circle_2)

#%%
print(grouped_parameter_values["Positive particle diffusion time scale [s]"],grouped_parameter_values["Measured cell capacity [A.s]"]) #70,0.005
#grouped_parameter_values["Positive particle diffusion time scale [s]"]= 70
#print(grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"]) #0.008
#grouped_parameter_values["Measured cell capacity [A.s]"] = 0.00685975
##grouped_parameter_values["Positive particle diffusion time scale [s]"] = 80
#grouped_parameter_values["Negative particle diffusion time scale [s]"] = grouped_parameter_values["Positive particle diffusion time scale [s]"]
#grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"]=0.60
#grouped_parameter_values["Cation transference number"] = .4
#grouped_parameter_values["Negative electrode electrolyte diffusion time scale [s]"] = grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"]
f = 1/(2*np.pi * grouped_parameter_values["Positive electrode electrolyte diffusion time scale [s]"])
print("f",f)
print("f_diff",1/(2 * np.pi* grouped_parameter_values["Positive particle diffusion time scale [s]"])) 

model_pybop = pybop.models.lithium_ion.GroupedSPMe(
options={"surface form": "differential", "contact resistance": "true"})
exp_r = pybamm.MeshGenerator(
pybamm.Exponential1DSubMesh,
submesh_params={"side": "right", "stretch": 10.0})
submesh_types = model_pybop.default_submesh_types.copy()
submesh_types["negative particle"] = exp_r
submesh_types["positive particle"] = exp_r
var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 100, "r_p": 100}
simulator_syn = pybop.pybamm.EISSimulator(
            model_pybop, parameter_values=grouped_parameter_values,
            f_eval=np.asarray(data_frequency),
            var_pts=var_pts, submesh_types=submesh_types) 
imp = simulator_syn.solve()["Impedance"].data
plt.plot(np.real(imp),-np.imag(imp))
plt.plot(data_Z_re,data_Z_im)
print(max(-np.imag(imp)))
print(data_Z_im.max())
print(grouped_parameter_values)
run(model = "GroupedSPMe",data_frequency=data_frequency,parameter_values=grouped_parameter_values,data_Z_re=data_Z_re,data_Z_im=data_Z_im)
#%% 
#grouped_parameter_values["Measured cell capacity [A.s]"] = 0.00685975 #1.56 initially.dropping to increase impedance
'''optimising the electrolyte/diffusion zone'''
grouped_parameter_values["Separator electrolyte diffusion time scale [s]"] = (
    0.507 * pybamm.Parameter("Positive electrode electrolyte diffusion time scale [s]"))
'''setting up second optimisation'''
optim_params_lowfreq = {
    "Positive particle diffusion time scale [s]": pybop.Parameter(
        stats.loguniform(1e-4, 5e5),
        transformation=pybop.LogTransformation(), initial_value=80),
    #"Measured cell capacity [A.s]": pybop.Parameter(
       # pybop.Uniform(0.00685975/ 1.2,0.00685975*1.2),
        #transformation=pybop.LogTransformation(), initial_value=0.00685975),
"Positive electrode electrolyte diffusion time scale [s]": pybop.Parameter(
        stats.loguniform(1e-2, 1e4),
        transformation=pybop.LogTransformation(), initial_value=0.43351564441412865),
}
low_freq,low_freq_cost = section_based_optimisation(model = "GroupedSPMe",frequencies=data_frequency, f_bounds = freq_bounds_electrolyte,
                                         parameter_values=grouped_parameter_values,optim_params=optim_params_lowfreq,Z_real = data_Z_re,Z_imag= data_Z_im)

#%%
print(1/(2*np.pi * 0.43351564441412865))
print(low_freq)
grouped_parameter_values.update(low_freq)
run(model= "GroupedSPMe", data_frequency=data_frequency,parameter_values=grouped_parameter_values,data_Z_re=data_Z_re,data_Z_im=data_Z_im)
optim_params_diffusion = {
    "Positive particle diffusion time scale [s]": pybop.Parameter(
        stats.loguniform(1e1, 5e2),
        transformation=pybop.LogTransformation(), initial_value=85),}
diff,diff_cost = section_based_optimisation(model = "GroupedSPMe",frequencies=data_frequency, f_bounds=freq_bounds_diffusion,
                                         parameter_values=grouped_parameter_values,optim_params=optim_params_diffusion,Z_real = data_Z_re,Z_imag= data_Z_im)
#%%
'''diffusion tail'''
#%%
'''plan to run this measured cell capacity in the semiciricle optimiser and then run the entire thing '''
'''electrolyte info'''

optim_params_grouped= {
   "Positive electrode charge transfer time scale [s]":pybop.Parameter(pybop.Gaussian(38986,1e6,truncated_at=[1e-1,1e8]),transformation = pybop.LogTransformation()),
    "Series resistance [Ohm]": pybop.Parameter(pybop.Gaussian(53,25,truncated_at=[1,100]),initial_value=53),
    "Positive electrode capacitance [F]": pybop.Parameter(pybop.Gaussian(7e-6,1e-6, truncated_at=[1e-8,1e-2]),transformation=pybop.LogTransformation(),initial_value= 6e-6)
}
semi_circle_2,semi_circle_cost_1 = section_based_optimisation(model = "GroupedSPMe",frequencies=data_frequency, f_bounds=freq_bounds_semicircle,
                                         parameter_values=grouped_parameter_values,optim_params=optim_params_grouped,Z_real = data_Z_re,Z_imag= data_Z_im)


#%%
grouped_parameter_values.update(low_freq)
#grouped_parameter_values.update({'Measured cell capacity [A.s]': np.float64(0.0068597500000000065), 'Positive particle diffusion time scale [s]': np.float64(136.99999999999673), 'Positive electrode electrolyte diffusion time scale [s]': np.float64(109.39556433526737)})

model_pybop = pybop.models.lithium_ion.GroupedSPMe(
options={"surface form": "differential", "contact resistance": "true"})
exp_r = pybamm.MeshGenerator(
pybamm.Exponential1DSubMesh,
submesh_params={"side": "right", "stretch": 6.0})
submesh_types = model_pybop.default_submesh_types.copy()
submesh_types["negative particle"] = exp_r
submesh_types["positive particle"] = exp_r
var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 100, "r_p": 100}
simulator_syn = pybop.pybamm.EISSimulator(
            model_pybop, parameter_values=grouped_parameter_values,
            f_eval=np.asarray(data_frequency),
            var_pts=var_pts, submesh_types=submesh_types) 
imp = simulator_syn.solve()["Impedance"].data
plt.plot(np.real(imp),-np.imag(imp))

plt.plot(data_Z_re,data_Z_im)
print(max(-np.imag(imp)))
print(data_Z_im.max())
#%%

print(low_freq)
print(grouped_parameter_values)

