#%%

import numpy as np
import pybamm 
import pybop
import matplotlib.pyplot as plt
import scipy as sc

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
    if model == "DFN":
        model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})  
    if model == "SPMe":
        model_pybop = pybamm.lithium_ion.SPMe(options = {"surface form":"differential", "contact resistance" : "true"})  
    elif model == "SPM":
        model_pybop = pybamm.lithium_ion.SPM(options = {"surface form":"differential", "contact resistance" : "true"})  
    simulator = pybop.pybamm.EISSimulator(model_pybop,parameter_values= parameter_values, f_eval= np.asarray(frequencies))
    cost= pybop.SumSquaredError(dataset=dataset,target="Impedance",weighting = "domain")
    cost.weighting = cost.weighting / np.abs(Impedance_data) **2 
    problem = pybop.Problem(simulator,cost);problem.set_target("Impedance")
    pybop.plot.nyquist(problem=problem)
    optim = pybop.CMAES(problem);optim.set_max_iterations(150);optim.set_max_unchanged_iterations(75)
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
    if model == "DFN":
        model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})  
    if model == "SPMe":
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

def run(model = "DFN" ,data_frequency = None,parameter_values= None,data_Z_re= None,data_Z_im = None):
    if model == "DFN":
        model_pybop = pybamm.lithium_ion.DFN(options = {"surface form":"differential", "contact resistance" : "true"})  
    if model == "SPMe":
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

#%%
frequencies = np.logspace(-4,4,100)
parameter_values = pybamm.ParameterValues("Chayambuka2022")
parameter_values["Negative electrode double-layer capacity [F.m-2]"] = 0.05
parameter_values["Positive electrode double-layer capacity [F.m-2]"] = 0.05
run(model= "DFN", data_frequency=frequencies, parameter_values=parameter_values)

