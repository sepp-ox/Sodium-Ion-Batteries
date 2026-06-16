
#%% #--preamble--#
import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.interpolate import CubicSpline
#%%
#---description of file---#
#imports Q,OCV dataset with stoich [0.5,1]
# linear map of Q to [0,1]
# polynomial cubic fit
#linear map to [0,5.1]
#voltage trimming
#findings: cubic perfectly interpolates
#%%--getting data---#
data = pd.read_csv(r"C:\Users\sepps\OneDrive\Oxford\diss\Pybamm\data\NMO_eq_charge.csv")
#%%---scaling to [0,1]- using Q.max()---#
def stoich(Q, data = None, Qmax = None, Qmin= None):
    if Qmax is None:
        Qmax = data["Q"].max()
    if Qmin is None:
        Qmin = data["Q"].min()
    return (Q - Qmin)/ (Qmax - Qmin)
Q = data["Q"]
V= data["Ewe"]
x= stoich(Q, data=data)
#%%-cleaning data pairs that include a nan---#
mask = ~np.isnan(x) & ~np.isnan(V)
x_clean = x[mask]
V_clean = V[mask]
#%% ---fitting a cubic and then mapping to 0.5,1]
cubic = CubicSpline(x_clean,V_clean)
evals = np.linspace(0,1,200)
#%% -mapping this x_clean to [0.5,1] to match stoichiometry values
def y(x):
    return 0.5 * x + 0.5
Y = y(x_clean)

#%%--trimming  OCV and plots
#if V is above an upper bound, trim it
trim = (cubic(Y)<=3) & (cubic(Y) >= 2.7)
plt.plot(Y,cubic(Y))
plt.plot(Y[trim],cubic(Y[trim]),label = "trimmed")
plt.xlabel("SOC [.]")
plt.ylabel("OCV [V]")
plt.legend()
plt.savefig("(OCV-SOC)_plot.png", dpi = 300, bbox_inches = "tight")
plt.show()
#-error-#
err = cubic(x_clean)- V_clean
fig, ax = plt.subplots(1,2,figsize = (6.5,4))
ax[0].plot(x_clean,V_clean, label ="data")
ax[0].plot(evals,cubic(evals), label = "cubic")
ax[1].set_title("error")
ax[0].set_xlabel("SOC [.]")
ax[0].set_ylabel("OCV [V]")
#ax.plot(evals,cs(evals,1), label= "first derivative of spline")
ax[0].legend(loc = "upper left")
#---error---#
ax[1].plot(x_clean,err)
plt.savefig("errorOCVSOV.png",dpi =300, bbox_inches = "tight")
plt.show()




