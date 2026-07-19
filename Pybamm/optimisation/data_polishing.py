
#%% #--preamble--#
import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from numpy.fft import fft, ifft, fftfreq
from scipy.interpolate import CubicSpline,make_interp_spline
import scienceplots
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
x_clean = x[mask];V_clean = V[mask]
#%%
#looking at the knots---#
x_sorted= np.sort(x_clean)
gaps = np.diff(x_sorted)
roe = gaps.max()/gaps.min()
print("min gap:    ", gaps.min())
print("max gap:    ", gaps.max())
print("mean gap:   ", gaps.mean())
print("median gap: ", np.median(gaps))
print("std of gaps:", gaps.std())
order = np.argsort(gaps)[::-1]      
for i in order[:5]:
    print(f"gap {gaps[i]:.4g} between x={x[i]:.4g} and x={x[i+1]:.4g}")
plt.style.use(["science"])
fig, ax = plt.subplots(2, 1, figsize=(8, 5))
ax[0].plot(x, np.zeros_like(x), '|', markersize=20)   # rug plot of sample locations
ax[0].set_title("sample locations")
ax[1].hist(gaps, bins=30)                              # distribution of spacings
ax[1].set_title("distribution of gaps")
plt.tight_layout()
plt.savefig("knots.png")
plt.show()
#%% ---fitting a cubic and then mapping to 0.5,1]


cubic = CubicSpline(x_clean,V_clean)
evals = np.linspace(0.5,1,200)
#%% -mapping this x_clean to [0.5,1] to match stoichiometry values
def y(x):
    return 0.5 * x + 0.5
Y = y(x_clean)
#%%--estimating error--
#using LOOCV method. Take an interior point out, interpolate with a cubic spline, compare error in predicted point to actual point
#loop through all points and taking interior point out
#Getting indicies
def error(x_clean,V_clean):
    err= []
    X = np.asarray(x_clean,float); V = np.asarray(V_clean,float)
    for i in range(1,len(x_clean)-1):
        cs = CubicSpline(np.delete(X,i),np.delete(V,i))
        err.append(cs(X[i]) -X[i])
    return np.abs(err).max() /16, err

#convergence rate is O(h^4)for C4 functions(cannot garauntee). so divide by 16 to account for error increase in 
#widening interval
residual = error(Y,V_clean)[1]
residual = [res / 16 for res in residual]
#Another Error method: Comparing to quintic
quintic = make_interp_spline(x_clean,V_clean,k=5)
#plt.plot(Y,quintic(Y))
#plt.plot(Y,cubic(Y))
err = quintic(evals) -cubic(evals)

#%%--trimming  OCV and plots
#if V is above an upper bound, trim it
trim = (cubic(Y)<=3) & (cubic(Y) >= 2.7)
plt.plot(Y,cubic(Y),label = "cubic")
plt.plot(Y[trim],cubic(Y[trim]),label = "trimmed")
plt.xlabel("SOC [.]")
plt.ylabel("OCV [V]")
plt.legend()
plt.savefig("(OCV-SOC)_plot.png", dpi = 300, bbox_inches = "tight")
plt.show()

fig, ax = plt.subplots(1,3,figsize = (6.5,4))
ax[0].plot(evals,quintic(evals),label= "quintic",linestyle = '--')
ax[1].set_title("LOOCV")
ax[1].set_ylabel("error [V]")
ax[0].set_xlabel("SOC [.]")
ax[0].set_ylabel("OCV [V]")
ax[0].plot(x_clean[int(len(x_clean)/2):],V_clean[int(len(x_clean)/2):], label ="data")
ax[0].plot(evals,cubic(evals), label = "cubic")
ax[2].plot(evals,err)
#ax.plot(evals,cs(evals,1), label= "first derivative of spline")
ax[0].legend(loc = "upper left")
#---error---#
ax[1].plot(Y[1:-1],residual)
ax[2].set_title("(w.r.t quintic)")
fig.tight_layout()
#plt.savefig("errorOCVSOC.png",dpi =300, bbox_inches = "tight")
plt.show()





