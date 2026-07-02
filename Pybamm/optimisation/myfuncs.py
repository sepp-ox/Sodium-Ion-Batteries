
import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from numpy.fft import fft, ifft, fftfreq
from scipy.interpolate import CubicSpline,make_interp_spline

def cubic(x,V):
    return CubicSpline(x,V)

def stoich(Q, data = None, Qmax = None, Qmin= None):
    if Qmax is None:
        Qmax = data["Q"].max()
    if Qmin is None:
        Qmin = data["Q"].min()
    return (Q - Qmin)/ (Qmax - Qmin)

def error(x_clean,V_clean):
    err= []
    X = np.asarray(x_clean,float); V = np.asarray(V_clean,float)
    for i in range(1,len(x_clean)-1):
        cs = CubicSpline(np.delete(X,i),np.delete(V,i))
        err.append(cs(X[i]) -X[i])
    return np.abs(err).max() /16, err

data = pd.read_csv(r"C:\Users\sepps\OneDrive\Oxford\diss\Pybamm\data\NMO_eq_charge.csv")
Q = data["Q"]
V= data["Ewe"]
x= stoich(Q, data=data)
mask = ~np.isnan(x) & ~np.isnan(V)
x_clean = x[mask].to_numpy();V_clean = V[mask].to_numpy()
def y(x):
        return 0.5 * x + 0.5
Y = y(x_clean)
if __name__ == "__main__":
    quintic = make_interp_spline(x_clean,V_clean,k=5)
    cubic = CubicSpline(x_clean,V_clean)
    evals = np.linspace(0.5,1,200)
    residual = error(Y,V_clean)[1]
    residual = [res / 16 for res in residual]
#Another Error method: Comparing to quintic
    quintic = make_interp_spline(x_clean,V_clean,k=5)
#plt.plot(Y,quintic(Y))
#plt.plot(Y,cubic(Y))
    err = quintic(evals) -cubic(evals)

#%%--trimming  OCV and plots
#if V is above an upper bound, trim it

#%%--estimating error--
#using LOOCV method. Take an interior point out, interpolate with a cubic spline, compare error in predicted point to actual point
#loop through all points and taking interior point out
#Getting indicies




