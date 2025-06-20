# Code Vorbereitungen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp
import scipy.constants as spc
import kafe2
from uncertainties import ufloat as uf
import uncertainties.unumpy as unp
import uncertainties.umath as umath
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
def linear(x, m, b):
    return m * x + b
matplotlib.rcParams.update({'font.size': 17})
def linear2(x, x0, a):
    return (x-x0)*a
def fit_LasignThreshold(I_LT, U_LT):
    x = I_LT[:-8]
    y = U_LT[:-8]
    xerr = 1
    yerr = 0.01

    #Fit Linear Curve
    container = kafe2.XYContainer(x_data=x,y_data=y)
    container.label="Datapoints"
    container.axis_labels=["I in mA","U in V"]
    fit = kafe2.XYFit(container, model_function = linear2)
    fit.add_error(axis='x', err_val = xerr)
    fit.add_error(axis='y', err_val = yerr)
    result=fit.do_fit()
    
    #show fit
    plot = kafe2.Plot(fit)
    plot.plot()
    plot.show()
    
    #Found Parameter
    x0=uf(result['parameter_values']['x0'],result["parameter_errors"]["x0"])
    a=uf(result['parameter_values']['a'],result["parameter_errors"]["a"])

    #plot results
    plt.errorbar(I_LT,U_LT, xerr=xerr, yerr=yerr, fmt="o", label="Measurements")
    plt.plot(x, linear2(x, x0.n,a.n ), label="Fit")
    plt.vlines(x0.n, np.min(y), np.max(y), color="red", label="Threshold:"+str(x0))

    plt.legend()
    plt.xlabel("I in mA")
    plt.ylabel("U in V")
    #plt.savefig("fig/LaserThresholdFit.pdf", bbox_inches="tight")
    plt.show()    
    return x0
    
I_LT, U_LT = np.genfromtxt(r"E:/kit_masters/lab_course/laserspectroscopy/Laserspectroscopy.zip/Laserspectroscopy - Kopie/TEK00000.CSV", delimiter=",", skip_header=1).T
fit_LasignThreshold(I_LT, U_LT) 