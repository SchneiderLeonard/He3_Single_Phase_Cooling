# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:28:56 2019

@author: L.Schneider
"""

# program to call he3pak from python
# by Jeff
# Thu May 23 08:48:56 CDT 2019 wrote this line

from numpy import pi
from math import *

# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import array as arr

# command line options
from optparse import OptionParser


from sys import platform
if any([platform.startswith(os_name) for os_name in ['linux', 'darwin', 'freebsd']]):
    import zugbruecke as ctypes
elif platform.startswith('win'):
    import ctypes
else:
    # Handle unsupported platforms
    print("Unknown platform")
    exit(1)
# Linux version:
# import zugbruecke as ctypes
    
    
# parse command line arguments

parser = OptionParser()

parser.add_option("-b", "--boundary",
                  action="store_true", dest="yes_boundary", default=False,
                  help="Include boundary layer")

(options, args) = parser.parse_args()
    

# first example:  calling DFPTdll
# "Density from Pressure and Temperature"

DFPTdll = ctypes.windll.LoadLibrary('he3eos.dll').DFPTdll
DFPTdll.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
# DFPTdll(dkgm3,idid,pres,Tk)

# second example:  calling Fundtdll
# Getting any state variable (40) from Density and Temperature

#        Xprop(1) = Tk
#        Xprop(2) = pres
#        Xprop(3) = dkgm3
#        Xprop(4) = volm
#        Xprop(5) = compfactor
#        Xprop(6) = enth
#        Xprop(7) = entr
#        Xprop(8) = cvsi
#        Xprop(9) = cpsi
#        Xprop(10) = helm
#        Xprop(11) = gibb
#        Xprop(12) = sound
#        Xprop(13) = Latent heat (hvap)
#        Xprop(14) = JTcoef
#        Xprop(15) = inte
#        Xprop(16) = adiacomp
#        Xprop(17) = isocomp
#        Xprop(18) = volexp
#        Xprop(19) = isenexp
#        Xprop(20) = sndvirial
#        Xprop(21) = dBdT
#        Xprop(22) = trdvirial
#        Xprop(23) = dpdd
#        Xprop(24) = dpdt
#        Xprop(25) = dddt
#        Xprop(26) = dpdds
#        Xprop(27) = grun
#
#        Xprop(28) = Con
#        Xprop(29) = Vis
#        Xprop(30) = Vis / dkgm3
#        Xprop(31) = STEN(Tk)
#        Xprop(32) = Vis * cpsi / Con
#        Xprop(33) = Con / dkgm3 / cpsi
#        Xprop(34) = "Dielectric constant"  #not defined yet
#        Xprop(35) = "Refractive index"  #not defined yet
#
#        Xprop(36) = "2nd sound velocity"  #not defined yet
#        Xprop(37) = "4th sound velocity"  #not defined yet
#        Xprop(38) = "Superfluid density fraction"  #not defined yet
#        Xprop(39) = "GM mutual friction parameter" #not defined yet
#        Xprop(40) = "Superfluid thermal conductivity"  #not defined yet

Fundtdll = ctypes.windll.LoadLibrary('he3eos.dll').Fundtdll
Fundtdll.argtypes = (ctypes.POINTER(ctypes.c_double*40),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
#Fundtdll(Xprop,dkgm3,Tk)

# third example:  calling SatXFunTdll
# Getting either the liquid or vapour properties at saturation from Temperature


#        XpropV(1) = Tk
#        XpropV(2) = pres
#        XpropV(3) = densV
#        XpropV(4) = volmV
#        XpropV(5) = compfactorV
#        XpropV(6) = enthV
#        XpropV(7) = entrV
#        XpropV(8) = cvsiV
#        XpropV(9) = cpsiV
#        XpropV(10) = helmV
#        XpropV(11) = gibbV
#        XpropV(12) = soundV
#        XpropV(13) = heatvapor
#        XpropV(14) = JTcoefV
#        XpropV(15) = inteV
#        XpropV(16) = adiacompV
#        XpropV(17) = isocompV
#        XpropV(18) = volexpV
#        XpropV(19) = isenexpV
#        XpropV(20) = sndvirial
#        XpropV(21) = dBdT
#        XpropV(22) = trdvirial
#        XpropV(23) = dpddV
#        XpropV(24) = dpdtV
#        XpropV(25) = dddTV
#        XpropV(26) = dpddsV
#        XpropV(27) = grunV
#
#        XpropV(28) = ThconV
#        XpropV(29) = viscV
#        XpropV(30) = viscV / densV
#        XpropV(31) = SurfT
#        XpropV(32) = viscV * cpsiV / ThconV
#        XpropV(33) = ThconV / densV / cpsiV
#        XpropV(34) = "Dielectric constant"  #not defined yet
#        XpropV(35) = "Refractive index"  #not defined yet
#
#        XpropV(36) = "2nd sound velocity"  #not defined yet
#        XpropV(37) = "4th sound velocity"  #not defined yet
#        XpropV(38) = "Superfluid density fraction"  #not defined yet
#        XpropV(39) = "GM mutual friction parameter"  #not defined yet
#        XpropV(40) = "Superfluid thermal conductivity"  #not defined yet
#c
#      ! isoexp, dvdt,alfa not used, should be used in the future.
#        0 XpropL(1) = Tk
#        1 XpropL(2) = pres
#        2 XpropL(3) = densL
#        3 XpropL(4) = volmL
#        4 XpropL(5) = compfactorL
#        5 XpropL(6) = enthL
#        6 XpropL(7) = entrL
#        7 XpropL(8) = cvsiL
#        8 XpropL(9) = cpsiL
#        9 XpropL(10) = helmL
#        10 XpropL(11) = gibbL
#        11 XpropL(12) = soundL
#        12 XpropL(13) = heatvapor
#        13 XpropL(14) = JTcoefL
#        14 XpropL(15) = inteL
#        15 XpropL(16) = adiacompL
#        16 XpropL(17) = isocompL
#        17 XpropL(18) = volexpL
#        18 XpropL(19) = isenexpL
#        19 XpropL(20) = sndvirial
#        20 XpropL(21) = dBdT
#        21 XpropL(22) = trdvirial
#        22 XpropL(23) = dpddL
#        23 XpropL(24) = dpdtL
#        24 XpropL(25) = dddTL
#        25 XpropL(26) = dpddsL
#        26 XpropL(27) = grunL
#c
#        27 XpropL(28) = ThconL
#        28 XpropL(29) = viscL
#        29 XpropL(30) = viscL / densL
#        30 XpropL(31) = SurfT
#        31 XpropL(32) = viscL * cpsiL / ThconL
#        32 XpropL(33) = ThconL / densL / cpsiL
#        XpropL(34) = "Dielectric constant"  #not defined yet
#        XpropL(35) = "Refractive index"  #not defined yet
#c
#        XpropL(36) = "2nd sound velocity"  #not defined yet
#        XpropL(37) = "4th sound velocity"  #not defined yet
#        XpropL(38) = "Superfluid density fraction"  #not defined yet
#        XpropL(39) = "GM mutual friction parameter"  #not defined yet
#        XpropL(40) = "Superfluid thermal conductivity"  #not defined yet

SatXFunTdll = ctypes.windll.LoadLibrary('he3eos.dll').SatXFunTdll
SatXFunTdll.argtypes = (ctypes.POINTER(ctypes.c_double*40),ctypes.POINTER(ctypes.c_double*40),ctypes.POINTER(ctypes.c_double))


# Fourth example:  TIPsatdll
# Temperature from Pressure at Saturation

TIPsatdll = ctypes.windll.LoadLibrary('he3eos.dll').TIPsatdll
TIPsatdll.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))

#
#CALCULATIONS------------------------------------------------------------------------------------------------------------------------

#_DIFFERENT CASES WE LOOKED INTO

# Parameter in standard case: d=6.0mm;  Tstart=3.2;     Tbath=1.6;  Tend=2.25;  mdot=1.1;   step=0.01;  p0=30000;   TWALL IS EQUIVALENT TO THE HE4 BATH
#(1) All comments beginning with (1) are needed to claculate the length including the boundary layer.


diameter = 6. # mm
Tstart = 4. # K; Temperature of He3 at Inlet of first gaseous cooling part
Tbath = 1.4 # K
Tend2 = 1.5 # K; Temperature of He3 at Outlet of last liquid cooling part
if options.yes_boundary:
    Tboundary = (Tstart + Tbath)/2
mdot = 0.5 # g/s
step = 0.01 # m 
p0 = 25000. # Pa

ell  = 0.0  # m; starting position


d_m = diameter/1000 # m
mdot_m = mdot/1000 # kg/s

Tk = ctypes.c_double()
pres = ctypes.c_double(p0)
TIPsatdll(Tk,pres)
temperature = Tk.value

Tend = Tk.value #K

Xprop = (ctypes.c_double*40)()
 
elllist = []
Tlist = []
Blist = []


while Tstart > Tend:    
    
    #Def rho
    T_f = ctypes.c_double(Tstart)
    pres = ctypes.c_double(p0)
    dkgm3 = ctypes.c_double()
    idid = ctypes.c_int()
    
    DFPTdll(dkgm3,idid,pres,T_f)
    #print(dkgm3.value,idid.value)      
    rho = ctypes.c_double(dkgm3.value)

    #Calc rest of Parameters
    Fundtdll(Xprop,dkgm3,T_f)
    
    C_p = Xprop[8]
    mu = Xprop[28]
    Re = 4*mdot_m/(pi*d_m)/mu
    Pr = Xprop[31]
    Nu = 0.023*Re**0.8*Pr**0.3
    k_f = Xprop[27]
    h = Nu*k_f/d_m
    if not options.yes_boundary:
        dT_f = h*pi*d_m*(Tbath-Tstart)/mdot_m/C_p*step #in case (1): comment this line out
    else:
        dT_f = h*pi*d_m*(Tboundary-Tstart)/mdot_m/C_p*step
#REST PARAMETERS

    elllist.append(ell)
    Tlist.append(Tstart)
    if options.yes_boundary:
        Blist.append(Tboundary)    
    
    ell = ell + step
    Tstart = Tstart + dT_f
    if options.yes_boundary:
        Tboundary = (Tstart + Tbath)/2
    
print("{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4e} {:10.4f} {:10.4f} {:10.4f} {:10.4e} {:10.4f} {:10.4e}".format(ell,Tstart,rho.value,C_p,mu,Re,Pr,Nu,k_f,h,dT_f))
 
    
#Twall = 1.6 # K    
x0 = 1.0 # quality starts as vapour
x1 = 0.0 # quality ends as liquid
x = x0 # starting quality

if options.yes_boundary:
    T_boundary = (Tk.value + Tbath)/2

XpropL = (ctypes.c_double*40)()
XpropV = (ctypes.c_double*40)()
SatXFunTdll(XpropL,XpropV,Tk)
xlist = []
dQdLlist = []

while x > 0:

    rho_V = XpropV[2] # kg/m^3
    h_V = XpropV[5] # J/kg
    rho_L = XpropL[2] # kg/m^3
    h_L = XpropL[5] # J/kg
    mu_L = XpropL[28] # Pa*s
    Re_0 = 4 * (mdot/1000.)/(pi*(diameter/1000.)*mu_L)
    Pr_L = XpropL[31]
    Nu = 0.023 * Re_0**0.8 * Pr_L**(1./3.)*sqrt((1-x)+rho_L/rho_V*x)
    k_L = XpropL[27] # W/(m*K)
    hc = Nu*k_L/(diameter/1000.)  # W/(m^2*K)
    
    if not options.yes_boundary:
        dQ_dot = hc*(pi*diameter/1000.)*(temperature-Tbath)*step # W
    else:
        dQ_dot = hc*(pi*diameter/1000.)*(temperature-T_boundary)*step # W
    
    dx = dQ_dot/(-(mdot/1000.)*(h_V-h_L))
    dQ_dot_over_dL = dQ_dot/step # W/m
    #print(ell,Tk.value,p0,x,rho_V,h_V,rho_L,h_L,mu_L,Re_0,Pr_L,Nu,k_L,hc,dQ_dot,dx,dQ_dot_over_dL)
    #further parameters for later calculations
    mu_V = XpropV[28] 


    xlist.append(x)
    elllist.append(ell)
    Tlist.append(Tk.value)
    dQdLlist.append(dQ_dot_over_dL)
    if options.yes_boundary:
        Blist.append(T_boundary)
    
    x = x + dx
    ell = ell + step
    
print(ell,Tk.value,p0,x,rho_V,h_V,rho_L,h_L,mu_L,Re_0,Pr_L,Nu,k_L,hc,dQ_dot,dx,dQ_dot_over_dL)

while Tstart > Tend2:    
    
    #Def rho
    T_f = ctypes.c_double(Tstart)
    pres = ctypes.c_double(p0)
    dkgm3 = ctypes.c_double()
    idid = ctypes.c_int()
    
    DFPTdll(dkgm3,idid,pres,T_f)
    #print(dkgm3.value,idid.value)      
    rho = ctypes.c_double(dkgm3.value)

    #Calc rest of Parameters
    Fundtdll(Xprop,dkgm3,T_f)
    
    C_p = Xprop[8]
    mu = Xprop[28]
    Re = 4*mdot_m/(pi*d_m)/mu
    Pr = Xprop[31]
    Nu = 0.023*Re**0.8*Pr**0.3
    k_f = Xprop[27]
    h = Nu*k_f/d_m
    if not options.yes_boundary:
        dT_f = h*pi*d_m*(Tbath-Tstart)/mdot_m/C_p*step #in case (1): comment this line out
    else:
        dT_f = h*pi*d_m*(Tboundary-Tstart)/mdot_m/C_p*step
#REST PARAMETERS

    elllist.append(ell)
    Tlist.append(Tstart)
    if options.yes_boundary:
        Blist.append(Tboundary)
    
    ell = ell + step
    Tstart = Tstart + dT_f
    if options.yes_boundary:
        Tboundary = (Tstart + Tbath)/2
    
print("{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4e} {:10.4f} {:10.4f} {:10.4f} {:10.4e} {:10.4f} {:10.4e}".format(ell,Tstart,rho.value,C_p,mu,Re,Pr,Nu,k_f,h,dT_f))



plt.plot(elllist,Tlist)
#custom size of graph _=plt.plot(3,1)


#plt.plot(elllist,p(elllist),"r--")
#plt.title("$T=%.3f(\mathrm{K m}^{-2})L^2+(%.3f)(\mathrm{K m}^{-1})L+%.3f\mathrm{K}$"%(z[0],z[1],z[2]))
if options.yes_boundary:
    plt.plot(elllist,Blist,color='orange')
plt.plot([elllist[0],elllist[-1]],[temperature,temperature],"--",color="cyan")
plt.xlabel('L [m]')     
plt.ylabel('T [K]')         
plt.show()
