# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:14:41 2019

@author: L.Schneider, J.W.Martin
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


from sys import platform
if any([platform.startswith(os_name) for os_name in ['linux', 'darwin', 'freebsd']]):
    calc = ctypes.cdll.LoadLibrary('./heprop.so').calc_
elif platform.startswith('win'):
    calc = ctypes.cdll.LoadLibrary('./hepak_source/heprop.dll').calc_
else:
    # Handle unsupported platforms
    print("Unknown platform")
    exit(1)
# Linux version:
# calc = ctypes.cdll.LoadLibrary('./heprop.so').calc_


#------------------------------------------------------------------------------

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




calc.argtypes=(
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER((ctypes.c_double*42)*3),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
)
'''
Usage:
In relation to the HEPAK xla file, which uses:
=HeCalc(Index, Phase, Input1, Value1, Input2, Value2, Unit)    

In the example below,
Input1, Value1, Input2, Value2 correspond to JIN1,VALU1,JIN2,VALU2
Unit corresponds to NUNITS
Index is j
Phase is i.  It is kind of returned by IDID.
If single phase, then i=0 should be used.
If dual phase, then i=1 has the saturated liquid and i=2 has the vapour.
'''

# inputs
NPRCIS=ctypes.c_int(3)
NUNITS=ctypes.c_int(1)
JOUT=ctypes.c_int(11111)
JIN1=ctypes.c_int(2)
VALU1=ctypes.c_double(1.6)
JIN2=ctypes.c_int(14)
VALU2=ctypes.c_double(0)

# returns
IDID=ctypes.c_int()
PROP=((ctypes.c_double*42)*3)()

calc(IDID,PROP,JOUT,JIN1,VALU1,JIN2,VALU2,NPRCIS,NUNITS)

#for i in range(3):
    #for j in range(42):
        #print(i,j,PROP[i][j])


#
#CALCULATIONS-----------------------------------------------------------------------------------------------------------------------------------------------------

#GASEOUS STATE-----------------------------------------------------------------
        
#_DIFFERENT CASES WE CAN LOOK INTO
# Parameter in standard case: d=6.0mm;  Tstart=3.2;  Tbath=1.6;  Tend=2.25;  mdot=1.1;  step=0.01;  p0=30000;   TWALL IS EQUIVALENT TO THE HE4 BATH

p0 = 25000. # Pa

# temperature at saturation
Tk = ctypes.c_double()
pres = ctypes.c_double(p0)
TIPsatdll(Tk,pres)
saturation_temperature=Tk.value

diameter = 6. # mm
Tstart = 4. # K 
Tbath = 1.4 # K
Tend = saturation_temperature # K; end of vapour phase cooling
Tend2 = 1.5 # K; end of liquid phase cooling
mdot = 0.5 # g/s
step = 0.01 # m 

x0 = 1.0 # quality starts as vapour
x1 = 0.0 # quality ends as liquid
x = x0 # starting quality

ell = 0.0  # m; starting position

deltax = 0.005 # m; thickness of wall
k_cu = 384.1 # W/m-K
sigma_k = 6.5*21 # W/K^4-m^2

d_m = diameter/1000 # m
mdot_m = mdot/1000 # kg/s


Xprop = (ctypes.c_double*40)()
 
elllist = []
Tlist = []
Tboundarylist = []
Tw_olist = []
Tw_ilist = []
Tw_i2list = []
qlist = []

h_clist = []
h_klist = []
thermalclist = []
h_k2list = []
h_efflist = []


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
    
    #Def q, h_c, h_k, k/delt(x), h_k', Tboundary, Tw,o, Tw,i1, Tw,i2     
    #He3 to boundary
    k_t = Xprop[27] # W/m-K; thermal conductivity He3; k_t = k_f
    h_c = k_t*Nu/d_m # W/m^2-K; heat transfer coefficient
    #boundary to Cu; using an approximation of h_k @ -> sigma_k
    h_k = sigma_k*Tstart**3 # W/m^2-K
    #thermal capacity
    thermalc = k_cu/deltax # [W/m^2-K]
    #Cu to He4; using an approximation of h_k2 @ -> sigma_k
    k_t2 = 0.003077  #need a HeCalc function: =HeClac(26,0,"SV","0","T",Tbath,1)
    k_t2 = PROP[2][26]
    h_k2 = sigma_k*Tbath**3 # W/m^2-K
    h_eff = (2/h_c + 1/h_k + 1/thermalc + 1/h_k2)**(-1)
    
    q = (Tstart - Tbath)/(2/h_c + 1/h_k + 1/thermalc + 1/h_k2) # W/m^2 
    
    Tboundary = -q/h_c + Tstart
    Tw_o =  -2*q/h_c + Tstart
    Tw_i = -q/h_k + Tw_o 
    Tw_i2 = -q/(k_cu/deltax) + Tw_i
    dT_f = h*pi*d_m*(Tw_i-Tstart)/mdot_m/C_p*step 
    dT_f = -q*pi*d_m/mdot_m/C_p*step
    
#REST PARAMETERS
    elllist.append(ell)
    Tlist.append(Tstart)
    Tboundarylist.append(Tboundary)
    Tw_olist.append(Tw_o)
    Tw_ilist.append(Tw_i)
    Tw_i2list.append(Tw_i2)
    qlist.append(q)
    
    h_clist.append(h_c)
    h_klist.append(h_k)
    thermalclist.append(thermalc)
    h_k2list.append(h_k2)
    h_efflist.append(h_eff)
    
    print("{:10.4f} {:10.4f} {:10.4e} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(ell,Tstart,Tboundary,Tw_o,Tw_i,Tw_i2,dT_f,q,Pr))
    
    ell = ell + step
    Tstart = Tstart + dT_f    
    

#DUAL PHASE: APPROXIMATED------------------------------------------------------

Tstart = saturation_temperature #K

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
    
    dQ_dot = hc*(pi*d_m)*(Tstart-Tboundary)*step # W
    
    dx = dQ_dot/(-(mdot/1000.)*(h_V-h_L))
    dQ_dot_over_dL = dQ_dot/step # W/m
    #print(ell,Tk.value,p0,x,rho_V,h_V,rho_L,h_L,mu_L,Re_0,Pr_L,Nu,k_L,hc,dQ_dot,dx,dQ_dot_over_dL)
    #further parameters for later calculations
    mu_V = XpropV[28] 


    xlist.append(x)
    elllist.append(ell)
    Tlist.append(Tstart)
    dQdLlist.append(dQ_dot_over_dL)
    Tboundarylist.append(Tboundary)
    Tw_olist.append(Tw_o)
    Tw_ilist.append(Tw_i)
    Tw_i2list.append(Tw_i2)
    qlist.append(q)
    
    h_clist.append(h_c)
    h_klist.append(h_k)
    thermalclist.append(thermalc)
    h_k2list.append(h_k2)
    h_efflist.append(h_eff)
    
    x = x + dx
    ell = ell + step


#LIQUID PHASE------------------------------------------------------------------

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
    
    #Def q, h_c, h_k, k/delt(x), h_k', Tboundary, Tw,o, Tw,i1, Tw,i2     
    #He3 to boundary
    k_t = Xprop[27] # W/m-K; thermal conductivity He3; k_t = k_f
    h_c = k_t*Nu/d_m # W/m^2-K; heat transfer coefficient
    #boundary to Cu; using an approximation of h_k @ -> sigma_k
    h_k = sigma_k*Tstart**3 # W/m^2-K
    #thermal capacity
    thermalc = k_cu/deltax # [W/m^2-K]
    #Cu to He4; using an approximation of h_k2 @ -> sigma_k
    k_t2 = 0.003077  #need a HeCalc function: =HeClac(26,0,"SV","0","T",Tbath,1)
    k_t2 = PROP[2][26]
    h_k2 = sigma_k*Tbath**3 # W/m^2-K
    h_eff = (2/h_c + 1/h_k + 1/thermalc + 1/h_k2)**(-1)
    
    q = (Tstart - Tbath)/(2/h_c + 1/h_k + 1/thermalc + 1/h_k2) # W/m^2 
    
    Tboundary = -q/h_c + Tstart
    Tw_o =  -2*q/h_c + Tstart
    Tw_i = -q/h_k + Tw_o 
    Tw_i2 = -q/(k_cu/deltax) + Tw_i
    #dT_f = h*pi*d_m*(Tboundary-Tstart)/mdot_m/C_p*step
    dT_f = -q*pi*d_m/mdot_m/C_p*step 

#REST PARAMETERS
    elllist.append(ell)
    Tlist.append(Tstart)
    Tboundarylist.append(Tboundary)
    Tw_olist.append(Tw_o)
    Tw_ilist.append(Tw_i)
    Tw_i2list.append(Tw_i2)
    qlist.append(q)
    
    h_clist.append(h_c)
    h_klist.append(h_k)
    thermalclist.append(thermalc)
    h_k2list.append(h_k2)
    h_efflist.append(h_eff)
    
    print("{:10.4f} {:10.4f} {:10.4e} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(ell,Tstart,Tboundary,Tw_o,Tw_i,Tw_i2,dT_f,q,Pr))
    
    ell = ell + step
    Tstart = Tstart + dT_f    


#Plot: T(l) plot: use the plot function

#z = np.polyfit(elllist, Tlist, 2)
#p = np.poly1d(z)
#plt.plot(elllist,p(elllist),"r--")
plt.figure(figsize=(10,5))
plt.rc('legend', fontsize=13)
plt.rc('font', size=13, family='serif')     #font for LaTeX
plt.plot(elllist,Tlist,label="T$_f$")
plt.plot(elllist,Tboundarylist,label="T$_{boundary}$")
plt.plot(elllist,Tw_olist,label="T$_{w,o}$")
plt.plot(elllist,Tw_ilist,label="T$_{w,i}$")
plt.plot(elllist,Tw_i2list,label="T$_{w,i}$")
plt.plot([elllist[0],elllist[-1]],[Tbath ,Tbath],"--",label='T$_{bath}$')
plt.plot([elllist[0],elllist[-1]],[saturation_temperature,saturation_temperature],"--",color="cyan",label="T$_{saturation}$")
#plt.title("$T=%.3f(\mathrm{K m}^{-2})L^2+(%.3f)(\mathrm{K m}^{-1})L+%.3f\mathrm{K}$"%(z[0],z[1],z[2]))
plt.xlabel('L [m]')     
plt.ylabel('T [K]')         
plt.legend(framealpha=1, frameon=True);
plt.show()
#f1.savefig("NAME",bbox_inches='tight')
    
    

plt.figure(figsize=(10,5))
plt.rc('legend', fontsize=13)
plt.rc('font', size=13, family='serif')     #font for LaTeX

plt.plot(elllist,h_clist,label="h$_c$")
plt.plot(elllist,h_klist,label="h$_k$")
plt.plot(elllist,h_k2list,label="h$_k'$")
#plt.plot(elllist,thermalclist,label="k$_{cu}/Delta(x)$")
plt.plot(elllist,h_efflist,label="h$_{eff}$")

# Display grid
plt.grid(True, which="both")
# Linear X axis, Logarithmic Y axis
plt.semilogy(elllist, h_klist )

plt.xlabel('L [m]')     
plt.ylabel('Conductance [W/m$^2$ K]')         
plt.legend(framealpha=1, frameon=True);
plt.show()
#f1.savefig("NAME",bbox_inches='tight')