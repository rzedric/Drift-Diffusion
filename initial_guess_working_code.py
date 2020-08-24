# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:11:58 2020

Initial solver for Poisson's equation
Trial run
Uniform mesh
Uses QFL=0 for n and p type

This works!!!!!!

This is adapted from the fortran code for the initial state which DV presents in 
Computational Electronics, 2010.  

@author: Rob
"""

import math
import numpy as np
import init
import TDMA

# Mesh definition
L = 20   # problem length, in µm 
N = 10000  # problem divisions
h = (L/1E4)/N # mesh spacing in cm

# Material definition
epsilon = init.epsilon_e()  # this is epsilon/q, in units of V^-1 cm^-1
n_i = init.intrinsic_charge()
N_a = 1E15
N_d = 7E13
KBT_q = init.KBT_eV()  # units of eV, equal to V/q
frac = 0.2 # location of junction relative to problem width of 1
jn = int(frac*N)
doping = np.zeros(2*N).reshape(2,N)
doping[0,:jn]=N_a
doping[1,jn:]=N_d



# Debye length with intrinsic doping.  units of [cm]
# N_max = max([N_a,N_d])
L_D = math.sqrt(epsilon*KBT_q/n_i)


Phi = np.zeros(N)
PHI = np.zeros(N)
# ===================================================
# initial guess of the potential.  Units of [V]

Phi_B = KBT_q*math.log(N_d*N_a/n_i**2)   # Built-in voltage
x_p = math.sqrt(2*epsilon*Phi_B/N_a**2/(1/N_a+1/N_d))
x_n = math.sqrt(2*epsilon*Phi_B/N_d**2/(1/N_a+1/N_d))


for i in range(N):
    if doping[0,i]>doping[1,i]:
        PHI[i] = math.log(-doping[0,i]/n_i/2*(1-math.sqrt(1+4*n_i**2/doping[0,i]**2)))
    elif doping[0,i]<=doping[1,i]:
        PHI[i] = math.log(doping[1,i]/n_i/2*(1+math.sqrt(1+4*n_i**2/doping[1,i]**2)))


# end of initial guess
# ==================================================

# convergence parameter
err=1E-1

# Scaled parameters
#    Given in all caps
H = h/L_D # scaled mesh spacing
#PHI = Phi/KBT_q
N_A = N_a/n_i
N_D = N_d/n_i
DOPING=doping/n_i

# iteration loop
converged=False
while converged != True:
    
    # create coefficients to go in the TDM 
    a1 = np.ones(N-1)/H**2
    a2 = np.ones(N)
    a3 = np.ones(N-1)/H**2
    B = np.zeros(N)
        
    for i in range(N):
        a2[i]=-2/H**2-math.exp(PHI[i])-math.exp(-PHI[i])
    
    # create the "forcing function"
    for i in range(1,N-1):
        B[i] = math.exp(PHI[i])-math.exp(-PHI[i])-PHI[i]*(math.exp(PHI[i])+math.exp(-PHI[i]))+(DOPING[0,i]-DOPING[1,i]) 
    
    # keep the left BC pinned.  
    a2[0]=1
    a3[0]=0
    B[0] = PHI[0]  
    
    # keep the right BC pinned.
    a2[N-1]=1
    a1[N-2]=0    
    B[N-1] = PHI[N-1]    
    

    

    
    
    # Assumes matrix multiplication takes the form AX=B 
    new_PHI = TDMA.TDMA(a1, a2, a3, B)
    
    dPHI = new_PHI-PHI
    
    # check convergence
    '''
    Manhattan norm: np.linalg.norm(np.arange(4),ord=0)
    Euclidean norm: np.linalg.norm(np.arange(4),ord=1)
    Infinite norm: np.linalg.norm(np.arange(4),np.inf)
    '''
    L2 = np.linalg.norm(dPHI,2)
    if L2 < err:
        converged=True
    
    print(L2)
    
    PHI = new_PHI
    PHI[0]=0
    PHI[-1]=0
    
# Un-scale parameters
Phi = PHI*KBT_q

