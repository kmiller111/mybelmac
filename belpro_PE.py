
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:51:19 2024

@author: aasim & sara 

version 5:
    Sets-up a 3D ensemble of gas clouds arranged by user-specified geometry and 
    velocity field parameters.  Reads a photoionization model grid created with
    Cloudy to calculate the lines flux of any user-desired broad emission line.
    Optionally plots the 3D velocity field, emission line fraction, and line 
    luminosity.  The output is the line profile.  No echo mapping
    is done, nothing changes with time.
    
    Updates to version:
        - includes approximation for inter-cloud opacity
    
    Caveats:
        - parameter estimation a bit off
        - slow
"""

import numpy as np
import pandas as pd
import time
from scipy.interpolate import RegularGridInterpolator
import os
#import dynesty
import emcee
import matplotlib.pyplot as plt
from functools import partial
import corner
from multiprocessing import Pool

st = time.time() # start timer
print('\n\033[03;36;m \t\tWelcome to BELPro\t\n\033[0;;m ')

# FITTING PARAMETERS -------------------------------------------------------------
M_range = [6.3,9]
i_range = [0,60]    # inclination
Y1_range = [5,30]
Y2_range = [5,30]
p_range = [0,3]
s_range = [-3,0] 
w_range = [10,90]    # width
o_range = [10,80]   # opening angle
c_range = [0.1,0.5]   # covering
n1_range = [8.5,12] # log gas density
n2_range = [8.5,11] # log gas density
field = 'KDisk'                 # velocity field, separate zones with underscore
# -----------------------------------------------------------------------------
# OPTIONAL FEATURES -----------------------------------------------------------
Rgem = 4861.32/(3e5/(186.5/2.355))
Rljt = 4861.32/(3e5/(695.6/2.355))
SR = Rgem
geo_plot = 'no'                   # would you like to plot the 3D ensemble?
lp_plot = 'no'                    # would you like to plot the line profile(s)?
save = 'single-epoch'             # path to save output 
# -----------------------------------------------------------------------------
# FIXED PARAMETERS ------------------------------------------------------------
L = 1e45                    # AGN luminosity 
Q = 1.67*10**55                        # total ionizing photon luminosity 
DL = None               # luminosity distance (Mpc to cm)
numclouds = 50_000
ice = 'no'                        # isotropic cloud emission
bel = ['H 1 4861.32A'] #  'H 1 1215.67A','H 1 6562.80A',,'Mg 2 2795.53A & 2802.71A',,'C 4 1550.77A & 1548.19A','Si 4 1393.75A & 1402.77A'
pim = 'CloudyGrid_LowEdd.csv'
data =  np.column_stack((np.loadtxt('fakedata.txt')[1:,0], np.loadtxt('fakedata.txt')[1:,10]))
wl = [float(line.split(' ')[2].split('A')[0]) for line in bel]
# -----------------------------------------------------------------------------
# CONSTANTS -------------------------------------------------------------------
D = 1                               # cloud depth scaling
res = 100
c=3E10                              # speed of light in cm/s
rec_co = 2.6*10**(-13)              # cm^3/s recombination coefficient
eff_co = 1.1*10**(-13)              # cm^3/s effective recombination coefficient
planck = 6.6261*10**(-27)           # cm^2g/s Planck's constant
nu_Ha = 4.57E14                     # 1/s
nu_Ly = 2.5E15                      # Hz Lyman-alpha frequency
G = 6.6741E-8                       # cm^3 /g /s^2
mp = 1.67E-24                       # g 
mH = 1.67E-24                       # mass of hydrogen g
Msol = 2E33                         # Solar mass in g
thom = 6.65E-25                     # cm^2 Thomson x-section
E_Ha = planck*nu_Ha

def BELPro_model(x):
    # FREE PARAMETERS -------------------------------------------------------------
    M = x[0]                        # SMBH mass (in sol mass)
    inclination = x[1]               # inclination
    Y = [x[2]]                        # scaled BLR size
    p = [x[3]]                         # cloud density PL index
    s = [x[4]]                        # gas density PL index
    sigma = [x[5]]                    # agular disk width
    opa = [90-sigma[0]/2]             # opening angle
    cf = [x[6]]                         # covering fraction
    no = [x[7]]                      # gas density
    sp = 1                          # ADI softening parameter
    edge = 'sharp'                  # shape of BLR edge
    # -----------------------------------------------------------------------------
    inbel = ['Inwd '+line for line in bel]
    cols = np.hstack((['log U','log hden','log Ncol','co'],bel,inbel))
    df = pd.read_csv('pims/'+pim,usecols=cols)
    
    # read the photoionization grid parameters abd fluxes for desired BELs
    logN = df['log Ncol'].to_numpy(dtype=float)
    logn = df['log hden'].to_numpy(dtype=float)
    logU = df['log U'].to_numpy(dtype=float)
    flx = df[bel].to_numpy(dtype=float)
    flx_in = df[inbel].to_numpy(dtype=float) # Inward flux
    co = df['co'].to_numpy(dtype=float) # absorption coefficent
    # make the cube of coordinates
    s1,s2,s3 = np.unique(logN),np.unique(logU),np.unique(logn)
    grid = flx.reshape(len(s1),len(s2),len(s3),len(bel))
    grid_in = flx_in.reshape(len(s1),len(s2),len(s3),len(bel))
    grid_co = co.reshape(len(s1),len(s2),len(s3))
    
    intrp_flx = RegularGridInterpolator((s1,s2,s3), grid,bounds_error=False,fill_value=-35)
    intrp_in = RegularGridInterpolator((s1,s2,s3), grid_in,bounds_error=False,fill_value=-35)
    intrp_co = RegularGridInterpolator((s1,s2,s3), grid_co,bounds_error=False,fill_value=0.5)
    
    # if not os.path.exists(save): os.makedirs(save)
    # path = save+'/'
    
    # eofn, ce = '','anisotropic'
    # if ice == 'yes': eofn,ce = '_ice','isotropic'
    
    # # file name
    # if 'Disk' in field and 'Cone' in field: name = "belmac_CF"+str(CF)+"_"+str(edge)+"_LP_sp"+str(sp)+"_p"+str(p[0])+"_pc"+str(p[1])+"_i"+str(int(inclination))+"_y"+str(int(Y[0]))+"_yc"+str(int(Y[1]))+"_sig"+str(int(sigma[0]))+"_sigc"+str(int(sigma[1]))+"_o"+str(int(opa[0]))+"_oc"+str(int(opa[1]))+"_M"+str(M)+"_s"+str(s[0])+"_sc"+str(s[1])+"_"+field+'_n'+str((no[0]))+'_nc'+str((no[1]))+eofn
    # elif 'Disk' in  field or 'Sph' in field: name = "belmac_CF"+str(CF)+"_"+str(edge)+"_LP_sp"+str(sp)+"_p"+str(p[0])+"_i"+str(int(inclination))+"_y"+str(int(Y[0]))+"_sig"+str(int(sigma[0]))+"_o"+str(int(opa[0]))+"_M"+str(M)+"_s"+str(s[0])+"_"+field+'_n'+str((no[0]))+eofn
    # elif 'Cone' in  field: name = "belmac_CF"+str(CF)+"_"+str(edge)+"_LP_sp"+str(sp)+"_pc"+str(p[0])+"_i"+str(int(inclination))+"_yc"+str(int(Y[0]))+"_sigc"+str(int(sigma[0]))+"_o"+str(int(opa[0]))+"_M"+str(M)+"_sc"+str(s[0])+"_"+field+'_nc'+str((no[0]))+eofn
    
    class BLR():
        def __init__(self,L,M,Q):
            self.L = L
            self.M = M
            self.Q = Q
            ER = L/(4*np.pi*c*M*G*mp/thom)
            self.ER = ER
                 
    class Ensemble(BLR):
        def __init__(self,L,M,ER,Q,cf,Y,p,s,width,o,rad,no):
            super(Ensemble, self).__init__(L,M,Q,ER)
            self.cf = cf
            self.Y = Y
            self.p = p
            self.s = s
            self.width = width
            self.o = o
            self.rad = rad
            self.no = no
        
        def zone(self,Y,p,s,width,o,rad,sp,numclouds):
            # all the ensembles are uniformly distributed with angle phi
            np.random.seed() # re-seed 
            numclouds = (numclouds//2)*2 # always round to even number of clouds
            phi=2*np.pi*np.random.uniform(size=numclouds)
            
            np.random.seed() # re-seed
            # cloud theta coordinates
            w = np.cos(np.pi/2-2*width)
            if edge =='sharp':
                theta1 = np.arccos(w*np.random.uniform(0,1,size=int(numclouds/2)))-(np.pi/2-o)+width
                np.random.seed() # re-seed
                theta2 = np.arccos(w*np.random.uniform(-1,0,size=int(numclouds/2)))+np.pi/2-o-width
            if edge =='fuzzy': 
                theta1 = np.random.normal(o+width/2,width,size=int(numclouds/2))
                np.random.seed() # re-seed
                theta2 = np.random.normal(np.pi-o-width/2,width,size=int(numclouds/2))
            theta = np.hstack((theta1,theta2))
            
            if sp!=1:
                rdth=0.29*((L*(sp+((1-sp)*(1/3*(np.abs(np.cos(theta))*(1+(2*(np.abs(np.cos(theta)))))))))/1E46)**0.5)/(0.4*((L/1E45)**(1/2))) 
                Y=Y+rdth-max(rdth) 
            
            np.random.seed()
            rnum=np.random.random(size=numclouds)
            ymin = 1
            if (p==-1):
                r=ymin*np.exp(rnum*np.log(Y/ymin))
            else:
                r=((rnum*((Y)**(p+1)))+(1-rnum)*(ymin**(p+1)))**(1/(p+1))
        
            # coord rotation is about y axis CW
            x = r*np.sin(theta)*np.cos(phi)*np.cos(rad)+(r*np.cos(theta)*np.sin(rad))
            y = r*np.sin(theta)*np.sin(phi)
            z = -r*np.sin(theta)*np.cos(phi)*np.sin(rad)+r*np.cos(theta)*np.cos(rad)
        
            # Anisotropic cloud emission factor
            elf=np.arccos(np.cos(theta)*np.sin(rad)+np.sin(theta)*np.cos(phi)*np.cos(rad))/np.pi
        
            return x,y,z,r,theta,phi,elf 
        
        def cloud(self,L,M,Q,sp,cf,Y,p,s,width,o,no,r):
              
            Tsub,T=1500,1500    # dust sublimation temp in K
            Rd=0.4*((L/1E45)**0.5)*((T/Tsub)**2.6)*3.0857E18  # dust sublimation radius in cm
            Ri = Rd/Y # inner radius in cm
            
            # cloud size
            if p-2*s/3 == 1:
                Rco = 2*Rd*np.sqrt(cf*(1-Y**(-p-1))/(np.sin(width)*np.sin(o)*(1e7)*(p+1)*np.log(Y)))
            else:    
                Rco = 2*Rd*np.sqrt(cf*np.sin(width)*np.sin(o)*(p-2*s/3-1)*(1-Y**(-p-1))/(1e7)/(p+1)/(1-Y**(2*s/3+1-p)))
            
            # gas properties
            Rc,n = Rco*(r/Y)**(-s/3),no*(r/Y)**s
            U = Q/4/np.pi/c/no/Rd**2*((r/Y)**-(s+2)) # U(r) scaled w/ inner U 
            if sp !=1: U = min(U)*((r/Y)**-(s+2))*(sp+((1-sp)*((1/3)*(np.abs(np.cos(theta))*(1+(2*(np.abs(np.cos(theta)))))))))
            
            self.n = n
            self.U = U
            self.Nc = Rc*n
            self.Ri = Ri
            
            return Rc,n,U,Ri,Rd
        
    class Velocity(Ensemble): ## VELOCITY FIELD in km/s
        def __init__(self, L, M, ER, Q, cf, Y, p, s, width, o, rad, field):
            super(Velocity, self).__init__(L, M, ER, Q, cf, Y, p, s, width, o, rad)
            #self.field = field
        
        def radial(self,M,ER,R,r,theta,phi,U,Nc,n,s,v0):
            
            if v0 == 'kep': v0 = 0.5 # inner radius velocity is Keplerian (default)
            elif v0 == 'esc': v0 = 1 # inner radius velocity is Escape Velocity
            # othewise, user gives a fraction of the escape velocity
            FM = (intrp_co((np.log10(Nc),np.log10(U),np.log10(n))))/thom/Nc # force muliplier
            acl = np.sign(ER*FM-1)
            v = 2*G*M/r*(ER*FM/(2*s/3+1)*((r/R)**(2*s/3+1)-1)+1+r/R*(v0-1)) # NOT in km/s
    
            Rtrn = 0
            if min(r/R) >= 1: # clouds launched from inner radius
                if np.any(v<=0): # failed wind
                    order = r.argsort(axis=0)
                    d = r[order]
                    Rtrn = d[np.where(np.sign(v[order])==-1)[0][1]]
                    v = np.sqrt(np.abs(v))
                    v = np.asanyarray(np.hstack((v[np.where(r<=Rtrn)][::2],-1*v[np.where(r<=Rtrn)][1::2])),dtype=float)
                    theta = np.asanyarray(np.hstack((theta[np.where(r<=Rtrn)][::2],theta[np.where(r<=Rtrn)][1::2])),dtype=float)
                    phi = np.asanyarray(np.hstack((phi[np.where(r<=Rtrn)][::2],phi[np.where(r<=Rtrn)][1::2])),dtype=float)
                    mo = 'Failed inner wind reaches {:.2f} e16 cm'.format(Rtrn/1e16)
                    #print('\t',mo);print('\t{:2.2%} of clouds are in the failed wind'.format(len(v)/len(r)))
                else: # outflow
                    v = np.sqrt(np.abs(v))
                    mo = '{:2.2%} of clouds are accelerating outwards'.format(len(np.where(acl==1)[0])/len(r))
                    #print('\t',mo);print('\t{:2.2%} of clouds are decelerating outwards.'.format(len(np.where(acl==-1)[0])/len(r)))
            elif min(r/R) <= 1: # clouds launched from outer radius
                if np.any(v<=0): # failed wind
                    order = r.argsort(axis=0)
                    d = r[order]
                    Rtrn = d[np.where(np.sign(v[order])==-1)[0][1]]
                    v = np.sqrt(np.abs(v))
                    v = np.asanyarray(np.hstack((v[np.where(r>=Rtrn)][::2],-1*v[np.where(r>=Rtrn)][1::2])),dtype=float)
                    theta = np.asanyarray(np.hstack((theta[np.where(r>=Rtrn)][::2],theta[np.where(r>=Rtrn)][1::2])),dtype=float)
                    phi = np.asanyarray(np.hstack((phi[np.where(r>=Rtrn)][::2],phi[np.where(r>=Rtrn)][1::2])),dtype=float)
                    mo = 'Failed outer wind reaches {:.2f} e16 cm'.format(Rtrn/1e16)
                    #print('\t',mo);print('\t{:2.2%} of clouds are in the failed wind'.format(len(v)/len(r)))
                else: # inflow
                    v = -np.sqrt(np.abs(v))
                    mo = '{:2.2%} of clouds are accelerating inwards'.format(len(np.where(acl==-1)[0])/len(r))
                    #print('\t',mo);print('\t{:2.2%} of clouds are deaccelerating inwards.'.format(len(np.where(acl==1)[0])/len(r)))
            # line of sight (LOS) velocity
            vel = -v/1e5*(np.cos(theta)*np.sin(rad)+np.sin(theta)*np.cos(phi)*np.cos(rad))
            return vel,Rtrn/Ri,v0+0.5,mo
    
        def keplerian(self,M,R,r,theta,field,v0='kep'):
            if v0 == 'kep': v0 = np.sqrt(G*M/R)/1e5 # default
            elif v0 == 'esc': v0 = np.sqrt(2*G*M/R)/1e5
            else: v0 = np.sqrt(v0*2*G*M/R)/1e5 # othewise, user gives a fraction of the escape velocity
            mo = 'Circ. v(Rin) = {:.2f} km/s'.format(v0)
            #print('\t',mo)
            if 'R' in field and 'K' in field: # vrot is the tangential velocity vector, vrad is perpendicular to vrot.  
                v = v0/r/np.sin(theta)
            elif 'K' in field and not 'R' in field:
                v = v0*r**-0.5
               
            return v, v0/(np.sqrt(G*M/R)/1e5),mo
        
        def turbulence(self,M,R,r,v0='kep'):
            if v0 == 'kep': v0 = np.sqrt(G*M/R)/1e5 # default
            elif v0 == 'esc': v0 = np.sqrt(2*G*M/R)/1e5
            else: v0 = np.sqrt(v0*2*G*M/R)/1e5 # othewise, user gives a fraction of the escape velocity
            mo = 'Random v(Rin) = {:.2f} km/s'.format(v0)
            #print('\t',mo)
            np.random.seed() # randomly generate number for position angles
            v = v0*r**(-0.5)*np.random.normal(loc=0,scale=0.25,size=len(r)) # random turbulence
    
            return v,v0/(np.sqrt(G*M/R)/1e5),mo
            
    blr = BLR(L, (10**M)*Msol, Q)
    # ------------------------------------------------------------------------
    angwidth = np.asanyarray(sigma,dtype=float)*np.pi/180/2
    op = np.asanyarray(opa,dtype=float)*np.pi/180
    rad = np.pi/2-inclination*np.pi/180
    
    zones = field.split('_')
    v,radius,x,y,z = [],[],[],[],[]
    U,hden,Nc,elf = [],[],[],[]
    
    for g in range(0,len(zones)):
        X = zones[g]

        i,j,k,r,theta,phi,frac = Ensemble.zone(blr,Y[g],p[g],s[g],angwidth[g],op[g],rad,sp,numclouds)
        Rc,n,u,Ri,Rd = Ensemble.cloud(blr,blr.L,blr.M,blr.Q,sp,cf[g],Y[g],p[g],s[g],angwidth[g],op[g],10**no[g],r)
        
        # parameter warnings
        #if np.any(n < 1e8)==True: print('\u001b[47m \tWARNING: gas density too low\033[0;;m')
        #if np.any(u > 10)==True: print('\u001b[47m \tWARNING: U exceeding 10\033[0;;m')
        
        vrad,vrot,vran = np.zeros(len(r)),np.zeros(len(r)),np.zeros(len(r))
        if 'K' in X: 
            vrot,vf,mo = Velocity.keplerian(blr,blr.M,Ri,r,theta,X)
            vrot = vrot*(np.cos(theta)*np.sin(phi)*np.sin(rad)-np.sin(theta)*np.sin(phi)*np.cos(rad)) 
                  
        if 'T' in X: vran,vf,mo = Velocity.turbulence(blr,blr.M,Ri,r)
        
        if 'R' in X: 
            if 'Ro' in X: 
                vrad,Rt,vf,mo = Velocity.radial(blr,blr.M,blr.ER,Ri,r*Ri,theta,phi,u,Rc*n,n,s[g],'esc')
            elif 'Ri' in X: vrad,Rt,vf,mo = Velocity.radial(blr,blr.M,blr.ER,Rd,r*Ri,theta,phi,u,Rc*n,n,s[g],'kep')
            else: vrad,Rt,vf,mo = Velocity.radial(blr,blr.M,blr.ER,Ri,r*Ri,theta,phi,u,Rc*n,n,s[g])
            if Rt != 0: 
                if 'Ri' in X: cut = np.where(r>=Rt)
                else: cut = np.where(r<=Rt)
                i = np.asanyarray(np.hstack((i[cut][::2],i[cut][1::2])),dtype=float)
                j = np.asanyarray(np.hstack((j[cut][::2],j[cut][1::2])),dtype=float)
                k = np.asanyarray(np.hstack((k[cut][::2],k[cut][1::2])),dtype=float)
                vrot = np.asanyarray(np.hstack((vrot[cut][::2],vrot[cut][1::2])),dtype=float)
                vran = np.asanyarray(np.hstack((vran[cut][::2],vran[cut][1::2])),dtype=float)
                u,n = np.asanyarray(np.hstack((u[cut][::2],u[cut][1::2]))),np.asanyarray(np.hstack((n[cut][::2],n[cut][1::2])))
                Rc,frac = np.asanyarray(np.hstack((Rc[cut][::2],Rc[cut][1::2]))),np.asanyarray(np.hstack((frac[cut][::2],frac[cut][1::2])))
                r = np.asanyarray(np.hstack((r[cut][::2],r[cut][1::2])),dtype=float)
            
        vtot = vrad[:len(r)]+vrot+vran # add the velocity vectors together
        v,radius,elf = np.append(v,vtot),np.append(radius,r),np.append(elf,frac)
        x,y,z = np.append(x,i),np.append(y,j),np.append(z,k)
        U,hden,Nc = np.append(U,u),np.append(hden,n),np.append(Nc,n*Rc)
        
        # opacity
        tv = (np.sqrt(y**2+Y[g]**2)-x)*Ri*thom*10**6
        # ------------------------------------------------------------------------
    # create velocity bins equal to the data
    vbins = data[:,0]
    lp = np.zeros((len(bel),len(vbins))) # initialize line profile bins
    sort = np.digitize(v, vbins,right=False) # sort the velocities into their appropriate bins
    
    Ftot = 10**(intrp_flx((np.log10(Nc),np.log10(U),np.log10(hden)))) 
    Finwd = 10**(intrp_in((np.log10(Nc),np.log10(U),np.log10(hden)))) 
    
    if ice == 'yes' or ice == 'y': flux = Ftot
    else: flux = elf*Finwd.T + (1-elf)*(Ftot.T-Finwd.T)
    
    # cloud's luminosity is flux * cross sectional area
    cloud_lum = flux*np.pi*(Nc/hden)**2*(g+1)*1e7/len(radius)*np.exp(-tv)
    
    # sort the cloud luminosities into the correct velocity bins
    for l in range(0,len(bel)):
        for b in range(min(sort),max(sort)):
            index = np.where(sort == b)
            lp[l,b] = np.sum(cloud_lum[l,index])
    
        if SR != None: 
            import InstBrd
            #print('Instrament Broadening Width {:.2f}'.format(wl[l]/RP))
            _,lp[l] = InstBrd.InstBrd(vbins*wl[l]/c+wl[l], lp[l], SR) # apply instramental broadening
        if DL != None: modelout = np.column_stack((vbins,lp.T*1e15/(4*np.pi*DL**2)))
        else: modelout = np.column_stack((vbins,lp.T))
    return modelout

# parameter space
prior_range = np.array([M_range,i_range,Y1_range,p_range,s_range,w_range,c_range,n1_range]) #, o_range,,w2_range,n2_range

def likelihood(b, data):
    try:
        if prior_range[0][0] <= b[0] <=  prior_range[0][1] and  prior_range[1][0] <= b[1] <=  prior_range[1][1] and prior_range[2][0] <= b[2] <=  prior_range[2][1] and prior_range[3][0] <= b[3] <=  prior_range[3][1] and prior_range[4][0] <= b[4] <=  prior_range[4][1] and prior_range[5][0] <= b[5] <=  prior_range[5][1] and prior_range[6][0] <= b[6] <=  prior_range[6][1] and prior_range[7][0] <= b[7] <=  prior_range[7][1]:# and prior_range[8][0] <= b[8] <=  prior_range[8][1] and prior_range[9][0] <= b[9] <=  prior_range[9][1] and prior_range[10][0] <= b[10] <=  prior_range[10][1] and prior_range[11][0] <= b[11] <=  prior_range[11][1] and prior_range[12][0] <= b[12] <=  prior_range[12][1] and prior_range[13][0] <= b[13] <=  prior_range[13][1] and prior_range[14][0] <= b[14] <=  prior_range[14][1]:# and prior_range[15][0] <= b[15] <=  prior_range[15][1]:
            model = BELPro_model(b)[:,1]
            d,m = data[:,1],model
            lnL = -np.sum((m**2-2*d*m)/0.1)/2 # chi^2
            return lnL
        else:
            return -np.inf # low likelihood to all points outside the prior range
    except: print('BELPRo Failure')
    
def prior_transform(u, prior_array):
     """Transforms the uniform random variable `u ~ Unif[0., 1.)`
     to the parameter of interest. Note: Needed for dynesty sampler."""
     b = np.array(u)
     b[0] = (prior_array[0][1]-prior_array[0][0]) * u[0] + prior_array[0][0]
     b[1] = (prior_array[1][1]-prior_array[1][0]) * u[1] + prior_array[1][0]
     b[2] = (prior_array[2][1]-prior_array[2][0]) * u[2] + prior_array[2][0]
     b[3] = (prior_array[3][1]-prior_array[3][0]) * u[3] + prior_array[3][0]
     b[4] = (prior_array[4][1]-prior_array[4][0]) * u[4] + prior_array[4][0]
     b[5] = (prior_array[5][1]-prior_array[5][0]) * u[5] + prior_array[5][0]
     b[6] = (prior_array[6][1]-prior_array[6][0]) * u[6] + prior_array[6][0]
     b[7] = (prior_array[7][1]-prior_array[7][0]) * u[7] + prior_array[7][0]
     # b[8] = (prior_array[8][1]-prior_array[8][0]) * u[8] + prior_array[8][0]
     # b[9] = (prior_array[9][1]-prior_array[9][0]) * u[9] + prior_array[9][0]
     # b[10] = (prior_array[10][1]-prior_array[10][0]) * u[10] + prior_array[10][0]
     # b[11] = (prior_array[11][1]-prior_array[11][0]) * u[11] + prior_array[11][0]
     # b[12] = (prior_array[12][1]-prior_array[12][0]) * u[12] + prior_array[12][0]
     # b[13] = (prior_array[13][1]-prior_array[13][0]) * u[13] + prior_array[13][0]
     # b[14] = (prior_array[14][1]-prior_array[14][0]) * u[14] + prior_array[14][0]
     #b[15] = (prior_array[15][1]-prior_array[15][0]) * u[15] + prior_array[15][0]
     return b

def get_initial_guess(prior_array, walkers):
    """Returns the initial guess/starting points for walkers using priors. Used for emcee sampler. Note: The initial guesses are uniformly sampled from the prior range."""
    guess = [np.random.uniform(low = prior_array[i][0], high = prior_array[i][1], size = walkers) for i in range(len(prior_array))]
    return np.array(guess)

iterations = 150
thin = 1
discard = 20
walkers = 200
pos = get_initial_guess(prior_range, walkers)
pos = pos.T
print("Starting points")
print(*pos , sep = "\n")
with Pool() as pool:
    # run MC
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, partial(likelihood,data=data), pool=pool
    )
    
    print(f"Iterations = {iterations}, walkers = {walkers}, thinning = {thin}, discard = {int(iterations*discard / 100)}")

    sampler.run_mcmc(pos, iterations, progress=True)
samples = sampler.get_chain(discard=int(iterations*discard/100), thin = thin) #chains, walkers, ndim
print(samples.shape)
flat_samples = sampler.get_chain(discard=int(iterations*discard/100), thin = thin, flat=True)
lnL_save = sampler.flatlnprobability
samples_ut = sampler.get_chain()
flat_samples_ut = sampler.get_chain(flat=True)

# cut off # as burn-in
fig = corner.corner(flat_samples, color="black", labels=[r'$M_\bullet$',r'$i$','$Y_{1}$','$p_1$','$s_1$',r"$\sigma_1$",r'$C_{f1}$',r'$n_{o1}$'], #,r'$\mathcal{O}$'  
        smooth=None,smooth1d =None, linewidth = 1.0,  plot_datapoints=True, plot_density=False, 
        no_fill_contours=True,contours=True, levels=[0.9],label_kwargs={'fontsize':16}, 
        contour_kwargs={"linewidths":1.0},hist_kwargs={"linewidth":1.0, "density": True})
fig.show()
print("\n\tRun time\n --- %.2f hrs ---" % ((time.time() - st)/3600)) 

           
                    