#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:42:04 2024
Last updated Jul 17 2024

Simulate the broad emission line profile of AGN.

Required Input
----------
L : float
    AGN bolometric luminosity
Q : float
    total ionizing photon luminosity
pim : csv data frame 
    photoionization model grids

Returns
-------
broad emission line profile(s) in a single .txt file.

@author: sara, sarosboro@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from scipy.interpolate import RegularGridInterpolator

st = time.time() # start run timer

# CONSTANTS -------------------------------------------------------------------
D = 1                               # cloud depth scaling
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
# -----------------------------------------------------------------------------
# FREE PARAMETERS ----------------------------------------------------------
M = 8                               # SMBH mass (in sol mass)
Y = [15]                         # scaled BLR size
p = [2]                           # cloud density PL index
s = [-2]                         # gas density PL index
sigma = [90]                     # agular disk width
opa = [90-sigma[0]/2]                       # opening angle; 90 - sigma/2 for a disk
inclination = 90                    # inclination
cf = [0.3]                       # zonal covering fraction
no = [9.5]                      # gas density
sp = 1                              # ADI softening parameter
edge = 'sharp'                      # shape of BLR edge
field = 'KSph'               # velocity field
# -----------------------------------------------------------------------------
# FIXED PARAMETERS ------------------------------------------------------------
R = None                # Resolving Power, None for no instramental broadening
L = 1e45                   # AGN luminosity 
Q = 1.67e55              # total ionizing photon luminosity 
Dl = None               # luminosity distance (Mpc to cm)
input_file = 'delta_short_double.txt' 
obsdata = None
pim = 'CloudyGrid_LowEdd.csv'
numclouds = 100_000
res = 100
ice = 'no'
comp = 'yes'              # eletron scattering
bel = ['H 1 6562.80A'] #   ,'H 1 4861.32A', 'Mg 2 2795.53A & 2802.71A', 'C 4 1550.77A & 1548.19A'
wl = [float(line.split(' ')[2].split('A')[0]) for line in bel]
# -----------------------------------------------------------------------------
geo_plot = 'no'
tds_plot = 'yes'
save = 'MyBLR'            # path to save output 

dat = np.loadtxt(input_file)
dat_time,dat_lum = dat[:,0],dat[:,1]

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

CF = np.sum(cf)

# file name
if 'Disk' in field and 'Cone' in field: name = "belmac_CF"+str(CF)+"_"+str(edge)+"_TDS_sp"+str(sp)+"_p"+str(p[0])+"_pc"+str(p[1])+"_i"+str(int(inclination))+"_y"+str(int(Y[0]))+"_yc"+str(int(Y[1]))+"_sig"+str(int(sigma[0]))+"_sigc"+str(int(sigma[1]))+"_M"+str(M)+"_s"+str(s[0])+"_sc"+str(s[1])+"_"+field+'_n'+str((no[0]))+'_nc'+str((no[1]))
elif 'Disk' in  field or 'Sph' in field: name = "belmac_CF"+str(CF)+"_"+str(edge)+"_TDS_sp"+str(sp)+"_p"+str(p[0])+"_i"+str(int(inclination))+"_y"+str(int(Y[0]))+"_sig"+str(int(sigma[0]))+"_M"+str(M)+"_s"+str(s[0])+"_"+field+'_n'+str((no[0]))
elif 'Cone' in  field: name = "belmac_CF"+str(CF)+"_"+str(edge)+"_TDS_sp"+str(sp)+"_pc"+str(p[0])+"_i"+str(int(inclination))+"_yc"+str(int(Y[0]))+"_sigc"+str(int(sigma[0]))+"_M"+str(M)+"_sc"+str(s[0])+"_"+field+'_nc'+str((no[0]))

if ice == 'yes': name = name+'_ice'

path = 'iterations/'+input_file.split('_')[0]+'/'

class BLR():
    def __init__(self,L,M,Q):
        self.L = L
        self.M = M
        self.Q = Q
        ER = L/(4*np.pi*c*M*G*mp/thom)
        self.ER = ER
        print('Edd. ratio: %.2f' % ER)
             
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
    
    def cloud(self,L,M,Q,cf,Y,p,s,width,o,no,r):
          
        Tsub,T=1500,1500    # dust sublimation temp in K
        Rd=0.4*((L/1E45)**0.5)*((T/Tsub)**2.6)*3.0857E18  # dust sublimation radius in cm
        Ri = Rd/Y # inner radius in cm
        
        # cloud size
        if p-2*s/3 == 1:
            Rco = 2*Rd*np.sqrt(cf*(1-Y**(-p-1))/(np.sin(width)*np.sin(o)*(1e7)*(p+1)*np.log(Y)))
        else:    
            Rco = 2*Rd*np.sqrt(cf*np.sin(width)*np.sin(o)*(p-2*s/3-1)*(1-Y**(-p-1))/(5e7)/(p+1)/(1-Y**(2*s/3+1-p)))
        
        # mass is conserved
        mc = np.pi*no*mH*(Rco**3)
        print('\tCloud mass: %.2f e25 g' % (mc/1E25))
        
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
    def __init__(self, L, M, ER, Q, Y, p, s, width, o, rad, field):
        super(Velocity, self).__init__(L, M, ER, Q, Y, p, s, width, o, rad)
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
                print('\t',mo);print('\t{:2.2%} of clouds are in the failed wind'.format(len(v)/len(r)))
            else: # outflow
                v = np.sqrt(np.abs(v))
                mo = '{:2.2%} of clouds are accelerating outwards'.format(len(np.where(acl==1)[0])/len(r))
                print('\t',mo);print('\t{:2.2%} of clouds are decelerating outwards.'.format(len(np.where(acl==-1)[0])/len(r)))
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
                print('\t',mo);print('\t{:2.2%} of clouds are in the failed wind'.format(len(v)/len(r)))
            else: # inflow
                v = -np.sqrt(np.abs(v))
                mo = '{:2.2%} of clouds are accelerating inwards'.format(len(np.where(acl==-1)[0])/len(r))
                print('\t',mo);print('\t{:2.2%} of clouds are deaccelerating inwards.'.format(len(np.where(acl==1)[0])/len(r)))
        # line of sight (LOS) velocity
        vel = -v/1e5*(np.cos(theta)*np.sin(rad)+np.sin(theta)*np.cos(phi)*np.cos(rad))
        return vel,Rtrn/Ri,v0+0.5,mo
    
    def keplerian(self,M,R,r,theta,field,v0='kep'):
        if v0 == 'kep': v0 = np.sqrt(G*M/R)/1e5 # default
        elif v0 == 'esc': v0 = np.sqrt(2*G*M/R)/1e5
        else: v0 = np.sqrt(v0*2*G*M/R)/1e5 # othewise, user gives a fraction of the escape velocity
        mo = 'Circ. v(Rin) = {:.2f} km/s'.format(v0)
        print('\t',mo)
        if 'R' in field and 'K' in field: # vrot is the tangential velocity vector, vrad is perpendicular to vrot.  
            v = v0/r/np.sin(theta)
        elif 'K' in field and not 'R' in field:
            v = v0*r**-0.5
           
        return v, v0/(np.sqrt(G*M/R)/1e5),mo
    
    def turbulence(self,M,R,r,width,o,v0='kep'):
        if v0 == 'kep': v0 = np.sqrt(G*M/R)/1e5 # default
        elif v0 == 'esc': v0 = np.sqrt(2*G*M/R)/1e5
        else: v0 = np.sqrt(v0*2*G*M/R)/1e5 # othewise, user gives a fraction of the escape velocity
        
        if o+width==np.pi/2: H = np.arctan(width)
        else: H = np.arctan(width/2)
        print('\t Scale heigth',H)
        np.random.seed() # randomly generate number for position angles
        v = v0*r**(-0.5)*np.random.normal(loc=0,scale=2*H,size=len(r)) # random turbulence
        mo = 'Random v(Rin) = {:.2f} km/s'.format(np.std(v))
        print('\t',mo)

        return v,v0/(np.sqrt(G*M/R)/1e5),mo
        
blr = BLR(L, (10**M)*Msol, Q)

# ------------------------------------------------------------------------
angwidth = np.asanyarray(sigma,dtype=float)*np.pi/180/2
op = np.asanyarray(opa,dtype=float)*np.pi/180
rad = np.pi/2-inclination*np.pi/180

zones,motion = field.split('_'),['-']

v,radius,x,y,z = [],[],[],[],[]
U,hden,Nc,elf = [],[],[],[]
for g in range(0,len(zones)):
    
    X = zones[g] # X must be a string not in a list 
    print('\033[4mZone: %s \033[0;;m'%X)
        
    i,j,k,r,theta,phi,frac = Ensemble.zone(blr,Y[g],p[g],s[g],angwidth[g],op[g],rad,sp,numclouds)
    Rc,n,u,Ri,Rd = Ensemble.cloud(blr,blr.L,blr.M,blr.Q,cf[g],Y[g],p[g],s[g],angwidth[g],op[g],10**no[g],r)
        
    if np.any(Rc*n > 10**25)==True: 
        print('\033[0;33;mWARNING:  log Nc > 25 cm-2\033[0;;m')
        print('\tRemoving',len(np.where(Rc*n > 10**25)[0]),'clouds from %.3f to %.3f (xRi)'% (min(r[np.where(Rc*n > 10**25)[0]]),max(r[np.where(Rc*n > 10**25)[0]])))
        
    if np.any(n < 10**7.5)==True: print('\033[0;33;m WARNING:  n_{H} \leq 10^{7.5}\033[0;;m')
    
    vrad,vrot,vran = np.zeros(len(r)),np.zeros(len(r)),np.zeros(len(r))
    if 'K' in X: 
        vrot,vf,mo = Velocity.keplerian(blr,blr.M,Ri,r,theta,X,'kep')
        vrot = vrot*(np.cos(theta)*np.sin(phi)*np.sin(rad)-np.sin(theta)*np.sin(phi)*np.cos(rad)) 
               
    if 'T' in X: vran,vf,mo = Velocity.turbulence(blr,blr.M,Ri,r,angwidth[g],opa[g],'kep')
    
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
                   
    vtot,motion = vrad+vrot+vran,np.append(mo,motion) # add the velocity vectors together
    v,radius,elf = np.append(v,vtot),np.append(radius,r),np.append(elf,frac)
    x,y,z = np.append(x,i),np.append(y,j),np.append(z,k)
    U,hden,Nc = np.append(U,u),np.append(hden,n),np.append(Nc,n*Rc)
    v0 = np.mean(np.abs(vtot))

    # opacity due to Compton scatter
    if 'y' in comp: ne = 6
    else: ne = -35
    tv = (np.sqrt(y**2+Y[g]**2)-x)*Ri*thom*10**ne 
# ------------------------------------------------------------------------
if geo_plot == 'yes': 
    import BLR3D
    # plot every other cloud for speed
    BLR3D.BLR3D(x[::2], y[::2], z[::2], v[::2])

lct_Ri,lct_Rd=Ri/c,Rd/c # light crossing time 
print('Light-crossing time of R_in = %s days' %(lct_Ri/3600/24)); print()

if obsdata == None:  
# create velocity binning array with a bin centered on zero.
    start, stop = -np.ceil(max(abs(v))/abs(vf))*abs(vf),np.ceil(max(abs(v))/abs(vf))*abs(vf)
    vbins = np.linspace(start,stop,int(2*round((abs(start))/res))+1,endpoint=True)

    time_agn = dat_time/lct_Ri # normalize input time file to light crossing time of BLR
    timeinterval_obs=2*Y[0]/350 # for regular arbitrary models
else: 
    
    vbins = (obsdata[1:,0]/wl[0]-1)*3e5  #convert wavelength to velocity
    
    time_agn = dat_time#(dat_time-min(dat_time))*3600*24/lct_Ri 
    obs_time_max=max(time_agn)-min(time_agn)
    timeinterval_obs=obs_time_max/98   # for input LC
    
sort = np.digitize(v, vbins,right=False) # sort the velocities into their appropriate bins

print('\nmax LOS vel %.2f'% (max(abs(v))), 
      'min LOS vel %.2f km/s'% (min(v)))
print('Center bin (km/s):', vbins[int((len(vbins)/2))])
print('Number of v bins:', len(vbins),'vel res %.2f km/s' % (vbins[1]-vbins[0]))

lum_agn = dat_lum/(sum(dat_lum)/len(dat_lum)) # normalize input file lum to average AGN luminosity

# calculate time lag based on isodelay surface wrt observer
rad_time = radius*lct_Ri/3600/24+min(dat_time)
tl = radius*(1-(x/radius))#*lct_Ri #already normalized to Ri/c aka lct_Ri

print('Time interval %.3f'%timeinterval_obs,len(dat_time))
t=min(time_agn)

obs_time=[]
for i in range(0,350):
    obs_time = np.append(obs_time,t)
    t=t+timeinterval_obs

#obs_time=time_agn
print('obs times', min(obs_time), max(obs_time),len(obs_time))
#obs_time = data[0,1:]

tb,lwr,Uave = [],[],[]
count,ce,ds,Fmax = 0,0,0,0 # matter bounded clouds
modelout = np.append(0,vbins)*np.ones((len(bel),len(vbins)+1))
# ------------------------------ TIME LOOP ------------------------------------
for i in range(0,len(obs_time)):

    RLF = np.asarray([obs_time[i]/(1-r) for r in x/radius]) # light front

    # get the indices of the radii where the light front has passed through
    on = np.nonzero(radius<=RLF)[0]
    source_time = 0
    lum, Utime = np.ones(len(radius))*lum_agn[0], U*lum_agn[0] # lum of AGN at "quiescent state"
    
    if len(on)!=0: 
        source_time = np.asarray([obs_time[i]-tl[int(h)] for h in on])
        lum[on] = np.interp(source_time,time_agn,lum_agn) # lum of AGN that the cloud is seeing at the observer time
        Ut = lum[on]*U[on]
        Utime[on] = Ut
    Uave = np.append(Uave,np.mean(Utime))

    # get the cloud fluxes at this time step
    Ftot = 10**(intrp_flx((np.log10(Nc),np.log10(Utime),np.log10(hden))))
    Finwd = 10**(intrp_in((np.log10(Nc),np.log10(Utime),np.log10(hden))))

    if 'n' in ice: flux = np.multiply(elf,Finwd.T) + np.multiply((1-elf),(Ftot-Finwd).T)
    else: flux = Ftot

    cloud_lum = flux*np.pi*(Nc/hden)**2*(g+1)*5e7/len(radius)*np.exp(-tv) # cloud's flux * cross sectional area cm^2
    # calculate luminosity-weighted radius (LWR) of structure
    clbb = radius*cloud_lum
    LWR = sum(clbb)/sum(cloud_lum)
    lwr = np.append(lwr,LWR)
   
    lp = np.zeros((len(bel),len(vbins)+1)) # initialize line profile bins
    # sort the velocities into their appropriate bins
    sort = np.digitize(v, vbins,right=False) 
    for l in range(0,len(bel)):
        for b in range(min(sort),max(sort)):
            #print(l,b)
            index = np.where(sort == b)
            lp[l,0] = obs_time[i]
            lp[l,b+1] = np.sum(cloud_lum[l,index]) 
        if R != None: 
            import InstBrd
            _,lp[l][1:] = InstBrd.InstBrd(vbins*wl[l]/c+wl[l], lp[l][1:], R) # apply instramental broadening
    
    if Dl == None: modelout = np.dstack((modelout,lp)) 
    else: modelout = np.dstack((modelout,lp*1e15/(4*np.pi*Dl**2))) # convert to observed flux
#  END TIME LOOP -------------------------------------------------------------- 

Uave = np.mean(Uave)
header = ' '.join(bel)+'\nVmax\t'+str(max(vbins))+'\nLWR\t'+str(np.mean(lwr))+'\nUmin,max\t'+str(min(Utime))+' '+str(max(Utime))+'\nUave\t'+str(Uave)
for e in range(0,len(bel)):
    path = save+'/'+bel[e]
    if not os.path.exists(path): os.makedirs(path)
    np.savetxt(path+'/'+name+'.dat',modelout[e,:,:],header=header)

print('LWR %.4f' % (np.mean(lwr)))
print('\nSaved as:',path+bel[e]+'/'+name);print()

if 'y' in tds_plot:
    import TDSpec
    TDSpec.tds(modelout,bel)

print("\nRun time\n--- %.2f mins ---" % ((time.time() - st)/60))

    