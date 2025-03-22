#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:51:19 2024
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

import numpy as np
import pandas as pd
import time
from scipy.interpolate import RegularGridInterpolator
import os

st = time.time() # start timer
print('\n\033[03;36;m \t\tWelcome to BELPro\t\n\033[0;;m ')

# FREE PARAMETERS -------------------------------------------------------------
M = 8                               # SMBH mass (in sol mass)
Y = [15]                         # scaled BLR size
p = [0]                           # cloud density PL index
s = [0]                         # gas density PL index
sigma = [90]                     # agular disk width
opa = [90-sigma[0]/2]               # opening angle; 90-sigma[0]/2 for a disk
inclination = 90                    # inclination
cf = [0.3]                            # zonal covering fraction
no = [9]                      # gas density
sp = 1                              # ADI softening parameter
edge = 'sharp'                      # shape of BLR edge
field = 'RoSph'               # velocity field
# -----------------------------------------------------------------------------
# OPTIONAL FEATURES -----------------------------------------------------------
RP = None                        # Resolving Power, None for no instramental broadening
geo_plot = 'yes'                   # would you like to plot the 3D ensemble?
lp_plot = 'yes'                    # would you like to plot the line profile(s)?
save = 'single-epoch'             # path to save output 
# -----------------------------------------------------------------------------
# FIXED PARAMETERS ------------------------------------------------------------
L = 1e45                          # AGN luminosity 
Q = 1.67*10**55                   # total ionizing photon luminosity 
numclouds = 100_000
res = 100                         # km/s per bin 
ice = 'yes'                        # isotropic cloud emission
comp = 'no'                      # eletron scattering
bel = ['H 1 6562.80A','C 4 1550.77A & 1548.19A'] # 'H 1 1215.67A',,'H 1 4861.32A','Si 4 1393.75A & 1402.77A',,'Mg 2 2795.53A & 2802.71A'
pim = 'CloudyGrid_LowEdd.csv'
wl = [float(line.split(' ')[2].split('A')[0]) for line in bel]
# -----------------------------------------------------------------------------
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

if not os.path.exists(save): os.makedirs(save)
path = save+'/'

eofn, ce = '','anisotropic'
if ice == 'yes': eofn,ce = '_ice','isotropic'

CF = np.sum(cf)

# file name
if 'Disk' in field and 'Cone' in field: name = "belmac_CF"+str(CF)+"_"+str(edge)+"_LP_sp"+str(sp)+"_p"+str(p[0])+"_pc"+str(p[1])+"_i"+str(int(inclination))+"_y"+str(int(Y[0]))+"_yc"+str(int(Y[1]))+"_sig"+str(int(sigma[0]))+"_sigc"+str(int(sigma[1]))+"_o"+str(int(opa[0]))+"_oc"+str(int(opa[1]))+"_M"+str(M)+"_s"+str(s[0])+"_sc"+str(s[1])+"_"+field+'_n'+str((no[0]))+'_nc'+str((no[1]))+eofn
elif 'Disk' in  field or 'Sph' in field: name = "anePresentation!belmac_CF"+str(CF)+"_"+str(edge)+"_LP_sp"+str(sp)+"_p"+str(p[0])+"_i"+str(int(inclination))+"_y"+str(int(Y[0]))+"_sig"+str(int(sigma[0]))+"_o"+str(int(opa[0]))+"_M"+str(M)+"_s"+str(s[0])+"_"+field+'_n'+str((no[0]))+eofn
elif 'Cone' in  field: name = "belmac_CF"+str(CF)+"_"+str(edge)+"_LP_sp"+str(sp)+"_pc"+str(p[0])+"_i"+str(int(inclination))+"_yc"+str(int(Y[0]))+"_sigc"+str(int(sigma[0]))+"_o"+str(int(opa[0]))+"_M"+str(M)+"_sc"+str(s[0])+"_"+field+'_nc'+str((no[0]))+eofn

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
            Rco = 2*Rd*np.sqrt(cf*(1-Y**(-p-1))/(np.sin(width)*np.sin(o)*1e7*(p+1)*np.log(Y)))
        else:    
            Rco = 2*Rd*np.sqrt(cf*np.sin(width)*np.sin(o)*(p-2*s/3-1)*(1-Y**(-p-1))/(1e7)/(p+1)/(1-Y**(2*s/3+1-p)))
        
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
    
    def radial(self,M,ER,R,r,theta,phi,U,Nc,n,s,v0='kep'):
        
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
                print('\t',mo);print('\t{:2.2%} of clouds are deaccelerating outwards.'.format(len(np.where(acl==-1)[0])/len(r)))
        elif min(r/R) <= 1: # clouds launched from outer radius
            if np.any(v<=0): # failed infall
                order = r.argsort(axis=0)
                d = r[order]
                Rtrn = d[np.where(np.sign(v[order])==-1)[0][-1]]
                v = np.sqrt(np.abs(v))
                v = np.asanyarray(np.hstack((v[np.where(r>=Rtrn)][::2],-1*v[np.where(r>=Rtrn)][1::2])),dtype=float)
                theta = np.asanyarray(np.hstack((theta[np.where(r>=Rtrn)][::2],theta[np.where(r>=Rtrn)][1::2])),dtype=float)
                phi = np.asanyarray(np.hstack((phi[np.where(r>=Rtrn)][::2],phi[np.where(r>=Rtrn)][1::2])),dtype=float)
                mo = 'Failed infall reaches {:.2f} e16 cm'.format(Rtrn/1e16)
                print('\t',mo);print('\t{:2.2%} of clouds are in the failed wind'.format(len(v)/len(r)))
            else: # inflow
                v = -np.sqrt(np.abs(v))
                mo = '{:2.2%} of clouds are accelerating inwards'.format(len(np.where(acl==-1)[0])/len(r))
                print('\t',mo);print('\t{:2.2%} of clouds are deaccelerating inwards.'.format(len(np.where(acl==1)[0])/len(r)))
        # line of sight (LOS) velocity
        vel = -v/1e5*(np.cos(theta)*np.sin(rad)+np.sin(theta)*np.cos(phi)*np.cos(rad))
        return vel,Rtrn/R,v0+0.5,mo
    
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
    
    def turbulence(self,M,R,r,v0='kep'):
        if v0 == 'kep': v0 = np.sqrt(G*M/R)/1e5 # default
        elif v0 == 'esc': v0 = np.sqrt(2*G*M/R)/1e5
        else: v0 = np.sqrt(v0*2*G*M/R)/1e5 # othewise, user gives a fraction of the escape velocity
        mo = 'Random v(Rin) = {:.2f} km/s'.format(v0)
        print('\t',mo)
        np.random.seed() # randomly generate number for position angles
        v = r**(-0.5)*np.random.normal(loc=0,scale=0.25*v0,size=len(r)) # random turbulence

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
    
    X = zones[g]
    print('\033[4mZone: %s \033[0;;m'%X)
    print('\t',Y[g],p[g],s[g],angwidth[g],op[g],no[g])
    i,j,k,r,theta,phi,frac = Ensemble.zone(blr,Y[g],p[g],s[g],angwidth[g],op[g],rad,sp,numclouds)
    Rc,n,u,Ri,Rd = Ensemble.cloud(blr,blr.L,blr.M,blr.Q,sp,cf[g],Y[g],p[g],s[g],angwidth[g],op[g],10**no[g],r)
    
    # parameter warnings
    if np.any(n < 10**min(logn))==True: print('\u001b[47m \tWARNING: gas density too low\033[0;;m')
    if np.any(n > 10**max(logn))==True: print('\u001b[47m \tWARNING: gas density too high\033[0;;m')
    if np.any(u > 10)==True: print('\u001b[47m \tWARNING: U exceeding 10\033[0;;m')
    if np.any(u < 10**min(logU))==True: print('\u001b[47m \tWARNING: U low\033[0;;m')
    
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
                   
    vtot,motion = vrad+vrot+vran,np.append(mo,motion) # add the velocity vectors together
    v,radius,elf = np.append(v,vtot),np.append(radius,r),np.append(elf,frac)
    x,y,z = np.append(x,i),np.append(y,j),np.append(z,k)
    U,hden,Nc = np.append(U,u),np.append(hden,n),np.append(Nc,n*Rc)
    
    if np.any(Nc < 10**min(logN))==True: print('\u001b[47m \tWARNING: column density low\033[0;;m')
    if np.any(Nc > 10**max(logN))==True: print('\u001b[47m \tWARNING: column density high\033[0;;m')
    
    # opacity due to Compton scatter
    if 'y' in comp: ne = 6
    else: ne = -35
    tv = (np.sqrt(y**2+Y[g]**2)-x)*Ri*thom*10**ne
    # ------------------------------------------------------------------------

start, stop = -np.ceil(max(abs(v))/abs(vf))*abs(vf),np.ceil(max(abs(v))/abs(vf))*abs(vf) #normalize to max vLOS
vbins = np.linspace(start,stop,int(2*round((abs(start))/res))+1,endpoint=True)
lp = np.zeros((len(bel),len(vbins))) # initialize line profile bins
sort = np.digitize(v, vbins,right=False) # sort the velocities into their appropriate bins
    
print('\nmax LOS vel %.2f'% (max(abs(v))), 
      'min LOS vel %.2f km/s'% (min(v)),
      'step size %.2f km/s'% (2*(abs(start))/len(vbins)))

Ftot = 10**(intrp_flx((np.log10(Nc),np.log10(U),np.log10(hden)))) 
Finwd = 10**(intrp_in((np.log10(Nc),np.log10(U),np.log10(hden)))) 

if ice == 'yes' or ice == 'y': flux = Ftot.T
else: flux = elf*Finwd.T + (1-elf)*(Ftot.T-Finwd.T)

# cloud's luminosity is flux * cross sectional area
cloud_lum = flux*np.pi*(Nc/hden)**2*(g+1)*1e7/len(radius)*np.exp(-tv)

# sort the cloud luminosities into the correct velocity bins
for l in range(0,len(bel)):
    for b in range(min(sort),max(sort)):
        index = np.where(sort == b)
        lp[l,b] = np.sum(cloud_lum[l,index])

    if RP != None: 
        import InstBrd
        print('Instrament Broadening Width {:.2f}'.format(wl[l]/RP))
        _,lp[l] = InstBrd.InstBrd(vbins*wl[l]/c+wl[l], lp[l], wl[l]/RP) # apply instramental broadening

model = np.column_stack((vbins,lp.T))

flux = np.sum(lp)*((vbins[2]-vbins[1]))
print('Line Flux: %.3fe45'%(flux/1e45))

if len(zones) == 1: Y.append('-'), p.append('-'), s.append('-'), sigma.append('-'),opa.append('-'),cf.append('-')
header = ['Black hole mass = 10^'+str(M)+' Msol'+ 
        '\nAGN L =\t'+str(L/1e45)+'10^45 erg/s'+ 
        '\nQ =\t'+str(Q/1e55)+'10^55 photons/s'+
        '\nY (zone 1) =\t'+str(Y[0])+
        '\nY (zone 2) =\t'+str(Y[1])+
        '\ndistribution index (zone 1) =\t'+str(p[0])+
        '\ndistribution index (zone 2) =\t'+str(p[1])+
        '\nMotion (zone 1) =\t'+motion[0]+
        '\nMotion (zone 2) =\t'+motion[1]+
        '\nmax velocity =\t{:.2f} km/s'.format(max(v))+
        '\ncloud emission =\t'+str(ce)+
        '\nresolution =\t'+str(res)+
        '\ngas density index (zone 1) =\t'+str(s[0])+
        '\ngas density index (zone 2) =\t'+str(s[1])+
        #'\nfraction of matter bounded clouds =\t'+str(mb/100)+
        '\nnumber of clouds =\t'+str(numclouds)+ 
        '\ninclination =\t'+str(inclination)+ 		
        '\nangle width (zone 1) =\t'+str(sigma[0])+ 
        '\nangle width (zone 2) =\t'+str(sigma[1])+
        '\nhalf-opening angle (zone 1) =\t'+str(opa[0])+
        '\nhalf-opening angle (zone 2) =\t'+str(opa[1])+
        '\nillumination =\t'+str(sp)+  		 
        '\nedge =\t'+str(edge)+ 			
        '\nCovering Fraction (zone 1)=\t'+str(cf[0])+ 
        '\nCovering Fraction (zone 2)=\t'+str(cf[1])+ 
         '\n---------------------------------------'+
         '\n\t\t\tEmission line luminosity in erg/s\nVelocity (km/s)\t'+'\t'.join(bel)]

np.savetxt(path+name+'.txt', model,header='\n'.join(header),delimiter=' ')
print('\nModel saved as',path+name)

if 'y' in geo_plot: 
    import BLR3D
    # observer's veiw is 0, looking down the x-axis
    BLR3D.BLR3D(x[::2], y[::2], z[::2], v[::2]) # plot 1/2 of the clouds for computing speed

if 'y' in lp_plot:  
    import line
    line.profile(bel,vbins,lp)  

print("\n\tRun time\n --- %.2f sec ---" % ((time.time() - st))) 

           
                    