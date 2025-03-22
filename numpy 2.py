# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:47:56 2024

@author: mlm82
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian1D
import pandas as pd

#pd.read_csv('8awoebelmac_CF0.3_sharp_LP_sp1_p2_i90_y15_sig90_o45_M8_s0_RoSph_n9.txt')

print('8awoebelmac_CF0.3_sharp_LP_sp1_p2_i90_y15_sig90_o45_M8_s0_RoSph_n9.txt')
data= np.loadtxt('8awoebelmac_CF0.3_sharp_LP_sp1_p2_i90_y15_sig90_o45_M8_s0_RoSph_n9.txt')

vel,flux = data[:,0],data[:,1]
flux_new = flux / np.trapz(flux, vel)
flux_norm = flux_new / np.sum(flux_new)

plt.plot(vel,flux,'k',label='The OG')
plt.plot(vel,flux_new,label='New Flux')
plt.plot(vel,flux_norm,label='Norm. Flux')

plt.xlim([min(vel),max(vel)])
#plt.ylim([-0.01,0.02]) # "zoom-in" on the Normalized flux
plt.xlabel('Velocity')
plt.ylabel('Flux')
plt.legend()
plt.show()

cdf = np.cumsum(flux_norm)

plt.plot(vel[::5],cdf[::5],'ko',markersize=1) # [::5] plots every fifth point
plt.xlabel('Velocity')
plt.ylabel('Normalized Cumulative Flux')
plt.show()

ipv10_idx = np.where(cdf==0.1) # find where the cdf = 10%
print(ipv10_idx)

nearest = min(cdf, key=lambda cdf:abs(cdf-0.1))
ipv10_idx = np.where(cdf==nearest)
print(ipv10_idx)

nearest = min(cdf, key=lambda cdf:abs(cdf-0.1))
ipv10_idx = np.where(cdf==nearest)[0][0] # [0] twice - one for matrix to array, then array to scalar
print('CDF at 10% =',cdf[ipv10_idx],'velocity (km/s) = ',vel[ipv10_idx])

# likewise for 90%
nearest = min(cdf, key=lambda cdf:abs(cdf-0.9))
ipv90_idx = np.where(cdf==nearest)[0][0]
print('CDF at 90% =', cdf[ipv90_idx],'velocity (km/s) = ',vel[ipv90_idx])

plt.plot(vel,cdf,'k')
plt.plot(vel[ipv10_idx],cdf[ipv10_idx],'bo')
plt.plot(vel[ipv90_idx],cdf[ipv90_idx],'go')
plt.xlabel('Velocity')
plt.ylabel('Normalized Cumulative Flux')
plt.show()

nearest = min(cdf, key=lambda cdf:abs(cdf-0.5))
cnt_idx = np.where(cdf==nearest)[0][0]
# for a symmetrical profile, this should be at nearly 0km/s
center = vel[cnt_idx]
print('CDF at 50% =', cdf[cnt_idx],'velocity (km/s) = ',center)

a,b = center-vel[ipv10_idx], vel[ipv90_idx]-center # define blue-skew as negative
IPV20 = a+b

print('The IPV(20%) is',IPV20,'km/s')

#asym
A = (a-b)/(a+b) # or (a-b)/IPV
# if the profile is symmetric, this should be near 0 [dimensionless]
print('Asymmetry = ',A)

#Kurtosis and Full-Width, Half-Max
hm = max(flux)/2  # 0.5 of the maximum flux
# set-up two halves of width ranges (similar to a and b)
w1,w2 = flux[:int(cnt_idx)],flux[int(len(flux)-cnt_idx):]
# find the nearest point, as done in 5. a)
near1,near2 = min(w1, key=lambda w1:abs(w1-hm)),min(w2, key=lambda w2:abs(w2-hm))
# calculate to 2 half-widths
hw1,hw2 =vel[int(np.where(w1==near1)[0][0])],vel[int(np.where(w2==near2)[0][0]+len(flux)-cnt_idx)]
# calculate the full-width
fwhm = abs(vel[int(np.where(w1==near1)[0][0])])+abs(vel[int(np.where(w2==near2)[0][0]+len(flux)-cnt_idx)])
print('FWHM = ',fwhm,' km/s')

plt.plot(vel,flux,'k')
plt.plot(hw1,near1,'bo')
plt.plot(hw2,near2,'go')
plt.hlines(hm,-5,5,color='gray',linestyle='dashed')
plt.xlabel('Velocity')
plt.ylabel('Flux')
plt.show()

K = 1.397*fwhm/IPV20
print('Kurtosis is', K)