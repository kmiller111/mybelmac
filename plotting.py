# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:31:38 2025

@author: mlm82
"""

import matplotlib.pyplot as plt
import numpy as np

#pd.read_csv('8awoebelmac_CF0.3_sharp_LP_sp1_p2_i90_y15_sig90_o45_M8_s0_RoSph_n9.txt')

#print('1anoebelmac_CF0.3_sharp_LP_sp1_p0_i90_y15_sig90_o45_M8_s0_RoSph_n9.txt')
data_1= np.loadtxt('single-epochKM-Belprobelpro_CF0.3_sharp_LP_sp1_p0_i90_y15_sig90_o45_M8_s-2_RoSph_n9Belprobelpro_CF0.3_sharp_LP_sp0_p0_i90_y15_sig90_o45_M8_s0_RoSph_n9.txt')
#data_2= np.loadtxt('1anoebelmac_CF0.3_sharp_LP_sp1_p0_i90_y15_sig90_o45_M8_s0_RoSph_n9')
#data_3= np.loadtxt('1anoebelmac_CF0.3_sharp_LP_sp1_p0_i90_y15_sig90_o45_M8_s0_RoSph_n9')
#data_4= np.loadtxt('1anoebelmac_CF0.3_sharp_LP_sp1_p0_i90_y15_sig90_o45_M8_s0_RoSph_n9')

vel,flux_H1,flux_C1 = data_1[:,0],data_1[:,1], data_1[:,2]

#H-ALPHA PLOTS
plt.plot(vel,flux_H1,flux_C1, 'k',label='Anisotropic w/o Scatter')
#plt.plot(vel,flux_new,label='New Flux')
#plt.plot(vel,flux_norm,label='Norm. Flux')

plt.xlim([min(vel),max(vel)])
#plt.ylim([-0.01,0.02]) # "zoom-in" on the Normalized flux
plt.xlabel(r'Velocity ($\times10^3\,$km$\,$s$^{-1}$)',fontsize=13)
plt.ylabel(r'Luminosity (ergs\s)',fontsize=13)
plt.legend(fontsize=12)
plt.legend()
plt.show()