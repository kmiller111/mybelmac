#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:23:43 2022
Last updated Jul 17 2024

Plot the broad emission line profile(s).

Parameters
----------
bel : string or list
    broad emission line names.
vbins : array
    velocity bin values.
lp : array
    luminosity for each velocity bin.
unit : string, optional
    x-axis unit for plotting. The default is 'velocity'.

Returns
-------
None.

@author: sara
"""

from roman import toRoman
import matplotlib.pyplot as plt

def profile (bel,vbins,lp,unit = 'vel'):

    plt.style.use('default') # makes the background white
    for l in range(0,len(bel)):
        
        line = bel[l].split(' ')
        wave = float(bel[l].split(' ')[2].split('A')[0])
        if 'H 1' not in bel[l] and 'Ly' not in bel[l]: 
            rg = toRoman(int(line[1]))
        else:
            if '1215' in bel[l] or '6562' in bel[l]: rg = r'$\alpha$'
            if '4861' in bel[l]: rg = r'$\beta$'
            if wave < 3640: line[0] = 'Ly' # Balmer break
            if wave > 6565: line[0] = 'Pa' # Paschen break
        line = line[0]+rg
        
        if 'vel' in unit: 
            plt.plot(vbins/1e3,lp[l],label=line+' model',linewidth=2) #
            plt.xlabel(r'Velocity ($\times10^3\,$km$\,$s$^{-1}$)',fontsize=13)
           # plt.xlim(-5000,5000)
        else: 
            plt.plot((1+vbins/3e5)*wave,lp[l],'r',label=line+' model')
            plt.xlabel(r'Wavelength ($\AA$)',fontsize=13)
        plt.ylabel(r'Luminosity (ergs\s)',fontsize=13)
        plt.legend(fontsize=12)
    plt.show()
    
    return