#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:23:43 2022
Last updated Jul 17 2024

Plot the 3D ensemble of velocity, cloud emission, or emission line fraction.

Parameters
----------
x : array
    cloud postion's x-coordinate
y : array
    cloud postion's y-coordinate
z : array
    cloud postion's z-coordinate
values : array
    cloud velocity, emission, or emission line fraction.
view : string, optional
    xy-axis viewing angle.  The default is 0 (observer).
save : string, optional
    save the 3D plot. The default is 'no'.

Returns
-------
None.

@author: sara
"""

import numpy as np
import matplotlib.pyplot as plt

def BLR3D (x,y,z,values,view=0,save='no'):
    plt.style.use('dark_background') # makes the background black
    ax = plt.axes(projection='3d')

    if 1e2 < max(values) < 1e5: 
        name,label = 'velocity','Velocity (km/s)'
        cmap = 'seismic'
        vmin = -max(np.abs(values))
        vmax = max(np.abs(values))
    if max(values) > 1e20:
        name,label = 'emission',r'H$\beta$ Line Luminosity (erg/s)'
        cmap = 'Blues_r'
        vmin = min(values)
        vmax = max(values)
        blr = ax.scatter(x,y,z,marker='o',s=5,c=values,alpha=1,cmap=cmap,vmin=vmin,vmax=vmax) 
    if 0 <= max(values) <= 1:
        name,label = 'elf','Emission Line Fraction'
        cmap = 'Greens_r'
        vmin = min(values)
        vmax = max(values)
        blr = ax.scatter(x,y,z,marker='.',s=1,c=values,alpha=1,cmap=cmap,vmin=vmin,vmax=vmax) 

    blr = ax.scatter(x,y,z,marker='.',s=1,c=values,alpha=1,cmap=cmap,vmin=vmin,vmax=vmax)
    
    # observer's veiw is (0, 0)
    ax.view_init(60, view)

    ax.axis('off')
    # remove background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
     # set axii ranges
    w,h = max(y),max(x)
    size = h-3
    if w > h: size = w-5
    ax.set_xlim([-size,size]) # along the LOS of observer
    ax.set_ylim([-size,size]) # mid-plane when inclined edge-on
    ax.set_zlim([-size,size]) # pole when inclined edge-on

    ax.set_aspect('equal') # x, y, and z axii are equal
    
    # color bar legend  
    bar = plt.colorbar(blr,pad=0.01,location='left')
    bar.set_label(label,fontsize=14)
      
    fig = plt.gcf()
    fig.set_size_inches(6, 6) # size of figure in inches
    # save the file, default is 'no'
    if save == 'yes': fig.savefig(name+'_field.pdf', dpi=300,bbox_inches='tight',pad_inches=.1)
    
    plt.show()
    
    return