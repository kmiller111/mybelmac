# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:47:56 2024

@author: mlm82
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian1D
x = np.linspace(-10,10,1000)

#blue wing
A, cnt, w = 0.20, -0.3, 0.25
bw_para = [A, cnt, w] 

#blue-core 
A, cnt, w = 0.75, -.075, .1875
bc_para = [A, cnt, w]

#red core: 
A, cnt, w = 0.10, 0.15, .4375
rc_para = [A,cnt, w]

#red wing
A, cnt, w = 0.0, 0.075, .21875
rw_para = [A, cnt, w]

bw = Gaussian1D(amplitude=bw_para[0], mean=bw_para[1], stddev=bw_para[2]) # blue wing
bc = Gaussian1D(amplitude=bc_para[0], mean=bc_para[1], stddev=bc_para[2])  #blue core
rc = Gaussian1D(amplitude=rc_para[0], mean=rc_para[1], stddev=rc_para[2])
rw = Gaussian1D(amplitude=rw_para[0], mean= rw_para[1], stddev=rw_para[2])

plt.plot(x,bw(x),color='skyblue')  
plt.plot(x,bc(x), 'b')  
plt.plot(x,rc(x), 'r')
plt.plot(x,rw(x), color= "orange")
total= bw + bc + rc +  rw
plt.plot(x,total(x),'black')
plt.plot(x,bw(x),'skyblue',label='Blue Wing')
plt.plot(x,bc(x),'b',label='Blue Core')
plt.plot(x,rw(x),'orange',label='Red Wing')
plt.plot(x,rc(x),'r',label='Red Core')
plt.plot(x,total(x),'black', label='Total Flux')
plt.xlabel('Velocity')
plt.ylabel('Flux')
plt.vlines(0,0,1, color='dimgray', linestyle='dashed')
plt.xlim([-1.5,1.5])
plt.legend()
plt.show()

#has the shpe i want, now to do the scale-- this is to match the accelerating outflows
#this run i divided by two for the amplitude and did not touch the widths--- this gave me
# a lower y range so maybe need to change the x better to fit

#so my scale needs to be changed the values from earlier matched the overall shape best

x = np.linspace(-1,1,5)# this was a no
x = np.linspace(-5,5,10)#pointy
x = np.linspace(-10,10,1000) #for now this is what we want, ask how to better fit the scale later or check notes
#blue wing
A, cnt, w = 0.20, -2, 2
bw_para = [A, cnt, w] 

#blue-core 
A, cnt, w = 0.75, -.5, 1.5
bc_para = [A, cnt, w]

#red core: 
A, cnt, w = 0.10, 1, 1.75
rc_para = [A,cnt, w]

#red wing
A, cnt, w = 0.0, 0.5, 1.5
rw_para = [A, cnt, w]

bw = Gaussian1D(amplitude=bw_para[0], mean=bw_para[1], stddev=bw_para[2]) # blue wing
bc = Gaussian1D(amplitude=bc_para[0], mean=bc_para[1], stddev=bc_para[2])  #blue core
rc = Gaussian1D(amplitude=rc_para[0], mean=rc_para[1], stddev=rc_para[2])
rw = Gaussian1D(amplitude=rw_para[0], mean= rw_para[1], stddev=rw_para[2])

plt.plot(x,bw(x),color='skyblue')  
plt.plot(x,bc(x), 'b')  
plt.plot(x,rc(x), 'r')
plt.plot(x,rw(x), color= "orange")
total= bw + bc + rc +  rw
plt.plot(x,total(x),'black')
plt.show()

data = np.array([x, total(x),bw(x),bc(x),rc(x),rw(x)]).T
plt.plot(data[:,0],data[:,2],color='skyblue') 
plt.plot(data[:,0],data[:,3],'b')
plt.plot(data[:,0],data[:,4],'r')
plt.plot(data[:,0],data[:,5],color='pink')
plt.plot(data[:,0],data[:,1],'k')
plt.show()

# create a list of column titles.
hdr = ['Velocity (km/s)','total','blue wing','blue core','red core','red wing']

header = '/t'.join(hdr)

header = '\t'.join(['Velocity (km/s)','total','blue wing','blue core','red core','red wing'])
 

np.savetxt('MyLineProfile.txt',data,header=header,delimiter='\t')

np.savetxt('MyLineProfile.csv',data)   
              
data=np.loadtxt('MyLineProfile.txt')
print('MyLineProfile.txt',data)