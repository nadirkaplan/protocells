#This code plots the potential and electric field present between the lipid membrane and solid substrate.

import matplotlib.pyplot as plt
import numpy as np

x1 = np.arange(0,101.e-10, 1.e-10) 
x2 = np.arange(1.e-8,201.e-10,1.e-10)

h = 1.e-8
l = 84.e-11

v1 = -((0.2 + np.exp(-h/l))*np.exp(-x1/l) + np.exp((x1-h)/l))
v2 = -(0.2*np.exp(-x2/l) + np.exp(-(x2+h)/l) + np.exp(-(x2-h)/l))
e1 = -(0.2*np.exp(-x1/l) + np.exp(-(x1+h)/l) - np.exp((x1-h)/l))
e2 = -(0.2*np.exp(-x2/l) + np.exp(-(x2+h)/l) + np.exp(-(x2-h)/l))


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6.4,9.6))
ax1.plot(x1,v1,linewidth = 4, color = '0')
ax1.plot(x2,v2,linewidth = 4, color = '0')
ax2.plot(x1,e1,linewidth = 4, color = '0')
ax2.plot(x2,e2,linewidth = 4, color = '0')

#tick parameters
ax1.tick_params(axis="x", direction = "in", length=10, width=2, labelsize=15, labelcolor = "white")
ax1.tick_params(axis="y", direction = "in", length=10, width=2, labelsize=15)#labelcolor = "white"
ax2.tick_params(axis="x", direction = "in", length=10, width=2, labelsize=15)
ax2.tick_params(axis="y", direction = "in", length=10, width=2, labelsize=15)

#tick labels
#ax.xaxis.set_ticks(np.arange(0.,1.e-8,3.e-8))
#ax.yaxis.set_ticks(np.arange(0,1.5,0.5))

#axis range

ax1.set_ylim([-1.2,0.1])
ax2.set_ylim([-1.2,1.2])
#ax.set_xlabel('')

plt.savefig('potential.eps', format='eps', dpi = 300)


