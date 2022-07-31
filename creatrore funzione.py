import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d 
from numpy import exp,arange
from matplotlib import pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show


def z_func(x,y):
 return (np.sin(np.sqrt(x**2 + y**2)))*6
 
x= np.arange(-10,10,0.1)
y= np.arange(-10,10,0.1)

X,Y = np.meshgrid(x,y)
Z = z_func(X, Y)


fig= plt.figure()
ax= fig.add_subplot(111, projection= '3d')
surf=ax.plot_surface(X,Y,Z,cmap='afmhot',linewidth=0,antialiased='True',rstride=3,cstride=3)
ax.contourf(X, Y, Z,100, zdir='z', offset=-1.5,cmap='afmhot')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-3, 3])
fig.colorbar(surf)
plt.show()