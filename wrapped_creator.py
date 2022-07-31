from tkinter import image_names
import numpy as np
from numpy import exp,arange
from matplotlib import pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib.ticker import LinearLocator

###################################################################################### Genera l'immagine wrappata e unwrappata da una funzione
resolution = 0.1
minrange = -10
maxrange = 10
def z_func(x,y):
 return (np.sin(np.sqrt(x**2 + y**2)))*6  ##########Funzione
 
x = arange(minrange,maxrange,resolution)
y = arange(minrange,maxrange,resolution)
X,Y = meshgrid(x, y) # grid of point
Z = z_func(X, Y) # evaluation of the function on the grid

image_wrapped = np.ma.array(np.angle(np.exp(1j * Z))) #crea l'immagine wrappata 

################################################################################ Genere l'imput e output formato rete neurale

reW = np.reshape(image_wrapped, len(image_wrapped)*len(image_wrapped))
allW = []
for x in reW:
    allW.append(x)
dataW = np.array(allW)
outputW = dataW[0:len(dataW)].reshape(len(dataW),1)  ######################## output rete neurale


reZ = np.reshape(Z, len(Z)*len(Z))
allZ = []
for x in reZ:
    allZ.append(x)
dataZ = np.array(allZ)
inputZ = dataZ[0:len(dataZ)].reshape(len(dataZ),1)  ####################### input rete neurale

########################################################################### Visualizzazione e stampa


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(X, Y, outputW.reshape(len(Z), len(Z)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-10, 10)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

"""
print(len(Z))
print("output")
print(outputW)
plt.imshow(inputZ.reshape(len(Z), len(Z)))
plt.imshow(outputW.reshape(len(Z), len(Z)))
plt.show()
"""

