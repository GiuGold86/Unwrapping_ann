
from cgi import print_environ_usage
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from tkinter import image_names
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

################################################################################### GENERA DATI DA FUNZIONE


############### Genera l'immagine wrappata e unwrappata da una funzione
resolution = 0.1
minrange = -10
maxrange = 10
def z_func(x,y):
 return  1-abs(x+y)-abs(y-x) # (x**2+y**2)**0.5    # piramide 1-abs(x+y)-abs(y-x)    #(np.sin(np.sqrt(x**2 + y**2)))*6  ##########Funzione
 
x = arange(minrange,maxrange,resolution)
y = arange(minrange,maxrange,resolution)
X,Y = meshgrid(x, y) # grid of point
Z = z_func(X, Y) # evaluation of the function on the grid

image_wrapped = np.ma.array(np.angle(np.exp(1j * Z))) #crea l'immagine wrappata 

############## Genere l'imput e output formato rete neurale

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


#################################################################################### RETE NEURALE


hidden_neuron = 10
output_layer = len(inputZ)
input_layer = len(outputW)

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""

input  = (outputW - (-np.pi)) / (np.pi - (-np.pi)) #normalizzazione delle variabili
target = (inputZ -  (-100))  / (100 -  (-100)) #/ np.linalg.norm(inputZ)     #normalizzazione delle variabili


w_i_h = np.random.uniform(-0.5, 0.5, (hidden_neuron, input_layer))
w_h_o = np.random.uniform(-0.5, 0.5, (output_layer, hidden_neuron))
b_i_h = np.zeros((hidden_neuron, 1))
b_h_o = np.zeros((output_layer, 1))

learn_rate = 0.1
nr_correct = 0
epochs = 500
cicli= 0
nr_correct = 0
soglia = 0.1
transfetforward = "gaussian" # "relu" "gaussian" "sigmoid"
transferback = "gaussian"
errore = 1000
while errore > soglia:
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ input
    
    if transfetforward  == "relu":
        out = np.ones(h_pre.shape) 
        out[(h_pre < 0)]=0
        h  =  out  
    
    if transfetforward  == "gaussian":
        h  =  np.exp(-h_pre**2)    
    
    if transfetforward == "sigmoid":
        h  =  1 / (1 + np.exp(-h_pre))
    

    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    
    if transferback == "relu":
        outo = np.ones(o_pre.shape) 
        outo[(o_pre < 0)]=0
        o  =  outo 
    
    if transferback == "gaussian":
        o  =  np.exp(-o_pre**2)    
    
    if transferback == "sigmoid":
        o  =  1 / (1 + np.exp(-o_pre))       
    
    plt.clf()    
    plt.subplot(231)
    plt.imshow(input.reshape(len(Z), len(Z)), )
    plt.gca().set_title('wrapp')
    plt.subplot(232)
    plt.imshow(target.reshape(len(Z), len(Z)), )
    plt.gca().set_title('unwrapp')
    plt.subplot(233)
    plt.imshow(o.reshape(len(Z), len(Z)), )
    plt.gca().set_title('unwrappcalc')
    plt.pause(0.01)
    
    # Cost / Error calculation
    e = 1 / len(o) * np.sum((o - target) ** 2, axis=0)
    nr_correct = np.mean(o - target)
    print(nr_correct)
    errore = nr_correct
    # Backpropagation output -> hidden (cost function derivative)
    delta_o = o - target
    w_h_o += -learn_rate * delta_o @ np.transpose(h)
    b_h_o += -learn_rate * delta_o
    # Backpropagation hidden -> input (activation function derivative)
    delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
    w_i_h += -learn_rate * delta_h @ np.transpose(input)
    b_i_h += -learn_rate * delta_h


    cicli = cicli + 1


while 1:
    fig , ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    surf2 = ax.plot_surface(X, Y, target.reshape(len(Z), len(Z)),rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf2, shrink=0.5, aspect=10)
       
    surf2 = ax.plot_surface(X, Y, o.reshape(len(Z), len(Z)),rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf2, shrink=0.5, aspect=10)

    plt.show()

    """
    fig1, ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax.set_box_aspect((100,100,100))

    surf1 = ax.plot_surface(X, Y, input.reshape(len(Z), len(Z)),rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf1, shrink=0.5, aspect=10)  

       
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    
    surf2 = ax.plot_surface(X, Y, target.reshape(len(Z), len(Z)),rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf2, shrink=0.5, aspect=10)
     
    
    surf2 = ax.plot_surface(X, Y, o.reshape(len(Z), len(Z)),rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf2, shrink=0.5, aspect=10)



    plt.show()


 
    
    
    
    print("finito")
    plt.clf()    
    plt.subplot(231)
    plt.imshow(input.reshape(len(Z), len(Z)), )
    plt.gca().set_title('wrapp')
    plt.subplot(232)
    plt.imshow(target.reshape(len(Z), len(Z)), )
    plt.gca().set_title('unwrapp')
    plt.subplot(233)
    plt.imshow(o.reshape(len(Z), len(Z)), )
    plt.gca().set_title('unwrappcalc')

"""