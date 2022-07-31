from dataclasses import replace
from http.client import OK
from matplotlib import container, pyplot as plt
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import numpy as np
from matplotlib import pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib.ticker import LinearLocator
from numpy import asarray
from numpy import savetxt

numpytxt=np.loadtxt('Monti_picentini_ascii.txt')
image_wrapped = np.ma.array(np.angle(np.exp(1j * numpytxt))) #crea l'immagine wrappata 

reZ = np.reshape(numpytxt, len(numpytxt) * len(numpytxt[0]))


seme = int(np.random.uniform(0, 200))
area = seme + 100
forsliceZ = reZ.reshape(len(numpytxt), len(numpytxt[0]))
sliceZ = forsliceZ[seme:area, seme:area]

ciclo = 6
contatore = 0
while ciclo > contatore:
    
    seme = int(np.random.uniform(0, 200))
    area = seme + 100
    forsliceZ = reZ.reshape(len(numpytxt), len(numpytxt[0]))
    sliceZ = forsliceZ[seme:area, seme:area]


    if len(sliceZ) == 100 and len(sliceZ[0]) == 100:
        print("ok")
        savetxt('data' + str(contatore) + '.txt', int(sliceZ), delimiter=',')
    else:
        print('erro')

    contatore = contatore +1
    print(contatore)



"""


reW = np.reshape(image_wrapped, len(numpytxt) * len(numpytxt[0]))
allW = []
for x in reW:
    allW.append(x)
dataW = np.array(reW)
outputW = dataW[0:len(dataW)].reshape(len(dataW),1)

reZ = np.reshape(numpytxt, len(numpytxt) * len(numpytxt[0]))
allZ = []
for x in reZ:
    allZ.append(x)
dataZ = np.array(allZ)
inputZ = dataZ[0:len(dataZ)].reshape(len(dataZ),1)

plt.imshow(inputZ.reshape(len(numpytxt), len(numpytxt[0])))
"""
#plt.imshow(numpytxt)
#plt.show()
generata = np.loadtxt('data0.txt')
plt.imshow(generata.reshape(100, 100))
plt.show()