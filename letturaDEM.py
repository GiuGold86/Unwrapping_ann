from dataclasses import replace
from http.client import OK
from matplotlib import container, pyplot as plt
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from numpy import asarray, conjugate
from numpy import savetxt
from tqdm import tqdm 

areaG = 30  #grandezza area

numpytxt=np.loadtxt('DEM_ITALIA.txt')
# image_wrapped = np.ma.array(np.angle(np.exp(1j * numpytxt))) #crea l'immagine wrappata 
reZ = np.reshape(numpytxt, len(numpytxt) * len(numpytxt[0]))

with open("data_wu.txt", "w") as f:

    for i in tqdm(range(200)):
        x = np.random.randint(0, numpytxt.shape[0]-areaG-1)
        y = np.random.randint(0, numpytxt.shape[1]-areaG-1)

        
        wrapp = numpytxt[x:x+areaG, y:y+areaG].astype(np.int64)

        if np.sum(wrapp) == 0:
            continue

        unwrapp = np.ma.array(np.angle(np.exp(1j * wrapp)))

        fmt_str = " ".join(["{:.1f}" for i in range(areaG*areaG)])
        fmt_str += ", " + " ".join(["{:.10f}" for i in range(areaG*areaG)]) + "\n"

        f.write(fmt_str.format(*wrapp.flatten(), *unwrapp.flatten()))


# with open("data_wu.txt", "r") as f:
#     data = f.read().split("\n")


# wrapp, unwrapp = data[4].split(",")
# wrapp = np.fromstring(wrapp, dtype=float, sep=' ')
# unwrapp = np.fromstring(unwrapp, dtype=float, sep=' ')
# # print(wrapp.shape)

# plt.subplot(1, 2, 1)
# plt.imshow(wrapp.reshape(areaG, areaG))

# plt.subplot(1, 2, 2)
# plt.imshow(unwrapp.reshape(areaG, areaG))

# plt.show()

# data = np.array([i.split(",") for i in data[:-1]])
# wrapp = np.array([np.fromstring(i, dtype=float, sep=' ') for i in data[:,0]])
# unwrapp = np.array([np.fromstring(i, dtype=float, sep=' ') for i in data[:,1]])
# print(unwrapp)

""" tuo schifoso
seme = int(np.random.uniform(0, 200))
area = seme + 20
forsliceZ = reZ.reshape(len(numpytxt), len(numpytxt[0]))
sliceZ = forsliceZ[seme:area, seme:area]

ciclo = 10
contatore = 0
#while ciclo > contatore:
for i in range(10):
    seme = int(np.random.uniform(0, 200))
    area = seme + areaG
    forsliceZ = reZ.reshape(len(numpytxt), len(numpytxt[0]))
    sliceZ = forsliceZ[seme:area, seme:area]


    if len(sliceZ) == areaG and len(sliceZ[0]) == areaG:
        print("ok")
        np.savetxt('data' + str(contatore) + '.txt', (sliceZ), fmt='%i', delimiter=' ')
    else:
        print('erro')

    contatore = contatore +1
    print(contatore)

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
# np.savetxt('data' + str(contatore) + '.txt', (sliceZ), fmt='%i', delimiter=' ')
plt.imshow(inputZ.reshape(len(numpytxt), len(numpytxt[0])))
"""
#plt.imshow(numpytxt)
#plt.show()
#generata = np.loadtxt('data0.txt')
