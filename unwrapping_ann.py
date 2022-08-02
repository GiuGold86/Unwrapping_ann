###########
#
# Davide Ruzza
# Licenza: MIT
############

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

size = 30
"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
########### READ data ad normalize
print("start read")
with open("data_wu.txt", "r") as f:
    data = f.read().split("\n")
print("end read")

data = np.array([i.split(",") for i in data[:-1]])
wrapp = np.array([np.fromstring(i, dtype=float, sep=' ') for i in data[:,0]])
unwrapp = np.array([np.fromstring(i, dtype=float, sep=' ') for i in data[:,1]])

input  = (unwrapp - (-np.pi)) / (np.pi - (-np.pi)) #normalizzazione delle variabili
target = (wrapp -  (0))  / (3000 -  (0))  #normalizzazione delle variabili

###############à

hidden_neuron = 10
output_layer = len(wrapp[0])
input_layer = len(unwrapp[0])

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""

w_i_h = np.random.uniform(-0.5, 0.5, (hidden_neuron, input_layer))
w_h_o = np.random.uniform(-0.5, 0.5, (output_layer, hidden_neuron))

b_i_h = np.zeros((hidden_neuron, 1))
b_h_o = np.zeros((output_layer, 1))

learn_rate = 0.0001
nr_correct = 0
epochs = 100
# plt.figure(figsize=(7, 8))
for epoch in tqdm(range(epochs)):
    for img, l in zip(input, target):
        # img = img.T
        # l = l.T
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

        

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / input.shape[0]) * 100, 2)}%")
    nr_correct = 0

    plt.clf()    
    plt.subplot(231)
    plt.imshow(img.reshape(size, size))
    plt.gca().set_title('wrapp')
    plt.subplot(232)
    plt.imshow(l.reshape(size, size))
    plt.gca().set_title('unwrapp')
    plt.subplot(233)
    plt.imshow(o.reshape(size, size))
    plt.gca().set_title('unwrappcalc')
    plt.pause(0.01)

    
    

# Show results
# while True:
#     index = int(input("Enter a number (0 - 59999): "))
#     img = images[index]
#     plt.imshow(img.reshape(28, 28), cmap="Greys")

#     img.shape += (1,)
#     # Forward propagation input -> hidden
#     h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
#     h = 1 / (1 + np.exp(-h_pre))
#     # Forward propagation hidden -> output
#     o_pre = b_h_o + w_h_o @ h
#     o = 1 / (1 + np.exp(-o_pre))

#     plt.title(f"Subscribe if its a {o.argmax()} :)")
#     plt.show()

