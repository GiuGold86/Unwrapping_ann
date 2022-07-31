import numpy as np
import matplotlib.pyplot as plt

hidden_neuron = 4
output_layer = 6
input_layer = 2
"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
images = np.random.uniform(-0.5, 0.5, (input_layer, 1))
labels = np.random.uniform(-1, 1, (output_layer, 1))


w_i_h = np.random.uniform(-0.5, 0.5, (hidden_neuron, input_layer))
w_h_o = np.random.uniform(-0.5, 0.5, (output_layer, hidden_neuron))
b_i_h = np.zeros((hidden_neuron, 1))
b_h_o = np.zeros((output_layer, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 100

for epoch in range(epochs):
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ images
    h = 1 / (1 + np.exp(-h_pre))
    print(h)

    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))
    
    plt.clf()
    plt.subplot(121)
    plt.imshow(labels, cmap="Greys")
    plt.subplot(122)
    plt.imshow(o, cmap="Greys")
    plt.pause(0.0001)
    
    #plt.imshow(o)
       
    
    # Cost / Error calculation
    e = 1 / len(o) * np.sum((o - labels) ** 2, axis=0)
    nr_correct += int(np.argmax(o) == np.argmax(labels))

    # Backpropagation output -> hidden (cost function derivative)
    delta_o = o - labels
    w_h_o += -learn_rate * delta_o @ np.transpose(h)
    b_h_o += -learn_rate * delta_o
    # Backpropagation hidden -> input (activation function derivative)
    delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
    w_i_h += -learn_rate * delta_h @ np.transpose(images)
    b_i_h += -learn_rate * delta_h


# Show results
while True:
    img = images
    plt.imshow(img.reshape(int(pow(input_layer,1/2)), int(pow(input_layer,1/2))), cmap="Greys")
    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(input_layer, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))
    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()



#originale
"""from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt


#w = weights, b = bias, i = input, h = hidden, o = output, l = label
#e.g. w_i_h = weights from input layer to hidden layer
images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3
for epoch in range(epochs):
    for img, l in zip(images, labels):
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
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()"""

