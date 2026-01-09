import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
# Add project root (parent of test/) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neural_network import NeuralNetwork

# ----------------------------
# 1. Load MNIST Test Set
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ----------------------------
# 2. Initialize Neural Network
# ----------------------------
input_size = 28*28
hidden_size = 16
output_size = 10
nn_model = NeuralNetwork(input_size, hidden_size, output_size)

# ----------------------------
# 3. Load Saved Weights
# ----------------------------
data = np.load("trained_model.npz")
nn_model._w1 = data["W1"]
nn_model._b1 = data["b1"]
nn_model._w2 = data["W2"]
nn_model._b2 = data["b2"]
nn_model._w3 = data["W3"]
nn_model._b3 = data["b3"]
print("Model weights loaded from trained_model.npz")

# ----------------------------
# 4. Evaluate Accuracy
# ----------------------------
correct = 0
total = 0

for images, labels in testloader:
    for i in range(len(images)):
        x = images[i].numpy().reshape(28*28,1)
        y_true = labels[i].item()
        
        output = nn_model.forward(x)
        pred = np.argmax(output)
        
        if pred == y_true:
            correct += 1
        total += 1

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# ----------------------------
# 5. Visualize Some Predictions
# ----------------------------
num_examples = 5
examples_shown = 0

for images, labels in testloader:
    for i in range(len(images)):
        if examples_shown >= num_examples:
            break
        
        #Get the image from data and reshape to vector (784, 1)
        x = images[i].numpy().reshape(28*28,1)
        output = nn_model.forward(x) #Prediction
        pred = np.argmax(output)
        
        #plotting
        plt.imshow(images[i].squeeze(), cmap='gray') #Plotting the image
        plt.title(f"True: {labels[i].item()}, Predicted: {pred}")
        plt.show()
        
        examples_shown += 1
    if examples_shown >= num_examples:
        break
