import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Add project root (parent of test/) to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.neural_network import NeuralNetwork
from src.utils import one_hot


# ----------------------------
# 1. Load MNIST Training Set
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # scale to [-1,1]
])

trainset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# ----------------------------
# 2. Initialize Neural Network
# ----------------------------
input_size = 28*28
hidden_size = 16
output_size = 10
learning_rate = 0.01
nn_model = NeuralNetwork(input_size, hidden_size, output_size, lr=learning_rate)

#Load saved weights
data = np.load("trained_model.npz")
nn_model._w1 = data["W1"]
nn_model._b1 = data["b1"]
nn_model._w2 = data["W2"]
nn_model._b2 = data["b2"]
nn_model._w3 = data["W3"]
nn_model._b3 = data["b3"]


# ----------------------------
# 3. Training Loop
# ----------------------------
num_epochs = 10
loss_history = []

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in trainloader:
        for i in range(len(images)):
            # Flatten image
            x = images[i].numpy().reshape(28*28,1)
            # One-hot encode label
            y = one_hot(labels[i].item()).reshape(10,1)
            
            # Forward pass
            output = nn_model.forward(x)
            
            # Compute loss (MSE)
            loss = np.mean((output - y)**2)
            total_loss += loss
            
            # Backward pass
            nn_model.backward(x, y)
    
    avg_loss = total_loss / len(trainset)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ----------------------------
# 4. Plot Loss
# ----------------------------
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss over Epochs")
plt.show()

# ----------------------------
# 5. Save Model
# ----------------------------
np.savez("trained_model.npz",
         W1=nn_model._w1, b1=nn_model._b1,
         W2=nn_model._w2, b2=nn_model._b2,
         W3=nn_model._w3, b3=nn_model._b3)
print("Model weights saved to trained_model.npz")
