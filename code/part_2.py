import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import random
import struct
from array import array

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


batch_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 1000


# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_psnr(original, reconstructed):

    mse = torch.mean(
        (original - reconstructed) ** 2,
        dim=(1, 2, 3)
    )

    psnr = 10 * torch.log10(1.0 / mse)

    return psnr


# MNIST dataset is of size 28x28.

# Loading the dataset and preprocessing
# getting a 256 subset
full_train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True,
)

dataset_256 = Subset(full_train_dataset, list(range(256)))

train_loader = DataLoader(dataset_256, batch_size=64, shuffle=True)
test_loader = DataLoader(
    dataset_256, batch_size=64, shuffle=False
)  # Evaluates memorization on the same 256 images

# Convolutional autoencoder, made to fit latent Nx3x4x4
class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        # Encoder: 1x28x28 -> 128x12x12 -> 3x4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=0),  # -> 128x24x24
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 128x12x12

            nn.Conv2d(128,128, 3, padding=1), # -> 128x12x12 extra refinement step
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=0),  # -> 3x8x8
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 3x4x4
            nn.BatchNorm2d(3),
        )

        # Decoder:
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 128, 8, stride=3, padding=2),   # -> 128x13x13
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # extra refinement steps
            nn.Conv2d(128, 128, 3, padding=1),                   
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 32, 4, stride=2),             # -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),


            nn.Conv2d(32, 1, 3, padding=1),                       # smooth out artifacts
            nn.Sigmoid(),
        )

    def forward(self, x):

        lats = self.encoder(x)

        #gaussian noise injection for better robustness (BEFORE tanh)
        if self.training:
            lats = lats + torch.randn_like(lats) * 0.5

        #Tanh for stable STE gradients of [-1, 1]
        lats = torch.tanh(lats)
        # binarize
        binary = torch.where(lats >= 0, 1.0, -1.0)

        #STE trick: forward is exactly ±1, backward flows through tanh
        # forward: lats_ste simplifies to =binary.
        # during backprop, .detach() treats binary-lats separately. 
        # -> binary-lats = constant, => derivative = 1, => goes back up to tanh
        lats_ste = lats + (binary - lats).detach()


        recs = self.decoder(lats_ste)

        return recs
model = AE().to(device)


# Setting the loss function
loss_function = nn.MSELoss()


# Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


epochs = num_epochs

outputs = []
losses = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


for epoch in range(epochs):

    for images, _ in train_loader:
        images = images.to(device)  # no flatten
        reconstructed = model(images)
        loss = loss_function(reconstructed, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    outputs.append((epoch, images, reconstructed))
    print(f"Epoch {epoch + 1}/{epochs}, " f"Loss: {loss.item():.6f}")


plt.style.use("fivethirtyeight")

plt.figure(figsize=(8, 5))

plt.plot(losses, label="Loss")

plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.legend()

plt.show()


# Evaluate PSNR only after the final training epoch
model.eval()

total_psnr = 0.0
total_images = 0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)

        reconstructed = model(images)

        psnr_values = calculate_psnr(images, reconstructed)

        total_psnr += psnr_values.sum().item()
        total_images += images.size(0)

# Print one binary latent
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)

    lats = torch.tanh(model.encoder(images))
    binary = torch.where(lats >= 0, 1.0, -1.0)

    print("Latent shape:", binary.shape)
    print("First image binary latent:")
    print(binary[0])


average_psnr = total_psnr / total_images

print(f"Final Average Test PSNR: {average_psnr:.2f} dB")



dataiter = iter(test_loader)
images, _ = next(dataiter)
images = images.to(device)
reconstructed = model(images)


fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 3))


for i in range(10):

    axes[0, i].imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
    axes[1, i].axis("off")
plt.show()
