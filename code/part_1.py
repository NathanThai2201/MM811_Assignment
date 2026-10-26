import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
import struct
from array import array
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10
    
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# input_path = ''

# training_images_filepath = 'train-images-idx3-ubyte'
# training_labels_filepath = 'train-labels-idx1-ubyte'
# test_images_filepath = 't10k-images-idx3-ubyte'
# test_labels_filepath = 't10k-labels-idx1-ubyte'

#
# MNIST Data Loader Class, adapted from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
# Things to note: MNIST Images are 28x28. Grayscale.
#

# class MnistDataloader(object):
#     def __init__(self, training_images_filepath,training_labels_filepath,
#                  test_images_filepath, test_labels_filepath):
#         self.training_images_filepath = training_images_filepath
#         self.training_labels_filepath = training_labels_filepath
#         self.test_images_filepath = test_images_filepath
#         self.test_labels_filepath = test_labels_filepath
    
#     def read_images_labels(self, images_filepath, labels_filepath):        
#         labels = []
#         with open(labels_filepath, 'rb') as file:
#             magic, size = struct.unpack(">II", file.read(8))
#             if magic != 2049:
#                 raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
#             labels = array("B", file.read())        
        
#         with open(images_filepath, 'rb') as file:
#             magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
#             if magic != 2051:
#                 raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
#             image_data = array("B", file.read())        
#         images = []
#         for i in range(size):
#             images.append([0] * rows * cols)
#         for i in range(size):
#             img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
#             img = img.reshape(28, 28)
#             images[i][:] = img            
        
#         return images, labels
            
#     def load_data(self):
#         x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
#         x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
#         return (x_train, y_train),(x_test, y_test)  
    
# class ExampleDataset(Dataset):
#     def __init__(self, data_dir, transform = None):
#         self.data = ImageFolder(data_dir, transform = transform)
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self,idx):
#         return self.data[idx]

# mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
# (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()




# LeNet works with images of size 32x32 so we will have to preprocess:


#Loading the dataset and preprocessing
# resizing option: transforms.Resize((32,32)),
train_dataset = torchvision.datasets.MNIST(root = './data',
                                               train = True,
                                               transform = transforms.Compose([
                                                      transforms.ToTensor(),]),
                                               download = True)
    
    
test_dataset = torchvision.datasets.MNIST(root = './data',
                                              train = False,
                                              transform = transforms.Compose([
                                                      transforms.ToTensor(),]),
                                              download=True)
    
    
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True)
    
    
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                               batch_size = batch_size,
                                               shuffle = True)


# Modified LeNet, using 28x28.
class ModifiedLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            # starts as image 28x28x1 
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2), # (28(in) - 5(kernel) + 2*2(padding))/1(stride) + 1  ->  28x28x5
            nn.BatchNorm2d(5),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2), # floor((28(in) - 2(kernel))/2(stride)) + 1  ->  14x14x5


            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=0), # -> 10x10x10
            nn.BatchNorm2d(10),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> 5x5x10
        )
        #if size is 32x32 -> dense layer is 576 instead of 400
        self.fc = nn.Linear(250, 120)
        self.sigmoid = nn.Sigmoid()   
        self.fc1 = nn.Linear(120, 84)
        self.sigmoid1 = nn.Sigmoid()  
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1) #flatten
        out = self.fc(out)
        out = self.sigmoid(out)
        out = self.fc1(out)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        return out

    
model = ModifiedLeNet(num_classes).to(device)
    
#Setting the loss function
cost = nn.CrossEntropyLoss()
    
#Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
#this is defined to print how many steps are remaining when training
total_step = len(train_loader)


total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)
            
        #Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In the test phase, we don't need to compute gradients (for memory efficiency)

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')