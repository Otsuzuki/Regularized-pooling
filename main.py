from network_model import VGG11
from variable import parameter
from plot import Loss
import os
import cv2
import csv
import glob
import random
import pprint
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

batch_size, n_classes, epochs, image_width, learning_rate, pool_kernel, pool_stride, output_width, smooth_kernel, smooth_padding, device = parameter()

FolderName = "./Folder/"
if not os.path.exists(FolderName):
    os.mkdir(FolderName)

# download MNIST
transformer = torchvision.transforms.Compose([
       transforms.Resize(image_width),
       transforms.ToTensor()
])

ds_train = torchvision.datasets.MNIST(
    './mnist',
    transform=transformer,
    download=True)

dataloader_train = torch.utils.data.DataLoader(
    dataset=ds_train,
    batch_size=batch_size,
    shuffle=True)
ds_test = torchvision.datasets.MNIST(
    './mnist',
    transform=transformer,
    train=False,
    download=True)

dataloader_test = torch.utils.data.DataLoader(
    dataset=ds_test,
    batch_size=batch_size,
    shuffle=True)

#training
def train(dataloader_train,epoch):
    model.train()
    train_loss = []
    for batch, (images, labels) in enumerate(dataloader_train, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return(train_loss)

#validation
def eval(dataloader_test,epoch):
    model.eval()
    correct = 0
    total = 0
    valid_loss = []
    valid_acc = []
    with torch.no_grad():
        for batch, (images, labels) in enumerate(dataloader_test, 1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            valid_acc.append(float(correct) / total)

    return valid_loss, valid_acc

if __name__ == "__main__":
    # define model
    model = VGG11(n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    global_step = 0

    avg_train_loss = []
    avg_valid_loss = []
    avg_valid_acc = []
    for epoch in range(1, epochs+1):
        train_loss = train(dataloader_train,epoch)
        valid_loss, valid_acc = eval(dataloader_test,epoch)

        t_loss = np.average(train_loss)
        v_loss = np.average(valid_loss)
        v_acc = np.average(valid_acc)
        avg_train_loss.append(t_loss)
        avg_valid_loss.append(v_loss)
        avg_valid_acc.append(v_acc)
        epoch_len = len(str(epochs))
        print('====== Regularized pooling ======')
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                    f'train_loss: {t_loss:.5f} ' +
                    f'valid_loss: {v_loss:.5f}')
        print(print_msg)
        print("Val Acc : %.4f" % v_acc)

    # Record loss and accuracy in CSV file
    with open(FolderName + "/Regularized_acc.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(avg_valid_acc)
    with open(FolderName + "/Regularized_trainloss.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(avg_train_loss)
    with open(FolderName + "/Regularized_validloss.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(avg_valid_loss)

    # Plot loss
    Loss(avg_valid_acc, avg_train_loss, avg_valid_loss, FolderName)