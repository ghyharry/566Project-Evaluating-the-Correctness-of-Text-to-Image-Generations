from __future__ import print_function
import torch
import torchvision
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image


# The parts that you should complete are designated as TODO
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO: define the layers of the network
        self.convl1 = nn.Conv2d(3, 32, kernel_size=9,stride=2,padding=4)
        self.convl2 = nn.Conv2d(32, 64, kernel_size=3,stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Dropout2d = nn.Dropout2d(p=0.25)
        self.l1 = nn.Linear(64 * 31 * 31, 512)
        self.Dropout = nn.Dropout2d(p=0.5)
        self.l2 = nn.Linear(512, 2)

    def forward(self, x):
        # TODO: define the forward pass of the network using the layers you defined in constructor
        x = F.relu(self.convl1(x))
        x = F.relu(self.convl2(x))
        x = self.maxpool(x)
        x = self.Dropout2d(x)
        goflatten = x.size(0)
        x = x.reshape(goflatten, -1)
        x = F.relu(self.l1(x))
        x = self.Dropout(x)
        return self.l2(x)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        accuracy = 100. * batch_idx / len(train_loader)
        if batch_idx % 100 == 0:  # Print loss every 100 batch
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                accuracy, loss.item()))
    accuracy = ttt(model, device, train_loader)
    return accuracy


def ttt(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        accuracy))

    return accuracy


def getAllFile(filename):
    goodSamples = os.walk(filename)

    files = []
    for _, _, fs in goodSamples:
        for f in fs:
            files.append(f)
    return files


def plot_filters_multi_channel(t):
    # get the number of kernals
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].cpu().numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.savefig('myimage.png', dpi=100)
    plt.tight_layout()
    plt.show()


def plot_filters_single_channel(t):
    # kernels depth * number of kernels
    nplots = t.shape[0] * t.shape[1]
    ncols = 12

    nrows = 1 + nplots // ncols
    # convert tensor to numpy image
    npimg = np.array(t.cpu().numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].cpu().numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()

def plot_weights(model, layer_num, single_channel=True, collated=False):
    # extracting the model features at the particular layer number
    layer = model.convl1

    # checking whether the layer is convolution layer or not
    if isinstance(layer, nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = model.convl1.weight.data

        if single_channel:
            if collated:
                pass
            else:
                plot_filters_single_channel(weight_tensor)


        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor)
            else:
                print("Can only plot weights with three channels with single channel = False")

    else:
        print("Can only visualize layers which are convolutional")





files = getAllFile('./sample')
x = np.array([np.array(Image.open('./sample' + '/' + file)) for file in files])
xx=[]
for i in range(len(x)):
    e0=x[i,:,:,0]
    e1 = x[i,:, :, 1]
    e2 = x[i,:, :, 2]
    xx.append(np.array([e0, e1, e2]))

x=np.asarray(xx)
#print(x[0][0])


torch.manual_seed(4)
np.random.seed(6)
# Training settings
use_cuda = True # Switch to False if you only want to use your CPU
learning_rate = 0.0004
NumEpochs = 10
batch_size = 8

device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")
df = pd.read_excel('./label.xlsx', engine='openpyxl', header=None)
label=df.to_numpy().T
label=label[0]
print(label)

tensor_x = torch.tensor(x, dtype=torch.float,device=device)
tensor_y = torch.tensor(label, dtype=torch.long, device=device)
train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset

train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [500, 75])
train_loader = utils.DataLoader(train_dataset, batch_size=8,shuffle=True) # create your dataloader

test_loader = utils.DataLoader(test_dataset)

model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_acclist=[]
test_acclist=[]

for epoch in range(NumEpochs):
    train_acc = train(model, device, train_loader, optimizer, epoch)
    train_acclist.append(train_acc)

    test_acc = ttt(model, device, test_loader)
    test_acclist.append(test_acc)

plot_weights(model, 0, single_channel=False)

plt.figure("train_acc and test_acc vs # of epoch")
plt.plot(list(range(NumEpochs)), train_acclist, c='r', label="train accuracy")
plt.plot(list(range(NumEpochs)), test_acclist, c='g', label="test accuracy")
plt.xlabel("# of epoch")
plt.ylabel("accuracy")
plt.legend(loc=0)
plt.show()