import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import dataloader

trnasform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=trnasform)
test_data = datasets.MNIST(root='./data', train=True, download=True, transform=trnasform)