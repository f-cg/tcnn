import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_mnist(nn.Module):

    def __init__(self):
        super(Net_mnist, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)  # (1,28,28)=>(6,24,24)
        self.mp1 = nn.MaxPool2d(2)  # (6,24,24)=>(6,12,12)
        self.conv2 = nn.Conv2d(6, 16, 3)  # (6,12,12)=>(16,10,10)
        self.mp2 = nn.MaxPool2d(2)  # =>(16,5,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)
        # [(6,24,24),(6,12,12),(16,10,10),(16,5,5),64,10]

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.mp1(F.sigmoid(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = self.mp2(F.sigmoid(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class Net_mnist_norm(nn.Module):

    def __init__(self):
        super(Net_mnist_norm, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)  # (1,28,28)=>(6,24,24)
        self.mp1 = nn.MaxPool2d(2)  # (6,24,24)=>(6,12,12)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3)  # (6,12,12)=>(16,10,10)
        self.mp2 = nn.MaxPool2d(2)  # =>(16,5,5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.bn3 = nn.BatchNorm1d(64, 64)
        self.fc2 = nn.Linear(64, 10)
        # [(6,24,24),(6,12,12),(16,10,10),(16,5,5),64,10]

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.mp1(F.sigmoid(self.conv1(x)))
        x = self.bn1(x)
        # If the size is a square you can only specify a single number
        x = self.mp2(F.sigmoid(self.conv2(x)))
        x = self.bn2(x)
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = self.bn3(x)
        x = self.fc2(x)
        return x


class Net_mnist_linear(nn.Module):

    def __init__(self):
        super(Net_mnist_linear, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)  # (1,28,28)=>(6,24,24)
        self.mp1 = nn.Conv2d(6, 6, 2, stride=2)  # (6,24,24)=>(6,12,12)
        self.conv2 = nn.Conv2d(6, 6, 3, padding=1)  # =>(16,12,12)
        self.conv3 = nn.Conv2d(6, 16, 3)  # (6,12,12)=>(16,10,10)
        self.mp3 = nn.Conv2d(16, 16, 2, stride=2)  # =>(16,5,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 10)
        # [(6,24,24),(6,12,12),(16,10,10),(16,5,5),10]

    def forward(self, x):
        x = self.mp1(self.conv1(x))
        x = self.conv2(x)
        x = self.mp3(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


class Net_cifar10(nn.Module):

    def __init__(self):
        super(Net_cifar10, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)  # (3,32,32)=>(6,28,28)
        self.mp1 = nn.MaxPool2d(2)  # (6,28,28)=>(6,14,14)
        self.conv2 = nn.Conv2d(6, 16, 3)  # (6,14,14)=>(16,12,12)
        self.mp2 = nn.MaxPool2d(2)  # =>(16,6,6)
        self.fc1 = nn.Linear(16 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)
        # [(6,24,24),(6,12,12),(16,10,10),(16,5,5),64,10]

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.mp1(F.sigmoid(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = self.mp2(F.sigmoid(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def prepare_model(args=None):
    if args.model == 'naive':
        if args.dataset == 'MNIST':
            return Net_mnist()
        if args.dataset == 'CIFAR10':
            return Net_cifar10()
    elif args.model == 'naive_linear':
        return Net_mnist_linear()
    elif args.model == 'naive_norm':
        return Net_mnist_norm()
    elif args.model == 'resnet18':
        from .resnet import ResNet18
        if args.dataset == 'MNIST':
            return ResNet18(1)
        if args.dataset == 'CIFAR10':
            return ResNet18()
