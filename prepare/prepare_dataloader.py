import socket

import torch
from torchvision import datasets, transforms
import os

# hostname = socket.gethostname()
valsplit = 0.2


def prepare_dataloader(args):
    dataset_name = args.dataset
    batch_size = args.batch_size
    data_dir = args.data_dir
    os.path.exists(data_dir) or os.mkdir(data_dir)
    if dataset_name == 'MNIST':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    elif dataset_name == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        print('No such dataset:{}'.format(dataset_name))
        exit(-1)

    fulldataset = datasets.__getattribute__(dataset_name)(
        data_dir, train=True, download=True, transform=train_transform)
    fullsize = len(fulldataset)
    valsize = int(valsplit*fullsize)
    trainsize = fullsize-valsize
    traindataset, valdataset = torch.utils.data.random_split(
        fulldataset, [trainsize, valsize])
    traindataset.transform = train_transform
    valdataset.transform = test_transform

    trainloader = torch.utils.data.DataLoader(
        traindataset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(
        valdataset, batch_size=batch_size, shuffle=False)

    testdataset = datasets.__getattribute__(
        dataset_name)(data_dir, train=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader
