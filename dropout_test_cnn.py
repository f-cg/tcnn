import argparse
import os
from functools import partial

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from prepare_loader import prepare_loader
from prepare_model import prepare_model
from train_test import train_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Should make training go faster for large models
torch.backends.cudnn.benchmark = True

metric_saved = {True: {'class_accs': [], 'total_acc': []},
                False: {'class_accs': [], 'total_acc': []}}

parser = argparse.ArgumentParser(description='Train CNN')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='D',
                    choices=['MNIST', 'CIFAR10'], help='(default: MNIST)')
parser.add_argument('--model', type=str, default='naive', metavar='M',
                    choices=['resnet18', 'naive', 'naive_norm', 'naive_linear'], help='(default: naive)')
parser.add_argument('--optim', type=str, default='Adam', metavar='O',
                    choices=['Adam', 'SGD'], help='(default: Adam)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
args = parser.parse_args()
print(args)


def dropout_rate(model, r):
    vec = torch.nn.utils.parameters_to_vector(model.parameters())
    length = len(vec)
    idx = vec.abs().topk(int(length*r), largest=False)[1]
    vec[idx] = 0.
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def main():
    print(device)
    trainloader, testloader = prepare_loader(args)
    model = prepare_model(args).to(device)
#    ptname='./MNIST_trained/MNIST_naive_SGD_epoch45.pt'
    ptname = './MNIST_trained/MNIST_naive_Adam_dropout_epoch10.pt'
    st = torch.load(ptname)
    model.load_state_dict(st)
    for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
        print('drop rate:{}'.format(r))
        dropout_rate(model, r)
        train_test(model, testloader, 10, train_flag=False)


if __name__ == '__main__':
    main()
