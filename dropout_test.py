'''
python3 dropout_test_naivemnist.py --max-depth 4 --model naive --dataset MNIST
'''
import argparse
import os
from functools import partial

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from model import SoftDecisionTree
from parse_args import parse_args
from train_cnn.prepare_loader import prepare_loader
from train_cnn.prepare_model import prepare_model
from train_cnn.train_test import train_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Should make training go faster for large models
torch.backends.cudnn.benchmark = True

args = parse_args()
metric_saved = {True: {'class_accs': [], 'hard_class_accs': [], 'total_acc': [], 'hard_total_acc': []},
                False: {'class_accs': [], 'hard_class_accs': [], 'total_acc': [], 'hard_total_acc': []},
                'leaves': [],  # (epochs,16,1,10)
                'nonezero': [],
                'fr_leaves': []}  # (epochs,16,10)


def count_parameters(m, thresh=1e-4):
    '''
    routers及leaves的参数:w,b,beta,Q.
    即 \sum_{i=1}^{L}((dim+1)*2^{i-1}+2^{i-1})+2^{L}*n_class
    '''
    return sum(p.numel() for p in m.parameters())


def dropout_rate(model, num, part=False):
    if part:
        get_params = (lambda: model.frozen_module_list.parameters())
        print(sum(p.numel()for p in get_params())-num)
    else:
        get_params = (lambda: model.parameters())
    vec = torch.nn.utils.parameters_to_vector(get_params())
    idx = vec.abs().topk(num, largest=False)[1]
    vec[idx] = 0.
    torch.nn.utils.vector_to_parameters(vec, get_params())


def dropout(model, stay, model_type, part=False):
    pnums = count_parameters(model)
    for s in stay:
        num = pnums-s
        r = num/pnums
        print('{:.3f},{}'.format(r, s), end=':')
        dropout_rate(model, num, part)
        if model_type == 'CNN':
            train_test(model, testloader, 10, train_flag=False)
        elif model_type == 'TCNN':
            model.train_(testloader, metric_saved, train=False)
        else:
            raise Exception('model type error')
        print('')


if __name__ == '__main__':
    print(device)
    trainloader, testloader = prepare_loader(args)
    net = prepare_model(args).to(device)
    model = SoftDecisionTree(args, net).to(device)
    if args.model == 'naive':
        st = 'naive_tcnn_tdall20190517_02:01:50.pt'
        st = 'tmp_hard_nodt20190519_20:30:16_200.pt'
    elif args.model == 'resnet18':
        #st = 'resnet_tcnn_tdall_unfr40_chopt20190517_15_25_24.pt'
        st = 'resnet_tcnn_tdall_unfr40_chopt220190518_14_30_23_179.pt'
#        st = 'resnet_tcnn_tdall_unfr40_chopt220190518_03_39_50_19.pt'
    st = torch.load(st)
    model.load_state_dict(st)
    model.count_parameters()
    print('cnn:', count_parameters(prepare_model(args).to(device)))
    pnums = count_parameters(model)
    print(pnums)
    rs = [0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    stay = [int(pnums-pnums*r) for r in rs]
    dropout(model, stay, 'TCNN', True)

#    trainloader, testloader = prepare_loader(args)
    model = prepare_model(args).to(device)
#    ptname = 'train_cnn/MNIST_trained/MNIST_naive_Adam_dropout_epoch10.pt'
    if args.model == 'naive':
        ptname = 'train_cnn/MNIST_trained/MNIST_naive_Adam_dropout_epoch60_20190516_23:35:02.pt'
    elif args.model == 'resnet18':
        ptname = 'CIFAR10_resnet18_SGD_dropout_epoch200_20190516_14_41_03.pt'
        ptname = 'CIFAR10_resnet18_SGD_dropout_epoch100_20190516_14_41_03.pt'
    st = torch.load(ptname)
    model.load_state_dict(st)
    pnums = count_parameters(model)
    stay = [int(pnums-pnums*r) for r in rs]
    dropout(model, stay, 'CNN')
