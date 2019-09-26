import getpass
import os
import time
from os.path import join

import torch

from prepare.prepare_args import parse_args
from prepare.prepare_dataloader import prepare_dataloader
from prepare.prepare_model import prepare_model
from tcnn_model import SoftDecisionTree
from utils import check_on_gpu, count_parameters

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
user_name = getpass.getuser()
drive = './' if user_name == 'lily' else '/gdrive/My Drive/'

args = parse_args()
date = time.strftime("%Y%m%d_%H:%M:%S", time.localtime())

torch.manual_seed(args.seed)
if 'cuda' in str(device):
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 2, 'pin_memory': 'cuda' in str(device)}
train_loader, valloader, test_loader = prepare_dataloader(args)


net = prepare_model(args).to(device)
# train_test(net, valloader, classes=1000, train_flag=False, cols=6)

model = SoftDecisionTree(args, net).to(device)
model.load_state_dict(torch.load(
    './IMAGENET_resnet18_Adam20190911_16:17:52_31.pt'))
print(count_parameters(model))
print('build tree ok')
model.count_parameters()
testacc = model.train_(test_loader, None, args.n_classes, train_flag=False,
                       hard=True, mode='test')
