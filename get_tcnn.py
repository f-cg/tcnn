import getpass
import os

import torch

from model import SoftDecisionTree
from parse_args import parse_args
from train_cnn.prepare_model import prepare_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()

user_name = getpass.getuser()
drive = './' if user_name == 'lily' else '/gdrive/My Drive/'
net = prepare_model(args)
pre_dir = os.path.join(drive, './train_cnn/{}_trained'.format(args.dataset))
model = SoftDecisionTree(args, net).to(device)
if args.model == 'naive':
    # st = torch.load('naive_tcnn_tdall20190517_02:01:50.pt')
    # st = torch.load('tmp_unfr10hard20_dt20190519_22:20:24_159.pt')
    st = torch.load('unfr20hard40lr220190521_23:01:00_100.pt')
elif args.model == 'resnet18':
    # st = torch.load('./resnet_tcnn_tdall_unfr40_chopt20190517_11_06_55.pt')
    st = torch.load(
        'resnet_tcnn_tdall_unfr40_chopt_hard20190519_12_34_55_39.pt')
model.load_state_dict(st)
model.to(device)
