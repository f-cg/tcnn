'''
python3 dropout_test_naivemnist.py --max-depth 4 --model naive --dataset MNIST
'''
import getpass
import math
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import array

from model import SoftDecisionTree
from parse_args import parse_args
from train_cnn.prepare_loader import prepare_loader
from train_cnn.prepare_model import prepare_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
user_name = getpass.getuser()
drive = './' if user_name == 'lily' else '/gdrive/My Drive/'

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


def bar_fc(fcs, subplot=False):
    if args.max_depth == 4:
        pltsnumw = 2
        pltsnumh = 2
        num = 3
        steps = [12*12]+[5*5]*2
    elif args.max_depth == 5:
        pltsnumw = 4
        pltsnumh = 4
        num = 7
        steps = [32*32]+[16*16]*2+[8*8]*4+[1*1]*8
    if subplot:
        plt.figure('bar_fc')
        plt.suptitle('bar_fc')
    pure_rate = [0]*num
    zero_count = [0]*num
    for i, fc in enumerate(fcs):
        if i == num:
            break
        if subplot:
            plt.subplot(pltsnumw, pltsnumh, i+1)
        else:
            plt.figure(i)
            plt.suptitle('bar_fc_'+str(i))
        step = steps[i]
        w = fc.weight
        w = w.detach().cpu().numpy().squeeze()
        wl = w.shape[0]
        max_steps = 40
        groupi = 3
        if wl//step > max_steps:
            wl = max_steps*step
            if (groupi+1)*wl > w.shape[0]:
                groupi = 0
            start = groupi*wl
            end = (groupi+1)*wl
        else:
            wl = wl
            start = 0
            end = wl
        w = w[start:end]
        bar = plt.bar(range(wl), w)
        cs = ['r', 'g']
        for j in range(0, wl, step):
            part = w[j:j+step]
            pos = np.sum(part > 0.001)
            neg = np.sum(part < -0.001)
            print(pos, neg)
            pure_rate[i] += max(pos, neg)/(pos+neg)
            if np.all(np.abs(part) < 0.05):
                zero_count[i] += 1
        pure_rate[i] /= (wl//step)
        for j in range(0, wl):
            ci = j//step % 2
            bar[j].set_color(cs[ci])
    plt.figure('pure_rate')
    plt.suptitle('pure_rate')
    plt.plot(pure_rate)
    plt.figure('zero_count')
    plt.suptitle('zero_count')
    plt.plot(zero_count)


def im_fc(fcs):
    if args.max_depth == 4:
        pltsnumw = 2
        pltsnumh = 2
        num = 3
        steps = [12*12]+[5*5]*2
    elif args.max_depth == 5:
        pltsnumw = 4
        pltsnumh = 4
        num = 15
        steps = [32*32]+[16*16]*2+[8*8]*4+[1*1]*8
    for i, fc in enumerate(fcs[1:]):
        plt.figure('im_fc'+str(i))
        plt.suptitle('im_fc'+str(i))
        w = fc.weight
        w = w.detach().cpu().numpy().squeeze()
        wl = w.shape[0]
        if wl == 864:
            step = 12*12
        elif wl == 400:
            step = 5*5
        else:
            step = -100
        for j in range(0, wl, step):
            plt.subplot(4, 4, j//step+1)
            size = int(math.sqrt(step))
            im = w[j:j+step].reshape([size, size])
            im -= im.min()
            im = np.abs(im)
            im /= im.max()
            im *= 255
            im = im.astype(np.uint8)
            plt.imshow(im, cmap='gray')


def bar_node_probs(model):
    trainloader, testloader = prepare_loader(args)
    plt.figure('node_probs')
    plt.suptitle('node_probs')
    node_probs = model.single_test(testloader)
    node_probs = array(node_probs)
    plt.hist(node_probs, bins=np.linspace(0., 1., num=50))


print(device)
net = prepare_model(args).to(device)
print('CNN: ', count_parameters(net))
model = SoftDecisionTree(args, net).to(device)
#    if args.model == 'naive':
#        st = 'naive_tcnn_tdall20190517_02:01:50.pt'
#        st = 'tmp_hard_nodt20190519_20:30:16_200.pt'
#        st = 'tmp_unfr10hard20_dt20190519_22:20:24_119.pt'
#        st = './tmp_unfr10hard20_dt20190519_22:20:24_200.pt'
#    elif args.model == 'resnet18':
#        #st = 'resnet_tcnn_tdall_unfr40_chopt20190517_15_25_24.pt'
#        st = 'resnet_tcnn_tdall_unfr40_chopt220190518_14_30_23_179.pt'
#        st = 'resnet_tcnn_tdall_unfr40_chopt_hard20190519_12_34_55_79.pt'
st = join(drive,  args.TCNN)
print(st)
st = torch.load(st)
model.load_state_dict(st)
model.count_parameters()
exit(0)
fcs = list(model.module_list.modules())
fcs = fcs[1:]
bar_fc(fcs)
# im_fc(fcs)
if args.bar_node_probs:
    bar_node_probs(model)
# plt.show()


def pltshow(folder='plt_tmp', figs=None, dpi=200):
    if "DISPLAY" in os.environ:
        plt.show()
        return
    os.path.exists(folder) or os.makedirs(folder, exist_ok=True)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        if fig._suptitle:
            sup = fig._suptitle.get_text()+'_'+str(i)
        else:
            sup = str(i)
        fig.savefig(join(folder, sup+'.png'), dpi=800)
    plt.close()


pltshow(join('plt_tmp', args.TCNN))
