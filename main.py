import getpass
import os
import time

import torch

from prepare.prepare_args import parse_args
from prepare.prepare_dataloader import prepare_dataloader
from prepare.prepare_model import prepare_model
from prepare.train_test import train_test
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


def save_metric():
    if best_epoch < 2:
        print('nothing saved')
        return
    saved_name = os.path.join(
        drive, 'tcnn_saved', args.saved_name+date+'_'+str(best_epoch))
    with open(saved_name+'.mt', 'wt') as f:
        print(metric_saved, file=f)
    torch.save(best_state, saved_name+'.pt')

    model.load_state_dict(best_state)
    testacc = model.train_(
        test_loader, None, args.n_classes, train_flag=False, hard=True)
    print('final test acc:{:.2f}'.format(testacc))


net = prepare_model(args).to(device)
pre_dir = os.path.join(drive, 'train_cnn/{}_trained'.format(args.dataset))
# id_name = '{}_{}_{}'.format(args.dataset, args.model, args.optim)

if args.load_pretrain != 'None':
    pretrain = os.path.join(pre_dir, args.load_pretrain)
    net.load_state_dict(torch.load(pretrain))
    print('preload net precision:')
    train_test(net, test_loader, args.n_classes, train_flag=False)

model = SoftDecisionTree(args, net).to(device)
print(count_parameters(model))
print('build tree ok')
model.count_parameters()
metric_saved = {'train': {'class_accs': [], 'hard_class_accs': [], 'total_acc': [], 'hard_total_acc': []},
                'val': {'class_accs': [], 'hard_class_accs': [], 'total_acc': [], 'hard_total_acc': []},
                'leaves': [],  # (epochs,16,1,10)
                'nonezero': [],
                'fr_leaves': []}  # (epochs,16,10)
metric_saved['args'] = str(args)
model.train_(test_loader, None, args.n_classes, train_flag=False)
best_valacc, best_epoch, best_state = 0, 0, None


def main():
    global best_valacc, best_state
    for epoch in range(1, args.epochs + 1):
        if args.unfreeze == epoch:
            model.freeze(False)
            if args.changeopt:
                if args.adamopt:
                    model.optimizer = torch.optim.Adam(model.parameters(),
                                                       lr=0.005, weight_decay=model.args.weight_decay)
                else:
                    model.optimizer = torch.optim.SGD(model.parameters(),
                                                      lr=0.0005, weight_decay=model.args.weight_decay)
        elif args.freeze == epoch:
            model.freeze(True)
        if args.dropout and args.unfreeze < epoch:
            if args.dropout_all:
                model.targeted_dropout_all()
            else:
                model.targeted_dropout()
        if args.make_hard == epoch:
            model.make_hard()
        print('Epoch {} {} {} {}'.format(
            epoch, count_parameters(model),
            count_parameters(model, 1e-6),
            count_parameters(model, 0)))
        model.train_(train_loader, metric_saved, args.n_classes,
                     save_target_on_leaves_=False, hard=False, mode='train')
        valacc = model.train_(valloader, metric_saved,
                              args.n_classes, hard=True, mode='val')
        if valacc > best_valacc:
            best_state = model.state_dict()
            best_valacc = valacc
            print('another best valacc:{:.2f}'.format(valacc))


try:
    main()
except KeyboardInterrupt:
    save_metric()
    exit(0)
save_metric()
