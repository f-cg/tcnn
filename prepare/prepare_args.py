import argparse

model_options = ['resnet18', 'naive', 'naive_norm', 'naive_linear']
dataset_options = ['MNIST', 'CIFAR10', 'IMAGENET']
optim_options = ['Adam', 'SGD']
num_classes = dict(zip(dataset_options, [10, 100, 10, 1000]))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='TCNN and CNN args')
    parser.add_argument('--dataset', type=str, default=dataset_options[0],
                        choices=dataset_options)
    parser.add_argument('--model', type=str, default=model_options[0],
                        choices=model_options)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_depth', type=int, default=5,
                        help='maximum depth of tree')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lmbda', type=float, default=0.1,
                        help='temperature rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--optim', type=str, default=optim_options[0],
                        choices=optim_options)
    parser.add_argument('--freeze', type=int, default=1,
                        metavar='FR', help='freeze self.module when epoch=FR')
    parser.add_argument('--unfreeze', type=int, default=1000,
                        metavar='UNFR', help='unfreeze self.module when epoch=UNFR')
    parser.add_argument('--saved_name', type=str, default='metric')
    parser.add_argument('--make_hard', type=int, metavar='MH',
                        default=1000, help='make leaves hard when epoch=MH')
    parser.add_argument('--dropout', action='store_true', default=False)
    parser.add_argument('--changeopt', action='store_true', default=False)
    parser.add_argument('--adamopt', action='store_true', default=False)
    parser.add_argument('--dropout_all', action='store_true', default=False)
    parser.add_argument('--load_pretrain', type=str, default='None')
#    parser.add_argument('--TCNN', type=str, default='None')
#    parser.add_argument('--bar-node-probs', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, default='/data/')
    parser.add_argument('--valsplit', type=float, default=0.2)

    args = parser.parse_args()
    args.n_classes = num_classes[args.dataset]

    if args.dataset == 'IMAGENET':
        args.data_augmentation = True
    return args
