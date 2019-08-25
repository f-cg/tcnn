import os

import torch
import torch.nn as nn

from prepare.prepare_args import parse_args
from prepare.prepare_dataloader import prepare_dataloader
from prepare.prepare_model import prepare_model
from prepare.prepare_optims import get_optim
from prepare.train_test import train_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Should make training go faster for large models
torch.backends.cudnn.benchmark = True

metric_saved = {True: {'class_accs': [], 'total_acc': []},
                False: {'class_accs': [], 'total_acc': []}}

args = parse_args()
torch.manual_seed(args.seed)

saved_dir = './cnn_saved/{}_trained'.format(args.dataset)
id_name = '{}_{}_{}'.format(args.dataset, args.model, args.optim)
if args.dropout:
    id_name += 'dropout'

bestmodelstate, testloader, bestepoch, model = None, None, None, None


def save_metric():
    with open(os.path.join(saved_dir, id_name), 'wt') as f:
        print(metric_saved, file=f)
    save_file = '{}_epoch{}.pt'.format(id_name, bestepoch)
    torch.save(bestmodelstate, os.path.join(saved_dir, save_file))

    model.load_state_dict(bestmodelstate)
    testacc = train_test(model, testloader, 10, None, None,
                         class_tags=None, train_flag=False, metric_saved=metric_saved)
    print('final test acc:{:.2f}'.format(testacc))


def targeted_dropout(model):
    print('dropout')
    gamma = 0.75
    alpha = 0.66
    vec = torch.nn.utils.parameters_to_vector(model.parameters())
    length = len(vec)
    idx = vec.abs().topk(int(length*gamma), largest=False)[1]
    mask = torch.rand(idx.shape[0], device=device) > alpha
    mask = mask.type(torch.float)
    vec[idx] = vec[idx].mul(mask)
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def dropout_rate(model, r):
    vec = torch.nn.utils.parameters_to_vector(model.parameters())
    length = len(vec)
    idx = vec.abs().topk(int(length*r), largest=False)[1]
    vec[idx] = 0.
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def main():
    global bestmodelstate, testloader, bestepoch, model
    print(device)
    os.path.exists(saved_dir) or os.makedirs(saved_dir)
    trainloader, valloader, testloader = prepare_dataloader(args)
    model = prepare_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optim(args, model)
    train_test(model, testloader, args.n_classes,
               optimizer, criterion, train_flag=False)
    best_valacc = 0
    for epoch in range(1, args.epochs+1):
        print('Epoch {}:'.format(epoch))
        train_test(model, trainloader, args.n_classes, optimizer, criterion,
                   train_flag=True, metric_saved=metric_saved, mode='train')
        valacc = train_test(model, valloader, args.n_classes, train_flag=False,
                            metric_saved=metric_saved, mode='val')
        if valacc > best_valacc:
            best_valacc = valacc
            bestmodelstate = model.state_dict()
            print('another best valacc:{:.2f}'.format(valacc))

        if args.dropout:
            targeted_dropout(model)
        if scheduler:
            scheduler.step(epoch)


try:
    main()
except KeyboardInterrupt:
    save_metric()
    exit(0)
save_metric()
