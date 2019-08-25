import torch

from torch.optim.lr_scheduler import MultiStepLR


def get_optim(args, model):
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = None
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=0.9, nesterov=True,
                                    weight_decay=args.weight_decay)
        if args.model == 'resnet18':
            scheduler = MultiStepLR(optimizer, milestones=[
                                    60, 120, 160], gamma=0.2)
        else:
            scheduler = MultiStepLR(optimizer, milestones=[
                                    10, 15, 20], gamma=0.2)
    return optimizer, scheduler
