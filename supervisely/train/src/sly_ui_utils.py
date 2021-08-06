import torch.optim as optim
from torch.optim import lr_scheduler


def get_optimizer(args, model):
    my_optimizer = None
    if args.optimizer == "SGD":
        my_optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov
        )
    elif args.optimizer == "Adam":
        my_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        my_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unknown optimizer {args.optimizer}")
    return my_optimizer


def get_scheduler(args, optimizer):
    scheduler = None
    if args.lr_schedule == "":
        pass
    elif args.lr_schedule == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma_step)
    elif args.lr_schedule == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma_exp)
    elif args.lr_schedule == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma_step)
    else:
        raise RuntimeError(f"Unknown lr_schedule {args.lr_schedule}")
    return scheduler