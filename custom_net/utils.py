from pathlib import Path
import torch.optim as optim
from torch.optim import lr_scheduler

import random
import cv2

import torch
import supervisely_lib as sly

from torchvision import transforms
transforms_img = transforms.Compose([
    # step0 - sly_augs will be applied here
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
])


def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


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


def prepare_image_input(image, input_width, input_height):
    # RGB -> Normalized Tensor
    input = cv2.resize(image, (input_width, input_height))
    input = transforms_img(input)  # totensor + normalize
    return input


def train(args, model, criterion, train_loader, valid_loader, validation, classes):
    if args.sly:
        import sly_integration

    #device = torch.device(args.gpu_id)
    lr = args.lr
    n_epochs = args.epochs

    root = Path(args.checkpoints_dir)
    model_path = root / 'model.pt'

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    report_each = args.metrics_period

    if args.sly:
        sly_integration.init_progress_bars(n_epochs, len(train_loader), len(valid_loader))

    for epoch in range(epoch, n_epochs + 1):
        if args.sly:
            sly_integration.progress_set_epoch(epoch)

        if scheduler is not None:
            scheduler.step()

        lr = None
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        model.train()
        random.seed()

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = cuda(inputs)
            with torch.no_grad():
                targets = cuda(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            float(loss.data.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            loss_cpu = float(loss.data.cpu().numpy())
            optimizer.step()
            step += 1 # @TODO: remove?
            if args.sly:
                sly_integration.progress_increment_iter(1)
            sly.logger.info("Train metrics", extra={"epoch": epoch, "iter": i, "lr": lr, "loss": loss_cpu})

            if (i % report_each == 0 or i == len(train_loader) - 1) and args.sly:
                sly_integration.report_train_metrics(epoch, len(train_loader), i, lr, loss_cpu)

        save(epoch + 1)
        metrics = validation(model, criterion, valid_loader, len(classes),
                             progress_cb=sly_integration.progress_increment_iter)
        if args.sly:
            #@TODO: visualize colored masks + change widget + butttons to change layout 3 column
            sly_integration.report_val_metrics(epoch, metrics["loss"], metrics["avg iou"], metrics["avg dice"])
            sly_integration.vis_inference(epoch, model, classes,
                                          args.input_height, args.input_width,
                                          args.project_dir, args.train_vis_items_path)
            # sly_integration.vis_inference(epoch, model, classes,
            #                               args.input_height, args.input_width,
            #                               args.project_dir, args.val_vis_items_path)

