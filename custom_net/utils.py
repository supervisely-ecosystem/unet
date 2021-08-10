import torch.optim as optim
from torch.optim import lr_scheduler
import os
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

best_model_path = None
best_val_loss = float('inf')
saved_models = []


def save(save_dir, epoch, model, val_loss=None, max_ckpts=-1):
    global best_model_path, best_val_loss, saved_models
    sly.fs.ensure_base_path(save_dir)

    if val_loss is not None:
        if val_loss < best_val_loss:
            # save model as best
            if best_model_path is not None:
                sly.fs.silent_remove(best_model_path)
            best_model_path = os.path.join(save_dir, f'model_{epoch:03d}_best.pth')
            torch.save(model.state_dict(), best_model_path)
            best_val_loss = val_loss
        return

    model_path = os.path.join(save_dir, f'model_{epoch:03d}.pth')
    torch.save(model.state_dict(), model_path)
    if max_ckpts != -1:
        # save last N
        saved_models.append(model_path)
        while len(saved_models) > max_ckpts:
            path_to_remove = saved_models.pop(0)
            sly.fs.silent_remove(path_to_remove)


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

    if args.custom_weights != "" and sly.fs.file_exists(args.custom_weights):
        state = torch.load(str(args.custom_weights))
        model.load_state_dict(state)
        sly.logger.info(f"Restored model: {args.custom_weights}")

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    report_each = args.metrics_period
    if args.sly:
        sly_integration.init_progress_bars(args.epochs, len(train_loader), len(valid_loader))

    epoch = 1
    for epoch in range(epoch, args.epochs + 1):
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
            if args.sly:
                sly_integration.progress_increment_iter(1)
            sly.logger.info("Train metrics", extra={"epoch": epoch, "iter": i, "lr": lr, "loss": loss_cpu})

            if (i % report_each == 0 or i == len(train_loader) - 1) and args.sly:
                sly_integration.report_train_metrics(epoch, len(train_loader), i, lr, loss_cpu)

        if epoch % args.val_interval:
            metrics = validation(model, criterion, valid_loader, len(classes),
                                 progress_cb=sly_integration.progress_increment_iter)
            if args.save_best:
                save(args.checkpoints_dir, epoch, model, metrics["loss"], args.max_keep_ckpts)

            if args.sly:
                sly_integration.report_val_metrics(epoch, metrics["loss"], metrics["avg iou"], metrics["avg dice"])
                sly_integration.vis_inference(epoch, model, classes,
                                              args.input_height, args.input_width,
                                              args.project_dir, args.train_vis_items_path)
                sly_integration.vis_inference(epoch, model, classes,
                                              args.input_height, args.input_width,
                                              args.project_dir, args.val_vis_items_path, update=True)
        if epoch % args.checkpoint_interval:
            save(args.checkpoints_dir, epoch, model, None, args.max_keep_ckpts)

    if args.save_last:
        torch.save(model.state_dict(), os.path.join(args.checkpoints_dir, f'model_{epoch:03d}_last.pth'))