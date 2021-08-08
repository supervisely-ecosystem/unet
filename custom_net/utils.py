import json
from datetime import datetime
from pathlib import Path
import torch.optim as optim
from torch.optim import lr_scheduler

import random
import numpy as np
import cv2

import torch
import tqdm

import sly_integration

from torchvision import transforms
transforms_img = transforms.Compose([
    # step0 - sly_augs will be applied here
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
])


def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


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

    report_each = 10
    log = root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        if scheduler is not None:
            scheduler.step()
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])

        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)
                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, len(classes))
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)

            if args.sly:
                #@TODO: remove __bg__ class frm visualization
                #@TODO: limit max value for input-number
                #@TODO: synced views - check (disable default flag to False?)
                sly_integration.vis_inference(epoch, model, classes,
                                              args.input_height, args.input_width,
                                              args.project_dir, args.train_vis_items_path)
                sly_integration.vis_inference(epoch, model, classes,
                                              args.input_height, args.input_width,
                                              args.project_dir, args.val_vis_items_path)
                xxx = 10
                xxx += 1
                #@TODO: predictions improvement over time
                #predictions =
                #img_path = "/app_debug_data/data/Lemons (Annotated)_seg/ds1/img/IMG_0748.jpeg"
                #save_path = f"/app_debug_data/{epoch}.jpeg"
                #inference(model, classes, args.input_height, args.input_width, img_path, save_path)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
