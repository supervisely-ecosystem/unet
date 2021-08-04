import argparse
import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import supervisely_lib as sly
import copy
import time

from sly_seg_dataset import SlySegDataset
from loss import dice_loss
from unet_classic import UNet


def main():
    parser = argparse.ArgumentParser()
    # model architecture
    parser.add_argument('--model', default='UNet-classic', help='model architecture name')

    # for data loader
    parser.add_argument('--project-dir', default='', help='path to sly project with segmentation masks')
    parser.add_argument('--classes-path', default='', help='path to the list of classes (order matters)')
    parser.add_argument('--train-set-path', default='', help='list of training items')
    parser.add_argument('--val-set-path', default='', help='list of validation')

    # basic hyperparameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--input-size', type=int, default=256, help='model input image size')
    parser.add_argument('--batch-size', type=int, default=8)

    # optimizer
    parser.add_argument('--optimizer', default='SGD', help='SGD / Adam / AdamW')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9, help='used only with SGD')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--nesterov', action='store_true', help='used only with SGD')

    # lr schedule
    parser.add_argument('--lr-schedule', default='', help='No schedule (default) / StepLR / ExponentialLR / MultiStepLR')
    parser.add_argument('--step-size', type=int, default=5, help='used only with StepLR')
    parser.add_argument('--gamma-step', type=float, default=0.1, help='used only with StepLR and MultiStepLR')
    parser.add_argument('--milestones', default='[5, 10, 15]', help='used only with MultiStepLR')
    parser.add_argument('--gamma-exp', type=float, default=0.9, help='used only with StepLR and ExponentialLR')

    # system
    parser.add_argument('--gpu-id', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=0)

    # logging
    parser.add_argument('--metrics-period', type=int, default=10, help='How often (num of iteration) metrics should be logged')

    # checkpoints
    parser.add_argument('--val-interval', type=int, default=1, help='Evaluate val set every N epochs')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--save-last', action='store_true', help='save last checkpoint')
    parser.add_argument('--save-best', action='store_true', help='save best checkpoint')

    # integration with dashboard (ignore flag during local dev)
    parser.add_argument('--sly', action='store_true', help='for Supervisely App integration')

    #@TODO: artifacts dir
    #@TODO: model architecture
    #@TODO: augs path

    opt = parser.parse_args()
    print("Input arguments:", opt)

    train(opt)


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(device, model, dataloaders, optimizer, scheduler=None, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train(opt):
    classes = sly.json.load_json_file(opt.classes_path)

    model = None
    if opt.model == "UNet-classic":
        model = UNet(len(classes))
    else:
        raise RuntimeError(f"Unknown model architecture {opt.model}")
    device = torch.device(opt.gpu_id)
    model = model.to(device)
    summary(model, input_size=(3, opt.input_size, opt.input_size))

    optimizer_ft = None
    if opt.optimizer == "SGD":
        optimizer_ft = optim.SGD(
            model.parameters(),
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov
        )
    elif opt.optimizer == "Adam":
        optimizer_ft = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == "AdamW":
        optimizer_ft = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise RuntimeError(f"Unknown optimizer {opt.optimizer}")

    exp_lr_scheduler = None
    if opt.lr_schedule == "":
        pass
    elif opt.lr_schedule == "StepLR":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=opt.step_size, gamma=opt.gamma_step)
    elif opt.optimizer == "ExponentialLR":
        exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=opt.gamma_exp)
    elif opt.optimizer == "MultiStepLR":
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=opt.milestones, gamma=opt.gamma_step)
    else:
        raise RuntimeError(f"Unknown lr_schedule {opt.lr_schedule}")

    train_set = SlySegDataset(opt.project_dir, opt.classes_path, opt.train_set_path, input_size=opt.input_size)
    val_set = SlySegDataset(opt.project_dir, opt.classes_path, opt.val_set_path, input_size=opt.input_size)
    dataloaders = {
        'train': DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers),
        'val': DataLoader(val_set, batch_size=opt.batch_size, num_workers=opt.num_workers)
    }
    model = train_model(device, model, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=200) #opt.epochs)


if __name__ == '__main__':
    main()