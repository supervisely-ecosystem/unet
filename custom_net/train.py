import argparse
import os
from pathlib import Path
from validation import validation_binary, validation_multi

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16, AlbuNet
from loss import LossBinary, LossMulti
import utils

import supervisely_lib as sly
from sly_seg_dataset import SlySegDataset



# Pytorch can automatically download pretrained weights, we use direct download to show progress bar in UI at step05
from torchvision.models.vgg import model_urls as vgg_urls
from torchvision.models.resnet import model_urls as resnet_urls

model_list = {
    'UNet': {
        "class": UNet,
        "description": "Vanilla UNet with random weights"
    },
    'UNet11': {
        "class": UNet11,
        "description": "Initialized from vgg-11 pretrained on ImageNet",
        "pretrained": vgg_urls['vgg11']
    },
    'UNet16': {
        "class": UNet16,
        "description": "Initialized from vgg-16 pretrained on ImageNet",
        "pretrained": vgg_urls['vgg16']
    },
    'AlbuNet': {
        "class": AlbuNet,
        "description": "UNet modification with resnet34 encoder pretrained on ImageNet",
        "pretrained": resnet_urls['resnet34']
    },
    'LinkNet34': {
        "class": LinkNet34,
        "description": "LinkNet with resnet34 encoder pretrained on ImageNet",
        "pretrained": resnet_urls['resnet34']
    }
}


def main():
    # arg('--model', type=str, default='UNet', choices=moddel_list.keys())

    parser = argparse.ArgumentParser()
    # model architecture
    parser.add_argument('--model', default='UNet11', help='model architecture name')

    # for data loader
    parser.add_argument('--project-dir', default='', help='path to sly project with segmentation masks')
    parser.add_argument('--classes-path', default='', help='path to the list of classes (order matters)')
    parser.add_argument('--train-set-path', default='', help='list of training items')
    parser.add_argument('--val-set-path', default='', help='list of validation')
    parser.add_argument('--sly-augs-path', default='123', help='path to SlyImgAug config')

    # basic hyperparameters
    parser.add_argument('--epochs', type=int, default=5)
    #parser.add_argument('--input-size', type=int, default=256, help='model input image size')
    parser.add_argument('--input-height', type=int, default=256, help='model input image size')
    parser.add_argument('--input-width', type=int, default=256, help='model input image size')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--jaccard-weight', default=0.5, type=float)

    # optimizer
    parser.add_argument('--optimizer', default='SGD', help='SGD / Adam / AdamW')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9, help='used only with SGD')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--nesterov', action='store_true', help='used only with SGD')

    # lr schedule
    parser.add_argument('--lr-schedule', default='',
                        help='No schedule (default) / StepLR / ExponentialLR / MultiStepLR')
    parser.add_argument('--step-size', type=int, default=5, help='used only with StepLR')
    parser.add_argument('--gamma-step', type=float, default=0.1, help='used only with StepLR and MultiStepLR')
    parser.add_argument('--milestones', default='[5, 10, 15]', help='used only with MultiStepLR')
    parser.add_argument('--gamma-exp', type=float, default=0.9, help='used only with StepLR and ExponentialLR')

    # system
    parser.add_argument('--gpu-id', default='cuda:0')
    # arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs') #@TODO: later
    parser.add_argument('--num-workers', type=int, default=0)

    # logging
    parser.add_argument('--metrics-period', type=int, default=10,
                        help='How often (num of iteration) metrics should be logged')

    # checkpoints
    parser.add_argument('--val-interval', type=int, default=1, help='Evaluate val set every N epochs')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--save-last', action='store_true', help='save last checkpoint')
    parser.add_argument('--save-best', action='store_true', help='save best checkpoint')
    parser.add_argument('--checkpoints-dir', default='', help='checkpoint dir')
    parser.add_argument('--max-keep-ckpts', type=int, default=-1, help='save last X checkpoints')
    parser.add_argument('--custom-weights', default='', help='path to custom weights path')

    # visualization settings
    parser.add_argument('--train-vis-items-path', default='', help='predictions over time on images from TRAIN')
    parser.add_argument('--val-vis-items-path',   default='', help='predictions over time on images from VAL')

    # integration with dashboard (ignore flag during local dev)
    parser.add_argument('--sly', action='store_true', help='for Supervisely App integration')

    args = parser.parse_args()
    print("Input arguments:", args)

    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    classes = sly.json.load_json_file(args.classes_path)
    num_classes = len(classes)

    model_name = model_list[args.model]["class"]
    pretrained = model_list[args.model].get("pretrained")
    if pretrained is not None:
        model = model_name(num_classes=num_classes, pretrained=True)
    else:
        # vanilla uner with random weights
        model = model_name(num_classes=num_classes)

    if torch.cuda.is_available():
        #@TODO: later can be used for bulti GPU training, now it is disabled
        if args.gpu_id:
            device_ids = [args.gpu_id]  # list(map(int, args.gpu_id.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    if num_classes == 1:
        raise RuntimeError("Train dashboard always gives min 2 classes (classX + BG)")
        #loss = LossBinary(jaccard_weight=args.jaccard_weight)
        #valid = validation_binary
    else:
        loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)
        valid = validation_multi

    cudnn.benchmark = True

    train_set = SlySegDataset(args.project_dir, args.classes_path, args.train_set_path, args.input_height, args.input_width, args.sly_augs_path)
    val_set = SlySegDataset(args.project_dir, args.classes_path, args.val_set_path,args.input_height, args.input_width, None)
    sly.logger.info("Train/Val splits", extra={"train_size": len(train_set), "val_size": len(val_set)})

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    valid_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    sly.json.dump_json_file(vars(args), os.path.join(checkpoints_dir, "train_args.json"))

    utils.train(
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        classes=classes
    )


if __name__ == '__main__':
    main()
