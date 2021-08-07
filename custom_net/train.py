import argparse
import json
import os
from pathlib import Path
from validation import validation_binary, validation_multi

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16, AlbuNet
from loss import LossBinary, LossMulti
#from dataset import RoboticsDataset
import utils
import sys
#from prepare_train_val import get_split

import supervisely_lib as sly
from sly_seg_dataset import SlySegDataset


from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)


model_list = {
    'UNet11': UNet11,
    'UNet16': UNet16,
    'UNet': UNet,
    'AlbuNet': AlbuNet,
    'LinkNet34': LinkNet34
}


def main():
    # arg('--train_crop_height', type=int, default=1024)
    # arg('--train_crop_width', type=int, default=1280)
    # arg('--val_crop_height', type=int, default=1024)
    # arg('--val_crop_width', type=int, default=1280)
    # arg('--model', type=str, default='UNet', choices=moddel_list.keys())

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

    # integration with dashboard (ignore flag during local dev)
    parser.add_argument('--sly', action='store_true', help='for Supervisely App integration')

    args = parser.parse_args()
    print("Input arguments:", args)

    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    classes = sly.json.load_json_file(args.classes_path)
    num_classes = len(classes)

    #@TODO: model selector later
    #if args.model == 'UNet':
    model = UNet(num_classes=num_classes)
    # else:
    #     model_name = moddel_list[args.model]
    #     model = model_name(num_classes=num_classes, pretrained=True)

    if torch.cuda.is_available():
        #@TODO: later can be used for bulti GPU training, now it is disabled
        if args.gpu_id:
            device_ids = [args.gpu_id] #list(map(int, args.gpu_id.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    if num_classes == 1:
        raise RuntimeError("Train dashboard always gives min 2 classes (x + gb)")
        #loss = LossBinary(jaccard_weight=args.jaccard_weight)
        #valid = validation_binary
    else:
        loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)
        valid = validation_multi

    cudnn.benchmark = True

    train_set = SlySegDataset(args.project_dir, args.classes_path, args.train_set_path, args.input_height, args.input_width)
    val_set = SlySegDataset(args.project_dir, args.classes_path, args.val_set_path,args.input_height, args.input_width)
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

    # def train_transform(p=1):
    #     return Compose([
    #         PadIfNeeded(min_height=args.train_crop_height, min_width=args.train_crop_width, p=1),
    #         RandomCrop(height=args.train_crop_height, width=args.train_crop_width, p=1),
    #         VerticalFlip(p=0.5),
    #         HorizontalFlip(p=0.5),
    #         Normalize(p=1)
    #     ], p=p)
    #
    # def val_transform(p=1):
    #     return Compose([
    #         PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
    #         CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
    #         Normalize(p=1)
    #     ], p=p)

    # train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=args.type,
    #                            batch_size=args.batch_size)
    # valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type,
    #                            batch_size=len(device_ids))

    sly.json.dump_json_file(vars(args), os.path.join(checkpoints_dir, "train_args.json"))

    utils.train(
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()
