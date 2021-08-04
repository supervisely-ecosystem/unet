import argparse
import supervisely_lib as sly

from sly_seg_dataset import SlySegDataset


def main():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--gpus-id', default='cuda:0')
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


# for debug
# --data coco128.yaml --cfg yolov5s.yaml --weights yolov5s.pt --batch-size 64
# https://stackoverflow.com/questions/21986194/how-to-pass-dictionary-items-as-function-arguments-in-python/21986301


def f(a, b, c, d):
    print(f"a={a}, b={b}, c={c}, d={d}")


if __name__ == '__main__':
    main()
    exit(0)

    params = {
        "b": 2,
        "c": 3,
        "d": 4
    }
    f(1, **params)

    exit(0)
    dataset = SlySegDataset(
        project_dir="/app_debug_data/data/Lemons (Annotated)_seg",
        model_classes_path="/app_debug_data/data/artifacts/info/model_classes.json",
        split_path="/app_debug_data/data/artifacts/info/train_set.json",
        input_size=256,
    )
    from torch.utils.data import Dataset, DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    inputs, masks = next(iter(dataloader))
    print(inputs.shape, masks.shape)
    for x in [inputs.numpy(), masks.numpy()]:
        print(x.min(), x.max(), x.mean(), x.std())

    #main()