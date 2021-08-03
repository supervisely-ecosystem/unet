import argparse
import supervisely_lib as sly

from sly_seg_dataset import SlySegDataset


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')

    parser.add_argument('--project-dir', default='', help='path to sly project with masks')
    parser.add_argument('--classes-path', default='', help='path to the list of classes (order matters)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=256, help='model input image size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='number of dataloader workers')
    parser.add_argument('--sly', action='store_true', help='for Supervisely App integration')

    opt = parser.parse_args()
    print("Input arguments:", opt)


# for debug
# --data coco128.yaml --cfg yolov5s.yaml --weights yolov5s.pt --batch-size 64
if __name__ == '__main__':
    dataset = SlySegDataset(
        project_dir="/app_debug_data/data/Lemons (Annotated)_seg",
        model_classes_path="/app_debug_data/data/artifacts/info/model_classes.json",
        split_path="/app_debug_data/data/artifacts/info/train_set.json",
        input_size=256,
    )
    from torch.utils.data import Dataset, DataLoader
    dataloader = DataLoader(dataset, batch_size=7, shuffle=True, num_workers=0)
    inputs, masks = next(iter(dataloader))
    print(inputs.shape, masks.shape)
    for x in [inputs.numpy(), masks.numpy()]:
        print(x.min(), x.max(), x.mean(), x.std())

    #main()