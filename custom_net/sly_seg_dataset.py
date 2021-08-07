from torch.utils.data import Dataset
from torchvision import transforms
import supervisely_lib as sly
import numpy as np


class SlySegDataset(Dataset):
    def __init__(self, project_dir, model_classes_path, split_path, input_height, input_width, sly_augs=None):
        self.project_fs = sly.Project(project_dir, sly.OpenMode.READ)

        self.input_items = sly.json.load_json_file(split_path)
        self.model_classes_path = model_classes_path
        model_classes_json = sly.json.load_json_file(model_classes_path)
        self.model_classes = [sly.ObjClass.from_json(data) for data in model_classes_json]

        #self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.sly_augs = sly_augs

        self.transforms_img = transforms.Compose([
            # step0 - sly_augs will be applied here
            transforms.ToTensor(),
            transforms.Resize(size=(input_height, input_width), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
        ])
        self.transforms_ann = transforms.Compose([
            # step0 - sly_augs will be applied here
            transforms.ToTensor(),
            transforms.Resize(size=(input_height, input_width), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.input_items)

    def __getitem__(self, idx):
        dataset_name = self.input_items[idx]["dataset_name"]
        item_name = self.input_items[idx]["item_name"]

        dataset_fs = self.project_fs.datasets.get(dataset_name)
        dataset_fs: sly.Dataset

        image_path = dataset_fs.get_img_path(item_name)
        mask_path = dataset_fs.get_seg_path(item_name)

        image = sly.image.read(image_path)  # RGB
        color_mask = sly.image.read(mask_path)    # RGB
        seg_mask = self._colors_to_indices(color_mask)

        if self.sly_augs:
            image, mask = self.sly_augs(image, seg_mask)

        image = self.transforms_img(image)
        seg_mask = self.transforms_ann(seg_mask)
        return [image, seg_mask]

    def _colors_to_indices(self, color_mask):
        # output shape - [height, width]
        height, width = color_mask.shape[:2]
        seg_mask = np.zeros((height, width), dtype=np.uint8)
        for class_index, obj_class in enumerate(self.model_classes):
            seg_mask[color_mask == obj_class.color] = class_index
        return seg_mask
