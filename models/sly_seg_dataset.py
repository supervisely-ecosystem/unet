from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
import supervisely_lib as sly


class SlySegDataset(Dataset):
    def __init__(self, project_dir, split_path, input_size, sly_augs=None):
        self.input_items = sly.json.load_json_file(split_path)

        #self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.sly_augs = sly_augs

        self.transforms_img = transforms.Compose([
            # step0 - sly_augs will be applied here
            transforms.Resize(size=input_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
        ])
        self.transforms_ann = transforms.Compose([
            # step0 - sly_augs will be applied here
            transforms.Resize(size=input_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.input_items)

    def __getitem__(self, idx):
        image_path = self.input_items[idx]["img_path"]
        mask_path = self.input_items[idx]["mask_path"]

        image = sly.image.read(image_path)  # RGB
        mask = sly.image.read(mask_path)

        if self.sly_augs:
            image, mask = self.sly_augs(image, mask)

        image = self.transforms_img(image)
        mask = self.transforms_ann(image)

        return [image, mask]

