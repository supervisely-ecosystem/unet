from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image


class SlySegDataset(Dataset):
    def __init__(self, size, sly_augs=None):
        #self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.sly_augs = sly_augs

        self.transforms_img = transforms.Compose([
            transforms.Resize(size=size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
        ])
        self.transforms_ann = transforms.Compose([
            transforms.Resize(size=size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]

        if self.sly_augs:
            image, mask = self.sly_augs(image, mask)

        image = self.transforms_img(image)
        mask = self.transforms_ann(image)

        return [image, mask]

