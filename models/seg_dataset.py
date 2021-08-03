from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image


class SlySegDataset(Dataset):
    def __init__(self, count, transform=None):
        #self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def get_dataset(size=256):
    trans = transforms.Compose([
        transforms.Resize(size=size, interpolation=Image.NEAREST),
        # augmentations here
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
    ])
    return SlySegDataset()
