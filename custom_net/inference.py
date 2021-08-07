import numpy as np
from torch import nn
import torch
import utils
import supervisely_lib as sly


def inference(model: nn.Module, classes, input_height, input_width, image_path):
    with torch.no_grad():
        model.eval()

    image = sly.image.read(image_path)  # RGB
    input = utils.transforms_img(image)
    input = utils.cuda(input)
    output = model(input)

    output_classes = output.data.cpu().numpy().argmax(axis=1)


