import numpy as np
from torch import nn
import torch
import utils
import cv2
import supervisely_lib as sly


def inference(model: nn.Module, input_height, input_width, image_path):
    with torch.no_grad():
        model.eval()
    image = sly.image.read(image_path)  # RGB
    input = utils.prepare_image_input(image, input_width, input_height)
    input = torch.unsqueeze(input, 0)
    input = utils.cuda(input)
    output = model(input)

    image_height, image_width = image.shape[:2]
    predicted_classes_indices = output.data.cpu().numpy().argmax(axis=1)[0]
    result = cv2.resize(predicted_classes_indices, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return result


    # Backup FYI: to build colored mask
    # classes - list of classes in json (order matters)
    # model_classes = [sly.ObjClass.from_json(data) for data in classes]
    # colors = np.array([cls.color for cls in model_classes])
    # colored_mask = colors[predicted_classes_indices[0]]
    # sly.image.write(save_path, colored_mask)



