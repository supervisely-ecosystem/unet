import os
from torch import nn
import torch
import utils
import cv2
import numpy as np
import supervisely as sly
from torchvision import transforms

from model_list import model_list

transforms_img = transforms.Compose([
    # step0 - sly_augs will be applied here
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
])

def prepare_image_input(image, input_width, input_height):
    # RGB -> Normalized Tensor
    input = cv2.resize(image, (input_width, input_height))
    input = transforms_img(input)  # totensor + normalize
    return input


def cuda(x, device=None):
    if device is not None:
        return x.to(device)
    else:
        return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def inference(model: nn.Module, input_height, input_width, image_path, device=None):
    with torch.no_grad():
        model.eval()
    image = sly.image.read(image_path)  # RGB
    input = utils.prepare_image_input(image, input_width, input_height)
    input = torch.unsqueeze(input, 0)
    input = utils.cuda(input, device)
    output = model(input)

    image_height, image_width = image.shape[:2]
    predicted_classes_indices = output.data.cpu().numpy().argmax(axis=1)[0]
    result = cv2.resize(predicted_classes_indices, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return result


def convert_prediction_to_sly_format(predicted_class_indices, classes_json, model_classes: sly.ProjectMeta):
    height, width = predicted_class_indices.shape[:2]
    labels = []
    for idx, class_info in enumerate(classes_json):  # curr_col2cls.items():
        class_mask = predicted_class_indices == idx  # exact match (3-channel img & rgb color)
        if not np.any(class_mask):
            # 0 pixels for class
            continue
        bitmap = sly.Bitmap(data=class_mask)
        obj_class = model_classes.get_obj_class(class_info["title"])
        labels.append(sly.Label(bitmap, obj_class))
    ann = sly.Annotation(img_size=(height, width), labels=labels)
    return ann


def load_model(weights_path, num_classes, model_name, device):
    model_class = model_list[model_name]["class"]
    model = model_class(num_classes=num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def download_model_and_configs():
    ui_state = sly.json.load_json_file(os.path.join(g.local_info_dir, "ui_state.json"))
    g.model_name = ui_state["selectedModel"]
    g.input_width = ui_state["imgSize"]["width"]
    g.input_height = ui_state["imgSize"]["height"]
    sly.logger.info("Model has been successfully downloaded")


def construct_model_meta():
    g.model_classes_json = sly.json.load_json_file(g.local_model_classes_path)

    obj_classes = []
    for obj_class_json in g.model_classes_json:
        obj_classes.append(sly.ObjClass.from_json(obj_class_json))
    g.model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))

