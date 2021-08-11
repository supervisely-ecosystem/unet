from torch import nn
import torch
import utils
import cv2
import numpy as np
import supervisely_lib as sly


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
    from train import model_list
    model_class = model_list[model_name]["class"]
    model = model_class(num_classes=num_classes)
    state = torch.load(weights_path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
