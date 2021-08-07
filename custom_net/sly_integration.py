import supervisely_lib as sly
from inference import inference
import numpy as np
from torch import nn
import torch
import utils
import supervisely_lib as sly


def _convert_prediction_to_sly_format(predicted_class_indices, classes_json, model_classes: sly.ProjectMeta):
    height, width = predicted_class_indices.shape[:2]

    labels = []
    for idx, class_info in enumerate(classes_json):  # curr_col2cls.items():
        class_mask = np.all(predicted_class_indices == idx, axis=2)  # exact match (3-channel img & rgb color)
        bitmap = sly.Bitmap(data=class_mask)
        obj_class = model_classes.get_obj_class(class_info["name"])
        labels.append(sly.Label(bitmap, obj_class))
    ann = sly.Annotation(img_size=(height, width), labels=labels)
    return ann


def vis_inference(model: nn.Module, classes, input_height, input_width, project_dir, items_path):
    # do not modify it
    # used only in training dashboard to visualize predictions improvement over time

    model_classes = sly.ProjectMeta(obj_classes=sly.ObjClassCollection.from_json(classes))
    project_fs = sly.Project(project_dir, sly.OpenMode.READ)
    items = sly.json.load_json_file(items_path)

    for item in items:
        dataset_name = item["dataset_name"]
        item_name = item["item_name"]

        dataset_fs = project_fs.datasets.get(dataset_name)
        dataset_fs: sly.Dataset

        image_path = dataset_fs.get_img_path(item_name)
        predicted_class_indices = inference(model, input_height, input_width, image_path)
        ann = _convert_prediction_to_sly_format(predicted_class_indices, classes, model_classes)


