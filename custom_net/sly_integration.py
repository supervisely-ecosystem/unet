import step07_train
import supervisely_lib as sly
from inference import inference
import numpy as np
from torch import nn
import torch
import utils
import supervisely_lib as sly
from step07_train import gallery
from step01_input_project import get_image_info_from_cache


_model_classes = None
_project_fs = None
_path_to_items = {}


def _convert_prediction_to_sly_format(predicted_class_indices, classes_json, model_classes: sly.ProjectMeta):
    height, width = predicted_class_indices.shape[:2]
    labels = []
    for idx, class_info in enumerate(classes_json):  # curr_col2cls.items():
        class_mask = np.all(predicted_class_indices == idx, axis=2)  # exact match (3-channel img & rgb color)
        if not np.any(class_mask):
            # 0 pixels for class
            continue
        bitmap = sly.Bitmap(data=class_mask)
        obj_class = model_classes.get_obj_class(class_info["title"])
        labels.append(sly.Label(bitmap, obj_class))
    ann = sly.Annotation(img_size=(height, width), labels=labels)
    return ann


def vis_inference(time_index, model: nn.Module, classes, input_height, input_width, project_dir, items_path):
    # do not modify it
    # used only in training dashboard to visualize predictions improvement over time

    # small optimization for debug
    global _model_classes, _project_fs, _path_to_items
    if _model_classes is None:
        model_classes = sly.ProjectMeta(obj_classes=sly.ObjClassCollection.from_json(classes))
    if _project_fs is None:
        project_fs = sly.Project(project_dir, sly.OpenMode.READ)
    if items_path not in _path_to_items:
        _path_to_items[items_path] = sly.json.load_json_file(items_path)
    items = _path_to_items[items_path]

    for item in items:
        dataset_name = item["dataset_name"]
        item_name = item["item_name"]

        dataset_fs = project_fs.datasets.get(dataset_name)
        dataset_fs: sly.Dataset

        image_path = dataset_fs.get_img_path(item_name)
        predicted_class_indices = inference(model, input_height, input_width, image_path)
        pred_ann = _convert_prediction_to_sly_format(predicted_class_indices, classes, model_classes)

        if not gallery.has_item(item_name):
            image_info = get_image_info_from_cache(dataset_name, item_name)
            gt_ann = sly.Annotation.load_json_file(dataset_fs.get_ann_path(item_name), project_fs.meta)
            gallery.create_item(item_name, image_info.full_storage_url, gt_ann)
        gallery.add_prediction(item_name, time_index, pred_ann)
    gallery.update()
