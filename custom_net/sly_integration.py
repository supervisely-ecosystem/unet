import numpy as np
from torch import nn
import torch
import utils
import supervisely_lib as sly

import sly_globals as g
from inference import inference
from step07_train import gallery, chart_lr, chart_loss, chart_acc
from step01_input_project import get_image_info_from_cache


_model_classes = None
_project_fs = None
_path_to_items = {}


def _convert_prediction_to_sly_format(predicted_class_indices, classes_json, model_classes: sly.ProjectMeta):
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


###############################################
######## DO NOT MODIFY THIS METHODS ###########
###############################################


def vis_inference(time_index, model: nn.Module, classes, input_height, input_width, project_dir, items_path):
    # do not modify it
    # used only in training dashboard to visualize predictions improvement over time

    # small optimization for debug
    global _model_classes, _project_fs, _path_to_items
    if _model_classes is None:
        model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection.from_json(classes))
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
        pred_ann = _convert_prediction_to_sly_format(predicted_class_indices, classes, model_meta)

        # debug raw predictions
        # colors = np.array([cls.color for cls in model_meta.obj_classes])
        # colored_mask = colors[predicted_class_indices]
        # sly.fs.ensure_base_path("/app_debug_data/debug")
        # sly.image.write(f"/app_debug_data/debug/{time_index:03d}.png", colored_mask)

        # debug predictions in sly Format
        # img_draw = sly.image.read(image_path)
        # pred_ann.draw_pretty(img_draw)
        # sly.fs.ensure_base_path("/app_debug_data/debug")
        # sly.image.write(f"/app_debug_data/debug/{time_index:03d}.png", img_draw)

        # if not gallery.has_item(item_name):
        #     image_info = get_image_info_from_cache(dataset_name, item_name)
        #     gt_ann = sly.Annotation.load_json_file(dataset_fs.get_ann_path(item_name), project_fs.meta)
        #     gallery.create_item(item_name, image_info.full_storage_url, gt_ann)
        # gallery.add_prediction(item_name, time_index, pred_ann)
    # gallery.update()


from step07_train import progress_epoch, progress_iter


def init_progress_bars(epochs, train_iters, val_iters):
    progress_epoch.set_total(epochs)
    progress_iter.set_total(train_iters + val_iters)

    progress_epoch.set(0)
    progress_iter.set(0)


def progress_set_epoch(epoch):
    progress_epoch.set(epoch, force_update=True)
    progress_iter.reset()
    progress_iter.set(0, force_update=True)


def progress_increment_iter(count):
    progress_iter.increment(count)


def report_train_metrics(epoch, iters_in_epoch, iter, lr, loss):
    x = epoch + iter / iters_in_epoch
    fields = [
        chart_lr.get_field(x, lr),
        chart_loss.get_field(x, loss, "train"),
    ]
    g.api.app.set_fields(g.task_id, fields)


def report_val_metrics(epoch, loss, avg_iou, agv_dice):
    fields = [
        chart_loss.get_field(epoch, loss, "val"),
        chart_acc.get_field(epoch, avg_iou, "avg IoU"),
        chart_acc.get_field(epoch, agv_dice, "avg Dice"),
    ]
    g.api.app.set_fields(g.task_id, fields)
