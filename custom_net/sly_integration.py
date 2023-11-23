import numpy as np
from torch import nn
import supervisely_lib as sly

from inference import inference, convert_prediction_to_sly_format
from step07_train import gallery, chart_lr, chart_loss, chart_acc
from step01_input_project import get_image_info_from_cache


_model_classes = None
_project_fs = None
_path_to_items = {}
label_id = 1

###############################################
######## MODIFY METHODS BELOW ONLY IF #########
######## YOU PERFORM DEEP INTEGRATION #########
###############################################

def get_visualization_step(epochs):
    total_visualizations_count = 20

    vis_step = int(epochs / total_visualizations_count) \
        if int(epochs / total_visualizations_count) > 0 else 1

    return vis_step


def vis_inference(time_index, model: nn.Module, classes, input_height, input_width, project_dir, items_path, update=False):
    import sly_globals as g
    # do not modify it
    # used only in training dashboard to visualize predictions improvement over time

    # small optimization for debug
    global _model_classes, _project_fs, _path_to_items, label_id
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
        pred_ann = convert_prediction_to_sly_format(predicted_class_indices, classes, model_meta)

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

        if not gallery.has_item(item_name):
            image_info = get_image_info_from_cache(dataset_name, item_name)
            gt_ann_path = dataset_fs.get_ann_path(item_name)
            gt_ann_json = sly.json.load_json_file(gt_ann_path)
            for label in gt_ann_json["objects"]:
                label['id'] = local_labels_id
                local_labels_id += 1
            gt_ann = sly.Annotation.from_json(gt_ann_json, project_fs.meta)
            gallery.create_item(item_name, image_info.path_original, gt_ann)

        pred_ann_json = pred_ann.to_json()
        for label in pred_ann_json["objects"]:
            label['id'] = local_labels_id
            local_labels_id += 1
        pred_ann = sly.Annotation.from_json(pred_ann_json, model_meta)
        print(time_index, item_name)
        gallery.add_prediction(item_name, time_index, pred_ann)

    if update is True:
        gallery.update()
        if gallery.is_show_last_time():
            gallery.show_last_time_index()
            g.api.task.set_field(g.task_id, "state.visEpoch", time_index)


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
    import sly_globals as g
    x = epoch + iter / iters_in_epoch
    fields = [
        chart_lr.get_field(x, lr),
        chart_loss.get_field(x, loss, "train"),
    ]
    g.api.app.set_fields(g.task_id, fields)


def report_val_metrics(epoch, loss, avg_iou, agv_dice):
    import sly_globals as g
    fields = [
        chart_loss.get_field(epoch, loss, "val"),
        chart_acc.get_field(epoch, avg_iou, "avg IoU"),
        chart_acc.get_field(epoch, agv_dice, "avg Dice"),
        {"field": f"data.finishedEpoch", "payload": epoch},
    ]
    g.api.app.set_fields(g.task_id, fields)
