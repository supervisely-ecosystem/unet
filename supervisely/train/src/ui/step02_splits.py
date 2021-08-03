import os
import supervisely_lib as sly
import sly_globals as g

import step01_input_project

train_set = None
val_set = None

train_set_path = os.path.join(g.info_dir, "train_set.json")
val_set_path = os.path.join(g.info_dir, "val_set.json")


def init(project_info, project_meta: sly.ProjectMeta, data, state):
    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]
    data["totalImagesCount"] = project_info.items_count

    train_percent = 80
    train_count = int(project_info.items_count / 100 * train_percent)
    state["randomSplit"] = {
        "count": {
            "total": project_info.items_count,
            "train": train_count,
            "val": project_info.items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    state["splitMethod"] = "random"

    state["trainTagName"] = ""
    if project_meta.tag_metas.get("train") is not None:
        state["trainTagName"] = "train"
    state["valTagName"] = ""
    if project_meta.tag_metas.get("val") is not None:
        state["valTagName"] = "val"

    state["trainDatasets"] = []
    state["valDatasets"] = []
    state["untaggedImages"] = "train"
    state["splitInProgress"] = False
    data["trainImagesCount"] = None
    data["valImagesCount"] = None
    data["done2"] = False
    state["collapsed2"] = True
    state["disabled2"] = True


def get_train_val_sets(project_dir, state):
    split_method = state["splitMethod"]
    if split_method == "random":
        train_count = state["randomSplit"]["count"]["train"]
        val_count = state["randomSplit"]["count"]["val"]
        train_set, val_set = sly.Project.get_train_val_splits_by_count(project_dir, train_count, val_count)
        return train_set, val_set
    elif split_method == "tags":
        train_tag_name = state["trainTagName"]
        val_tag_name = state["valTagName"]
        add_untagged_to = state["untaggedImages"]
        train_set, val_set = sly.Project.get_train_val_splits_by_tag(project_dir, train_tag_name, val_tag_name,
                                                                     add_untagged_to)
        return train_set, val_set
    elif split_method == "datasets":
        train_datasets = state["trainDatasets"]
        val_datasets = state["valDatasets"]
        train_set, val_set = sly.Project.get_train_val_splits_by_dataset(project_dir, train_datasets, val_datasets)
        return train_set, val_set
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")


@g.my_app.callback("create_splits")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def create_splits(api: sly.Api, task_id, context, state, app_logger):
    step_done = False
    global train_set, val_set
    try:
        api.task.set_field(task_id, "state.splitInProgress", True)
        train_set, val_set = get_train_val_sets(g.project_dir, state)
        sly.logger.info(f"Train set: {len(train_set)} images")
        sly.logger.info(f"Val set: {len(val_set)} images")
        verify_train_val_sets(train_set, val_set)
        step_done = True
    except Exception as e:
        train_set = None
        val_set = None
        step_done = False
        raise e
    finally:
        api.task.set_field(task_id, "state.splitInProgress", False)
        fields = [
            {"field": "state.splitInProgress", "payload": False},
            {"field": f"data.done2", "payload": step_done},
            {"field": f"data.trainImagesCount", "payload": None if train_set is None else len(train_set)},
            {"field": f"data.valImagesCount", "payload": None if val_set is None else len(val_set)},
        ]
        if step_done is True:
            fields.extend([
                {"field": "state.collapsed3", "payload": False},
                {"field": "state.disabled3", "payload": False},
                {"field": "state.activeStep", "payload": 3},
            ])
        g.api.app.set_fields(g.task_id, fields)

    _save_set_to_json(step01_input_project.project_fs, train_set_path, train_set)
    _save_set_to_json(step01_input_project.project_fs, val_set_path, val_set)


def _save_set_to_json(project_fs, save_path, items):
    res = []
    for item in items:
        dataset_fs = project_fs.datasets.get(item.dataset_name)
        dataset_fs: sly.Dataset

        res.append({
            "dataset_name": item.dataset_name,
            "item_name": item.name,
            "img_path": item.img_path,
            "sly_ann_path": item.ann_path,
            "seg_mask_path": dataset_fs.get_seg_path(item.name)
        })
    sly.json.dump_json_file(res, save_path)