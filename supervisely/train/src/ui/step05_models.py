import errno
import os
import requests
from pathlib import Path

import sly_globals as g
import supervisely_lib as sly
from sly_progress_utils import get_progress_cb, reset_progress, init_progress


local_weights_path = None


def get_models_list():
    res = [
        {
            "model": "UNet-classic",
            "description": "Weights are initialized randomly"
        }
    ]
    return res


def get_table_columns():
    return [
        {"key": "model", "title": "Model", "subtitle": None},
        {"key": "description", "title": "Description", "subtitle": None},
    ]


def get_model_info_by_name(name):
    models = get_models_list()
    for info in models:
        if info["model"] == name:
            return info
    raise KeyError(f"Model {name} not found")


def init(data, state):
    models = get_models_list()
    data["models"] = models
    data["modelColumns"] = get_table_columns()
    state["selectedModel"] = "UNet-classic"
    state["weightsInitialization"] = "random"  # "custom"
    state["collapsed5"] = True
    state["disabled5"] = True
    init_progress(5, data)

    state["weightsPath"] = ""# "/mmclassification/5687_synthetic products v2_003/checkpoints/epoch_10.pth"  #@TODO: for debug
    data["done5"] = False


def restart(data, state):
    data["done5"] = False


@g.my_app.callback("download_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    # global local_weights_path
    # try:
    #     if state["weightsInitialization"] == "custom":
    #         weights_path_remote = state["weightsPath"]
    #         if not weights_path_remote.endswith(".pth"):
    #             raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
    #                              f"Supported: '.pth'")
    #
    #         # get architecture type from previous UI state
    #         prev_state_path_remote = os.path.join(str(Path(weights_path_remote).parents[1]), "info/ui_state.json")
    #         prev_state_path = os.path.join(g.my_app.data_dir, "ui_state.json")
    #         api.file.download(g.team_id, prev_state_path_remote, prev_state_path)
    #         prev_state = sly.json.load_json_file(prev_state_path)
    #         api.task.set_field(g.task_id, "state.selectedModel", prev_state["selectedModel"])
    #
    #         local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
    #         if sly.fs.file_exists(local_weights_path) is False:
    #             file_info = g.api.file.get_info_by_path(g.team_id, weights_path_remote)
    #             if file_info is None:
    #                 raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path_remote)
    #             progress_cb = get_progress_cb(6, "Download weights", file_info.sizeb, is_size=True, min_report_percent=1)
    #             g.api.file.download(g.team_id, weights_path_remote, local_weights_path, g.my_app.cache, progress_cb)
    #             reset_progress(6)
    #     else:
    #         weights_url = get_pretrained_weights_by_name(state["selectedModel"])
    #         local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
    #         if sly.fs.file_exists(local_weights_path) is False:
    #             response = requests.head(weights_url, allow_redirects=True)
    #             sizeb = int(response.headers.get('content-length', 0))
    #             progress_cb = get_progress_cb(6, "Download weights", sizeb, is_size=True, min_report_percent=1)
    #             sly.fs.download(weights_url, local_weights_path, g.my_app.cache, progress_cb)
    #             reset_progress(6)
    #     sly.logger.info("Pretrained weights has been successfully downloaded",
    #                     extra={"weights": local_weights_path})
    # except Exception as e:
    #     reset_progress(6)
    #     raise e

    fields = [
        {"field": "data.done5", "payload": True},
        {"field": "state.collapsed6", "payload": False},
        {"field": "state.disabled6", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]
    g.api.app.set_fields(g.task_id, fields)