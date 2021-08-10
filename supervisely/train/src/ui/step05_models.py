import errno
import os
import requests
from pathlib import Path

import sly_globals as g
import supervisely_lib as sly
progress5 = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress5", "Download weights", is_size=True, min_report_percent=5)

local_weights_path = None


def get_models_list():
    from train import model_list
    res = []
    for name, data in model_list.items():
        res.append({
            "model": name,
            "description": data["description"]
        })
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
    state["selectedModel"] = models[0]["model"]
    state["weightsInitialization"] = "random"  # "custom"
    state["collapsed5"] = True
    state["disabled5"] = True

    progress5.init_data(data)

    state["weightsPath"] = ""
    data["done5"] = False


def restart(data, state):
    data["done5"] = False


@g.my_app.callback("download_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    #"https://download.pytorch.org/models/vgg11-8a719046.pth" to /root/.cache/torch/hub/checkpoints/vgg11-8a719046.pth
    from train import model_list

    global local_weights_path
    try:
        if state["weightsInitialization"] == "custom":
            weights_path_remote = state["weightsPath"]
            if not weights_path_remote.endswith(".pth"):
                raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                                 f"Supported: '.pth'")

            # get architecture type from previous UI state
            prev_state_path_remote = os.path.join(str(Path(weights_path_remote).parents[1]), "info/ui_state.json")
            prev_state_path = os.path.join(g.my_app.data_dir, "ui_state.json")
            api.file.download(g.team_id, prev_state_path_remote, prev_state_path)
            prev_state = sly.json.load_json_file(prev_state_path)
            api.task.set_field(g.task_id, "state.selectedModel", prev_state["selectedModel"])

            local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
            if sly.fs.file_exists(local_weights_path) is False:
                file_info = g.api.file.get_info_by_path(g.team_id, weights_path_remote)
                if file_info is None:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path_remote)
                progress5.set_total(file_info.sizeb)
                g.api.file.download(g.team_id, weights_path_remote, local_weights_path, g.my_app.cache, progress5.increment)
                progress5.reset_and_update()
        else:
            weights_url = model_list[state["selectedModel"]].get("pretrained")
            if weights_url is not None:
                default_pytorch_dir = "/root/.cache/torch/hub/checkpoints/"
                #local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
                local_weights_path = os.path.join(default_pytorch_dir, sly.fs.get_file_name_with_ext(weights_url))
                if sly.fs.file_exists(local_weights_path) is False:
                    response = requests.head(weights_url, allow_redirects=True)
                    sizeb = int(response.headers.get('content-length', 0))
                    progress5.set_total(sizeb)
                    sly.fs.download(weights_url, local_weights_path, g.my_app.cache, progress5.increment)
                    progress5.reset_and_update()
                sly.logger.info("Pretrained weights has been successfully downloaded",
                                extra={"weights": local_weights_path})
    except Exception as e:
        progress5.reset_and_update()
        raise e

    fields = [
        {"field": "data.done5", "payload": True},
        {"field": "state.collapsed6", "payload": False},
        {"field": "state.disabled6", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]
    g.api.app.set_fields(g.task_id, fields)


def restart(data, state):
    data["done5"] = False