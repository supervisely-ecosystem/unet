import os
import supervisely_lib as sly
import sly_globals as g


def init(data, state):
    state["epochs"] = 5
    state["gpusId"] = '0'

    state["imgSize"] = 256
    state["batchSizePerGPU"] = 8
    state["workersPerGPU"] = 0  #@TODO: 0 - for debug
    state["valInterval"] = 1
    state["metricsPeriod"] = 10
    state["checkpointInterval"] = 1
    state["saveLast"] = True
    state["saveBest"] = True

    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["momentum"] = 0.9
    state["weightDecay"] = 0.0001
    state["nesterov"] = False

    #state["gradClipEnabled"] = False
    #state["maxNorm"] = 1

    state["lrPolicyEnabled"] = False

    #@TODO: LR policy
    # file_path = os.path.join(g.root_source_dir, "supervisely/train/configs/lr_policy.py")
    # with open(file_path) as f:
    #     state["lrPolicyPyConfig"] = f.read()
    state["lrPolicyPyConfig"] = ""

    state["collapsed6"] = True
    state["disabled6"] = True
    state["done6"] = False


def restart(data, state):
    data["done6"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done6", "payload": True},
        {"field": "state.collapsed7", "payload": False},
        {"field": "state.disabled7", "payload": False},
        {"field": "state.activeStep", "payload": 7},
    ]
    g.api.app.set_fields(g.task_id, fields)
