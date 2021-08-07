import os
import supervisely_lib as sly
import sly_globals as g


def init(data, state):
    state["epochs"] = 500  # @TODO: 50 for debug
    state["gpusId"] = '0'

    #state["imgSize"] = 256
    state["imgSize"] = {
        "width": 512, #@TODO: 512 for debug 256 - for prod
        "height": 256,
        "proportional": True
    }
    state["batchSizePerGPU"] = 8
    state["numWorkers"] = 0  #@TODO: 0 - for debug
    state["valInterval"] = 1
    state["metricsPeriod"] = 10
    state["checkpointInterval"] = 1
    state["saveLast"] = True
    state["saveBest"] = True

    state["optimizer"] = "Adam" #"SGD"
    state["lr"] = 0.001
    state["momentum"] = 0.9
    state["weightDecay"] = 0.0001
    state["nesterov"] = False

    state["lrSchedule"] = "StepLR"
    state["stepSize"] = 5
    state["milestones"] = "[5, 10, 15]"
    state["gammaStep"] = 0.1
    state["gammaExp"] = 0.9

    state["lrPolicyEnabled"] = False

    #@TODO: LR policy
    # file_path = os.path.join(g.root_source_dir, "supervisely/train/configs/lr_policy.py")
    # with open(file_path) as f:
    #     state["lrPolicyPyConfig"] = f.read()
    state["lrPolicyPyConfig"] = ""

    # visualization settings
    state["trainVisCount"] = 1
    state["valVisCount"] = 1

    state["collapsed6"] = True
    state["disabled6"] = True
    state["done6"] = False


def restart(data, state):
    data["done6"] = False


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 (for this specific UNet-based models) and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    input_height = state["imgSize"]["height"]
    input_width = state["imgSize"]["width"]

    if not check_crop_size(input_height, input_width):
        raise ValueError('Input image sizes should be divisible by 32, but train '
                         'sizes (H x W : {train_crop_height} x {train_crop_width}) '
                         'are not.'.format(train_crop_height=input_height, train_crop_width=input_width))

    if not check_crop_size(input_height, input_width):
        raise ValueError('Input image sizes should be divisible by 32, but validation '
                         'sizes (H x W : {val_crop_height} x {val_crop_width}) '
                         'are not.'.format(val_crop_height=input_height, val_crop_width=input_width))

    fields = [
        {"field": "data.done6", "payload": True},
        {"field": "state.collapsed7", "payload": False},
        {"field": "state.disabled7", "payload": False},
        {"field": "state.activeStep", "payload": 7},
    ]
    g.api.app.set_fields(g.task_id, fields)
