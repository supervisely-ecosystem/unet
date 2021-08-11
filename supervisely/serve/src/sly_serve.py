import os
import functools
from functools import lru_cache
import supervisely_lib as sly

import globals as g
from inference import inference
from sly_integration import _convert_prediction_to_sly_format


@lru_cache(maxsize=10)
def get_image_by_id(image_id):
    img = g.api.image.download_np(image_id)
    return img


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value
    return wrapper


# send model meta (classes and tags that model predicts / produces)
@my_app.callback("get_output_classes_and_tags")
@sly.timeit
@send_error_data
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    my_app.send_response(request_id, data=g.meta.to_json())


# send information about deployed model and session, the structure is not defined,
# it can be anything that is JSON serrializable
@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "UNet Serve",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


# some models may have specific inference settings like confidence threshold, iou threshold and so on.
# Current segmentation model doen't have them so we will return empty dict
@my_app.callback("get_custom_inference_settings")
@sly.timeit
@send_error_data
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    my_app.send_response(request_id, data={"settings": {}})


def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})
    pred = inference(g.model, g.input_height, g.input_width, image_path) # mask with class indices
    ann: sly.Annotation = _convert_prediction_to_sly_format(pred, g.model_classes_json, g.model_meta)
    return ann.to_json()


@my_app.callback("inference_image_url")
@sly.timeit
@send_error_data
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})

    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(my_app.data_dir, sly.rand_str(15) + ext)

    sly.fs.download(image_url, local_image_path)
    ann_json = inference_image_path(local_image_path, context, state, app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    my_app.send_response(request_id, data=ann_json)


@my_app.callback("inference_image_id")
@sly.timeit
@send_error_data
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(my_app.data_dir, sly.rand_str(10) + image_info.name)
    api.image.download_path(image_id, image_path)
    ann_json = inference_image_path(image_path, context, state, app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    my_app.send_response(request_id, data=ann_json)


@my_app.callback("inference_batch_ids")
@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = []
    for info in infos:
        paths.append(os.path.join(my_app.data_dir, sly.rand_str(10) + info.name))
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    results = []
    for image_path in paths:
        ann_json = inference_image_path(image_path, context, state, app_logger)
        results.append(ann_json)
        sly.fs.silent_remove(image_path)

    request_id = context["request_id"]
    my_app.send_response(request_id, data=results)


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()
    nn_utils.deploy_model()

    g.my_app.run()


if __name__ == "__main__":
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()

    sly.main_wrapper("main", main)