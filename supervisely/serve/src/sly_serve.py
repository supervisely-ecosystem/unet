import os
import functools
from functools import lru_cache

import globals as g
import supervisely_lib as sly


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


@g.my_app.callback("get_model_meta")
@sly.timeit
@send_error_data
def get_model_meta(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_tags_examples")
@sly.timeit
@send_error_data
def get_tags_examples(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.labels_urls)


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "MM Classification Serve",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.tag_metas),
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})
    res_path = image_path
    if "rectangle" in state:
        image = sly.image.read(image_path)  # RGB image

        top, left, bottom, right = state["rectangle"]
        height, width = image.shape[:2]
        pad_percent = state.get("pad", 0)
        if pad_percent > 0:
            sly.logger.debug("before padding", extra={"top": top, "left": left, "right": right, "bottom": bottom})
            pad_lr = int((right - left) / 100 * pad_percent)
            pad_ud = int((bottom - top) / 100 * pad_percent)
            top = max(0, top - pad_ud)
            bottom = min(height - 1, bottom + pad_ud)
            left = max(0, left - pad_lr)
            right = min(width - 1, right + pad_lr)
            sly.logger.debug("after padding", extra={"top": top, "left": left, "right": right, "bottom": bottom})

        rect = sly.Rectangle(top, left, bottom, right)
        canvas_rect = sly.Rectangle.from_size(image.shape[:2])
        results = rect.crop(canvas_rect)
        if len(results) != 1:
            return {
                "message": "roi rectangle out of image bounds",
                "roi": state["rectangle"],
                "img_size": {"height": image.shape[0], "width": image.shape[1]}
            }
        rect = results[0]
        cropped_image = sly.image.crop(image, rect)
        res_path = os.path.join(g.my_app.data_dir, sly.rand_str(10) + sly.fs.get_file_ext(image_path))
        sly.image.write(res_path, cropped_image)

    res = nn_utils.inference_model(g.model, res_path, topn=state.get("topn", 5))
    if "rectangle" in state:
        sly.fs.silent_remove(res_path)
    return res


@g.my_app.callback("inference_image_url")
@sly.timeit
@send_error_data
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ext)
    sly.fs.download(image_url, local_image_path)
    results = inference_image_path(local_image_path, context, state, app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


@g.my_app.callback("inference_image_id")
@sly.timeit
@send_error_data
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]

    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.my_app.data_dir, f"{image_id}{sly.fs.get_file_ext(image_info.name)}")
    img = get_image_by_id(image_id)
    sly.image.write(image_path, img)

    predictions = inference_image_path(image_path, context, state, app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=predictions)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
@send_error_data
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    raise NotImplementedError("Please contact tech support")


def debug_inference():
    image_id = 903277
    image_path = f"./data/images/{image_id}.jpg"
    if not sly.fs.file_exists(image_path):
        g.my_app.public_api.image.download_path(image_id, image_path)
    res = nn_utils.inference_model(g.model, image_path, topn=5)


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


#@TODO: readme + gif - how to replace tag2urls file + release another app
if __name__ == "__main__":
    sly.main_wrapper("main", main)