import os
from collections import namedtuple
import random
import supervisely as sly
from supervisely.app.v1.widgets.progress_bar import ProgressBar

import sly_globals as g


_images_infos = None  # dataset_name -> image_name -> image_info
_cache_base_filename = os.path.join(g.my_app.data_dir, "images_info")
_cache_path = _cache_base_filename + ".db"
project_fs: sly.Project = None
_image_id_to_paths = {}
progress1 = ProgressBar(g.task_id, g.api, "data.progress1", "Download project from server")


def init(data, state):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)
    progress1.init_data(data)
    data["done1"] = False
    data['downloadInProgress'] = False
    state["collapsed1"] = False


@g.my_app.callback("download_project")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download(api: sly.Api, task_id, context, state, app_logger):
    if g.download_in_progress:
        return
    g.download_in_progress = True
    g.api.app.set_fields(g.task_id, [{"field": "data.downloadInProgress", "payload": True}])
    try:
        if sly.fs.dir_exists(g.project_dir):
            pass
        else:
            sly.fs.mkdir(g.project_dir)
            progress1.set_total(g.project_info.items_count * 2)
            sly.download_project(g.api, g.project_id, g.project_dir,
                                 cache=g.my_app.cache, progress_cb=progress1.increment, save_image_info=True)
            progress1.reset_and_update()
        global project_fs
        project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
    except Exception as e:
        progress1.reset_and_update()
        raise e

    fields = [
        {"field": "data.done1", "payload": True},
        {"field": "data.downloadInProgress", "payload": False},
        {"field": "state.collapsed2", "payload": False},
        {"field": "state.disabled2", "payload": False},
        {"field": "state.activeStep", "payload": 2},
    ]
    g.api.app.set_fields(g.task_id, fields)
    g.download_in_progress = False


def get_image_info_from_cache(dataset_name, item_name):
    global project_fs
    if project_fs is None:
        # for debug step07 without running all previous steps
        project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)

    dataset_fs = project_fs.datasets.get(dataset_name)
    img_info_path = dataset_fs.get_img_info_path(item_name)
    image_info_dict = sly.json.load_json_file(img_info_path)
    ImageInfo = namedtuple('ImageInfo', image_info_dict)
    info = ImageInfo(**image_info_dict)

    # add additional info - helps to save split paths to txt files
    _image_id_to_paths[info.id] = dataset_fs.get_item_paths(item_name)._asdict()

    return info


def get_paths_by_image_id(image_id):
    return _image_id_to_paths[image_id]


def get_random_item():
    global project_fs
    all_ds_names = project_fs.datasets.keys()
    ds_name = random.choice(all_ds_names)
    ds = project_fs.datasets.get(ds_name)
    items = list(ds)
    item_name = random.choice(items)
    return ds_name, item_name
