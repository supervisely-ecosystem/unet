import os
from collections import namedtuple
import random
import supervisely_lib as sly
import sly_globals as g
from sly_progress_utils import get_progress_cb, reset_progress, init_progress


progress_index = 1
_images_infos = None # dataset_name -> image_name -> image_info
_cache_base_filename = os.path.join(g.my_app.data_dir, "images_info")
_cache_path = _cache_base_filename + ".db"
project_fs: sly.Project = None
_image_id_to_paths = {}


def init(data, state):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)
    init_progress(progress_index, data)
    data["done1"] = False
    state["collapsed1"] = False


@g.my_app.callback("download_project")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download(api: sly.Api, task_id, context, state, app_logger):
    try:
        if sly.fs.dir_exists(g.project_dir):
            pass
        else:
            sly.fs.mkdir(g.project_dir)
            download_progress = get_progress_cb(progress_index, "Download project", g.project_info.items_count * 2)
            sly.download_project(g.api, g.project_id, g.project_dir,
                                 cache=g.my_app.cache, progress_cb=download_progress, save_image_info=True)
            reset_progress(progress_index)

        global project_fs
        project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
    except Exception as e:
        reset_progress(progress_index)
        raise e

    fields = [
        {"field": "data.done1", "payload": True},
        {"field": "state.collapsed2", "payload": False},
        {"field": "state.disabled2", "payload": False},
        {"field": "state.activeStep", "payload": 2},
    ]
    g.api.app.set_fields(g.task_id, fields)


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
