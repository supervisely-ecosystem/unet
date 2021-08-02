import os
import supervisely_lib as sly
import sly_globals as g
#from tags import get_random_image
from supervisely_lib.app.widgets import CompareGallery
import validate_training_data as td
import train_config


_templates = [
    {
        "config": "supervisely/train/augs/mmclass-lite.json",
        "name": "Lite (color + rotate)",
    },
    {
        "config": "supervisely/train/augs/mmclass-lite-with-fliplr.json",
        "name": "Lite + fliplr",
    },
    {
        "config": "supervisely/train/augs/mmclass-heavy-no-fliplr.json",
        "name": "Heavy",
    },
    {
        "config": "supervisely/train/augs/mmclass-heavy-with-fliplr.json",
        "name": "Heavy + fliplr",
    },
]

_custom_pipeline_path = None
custom_pipeline = None
gallery1: CompareGallery = None
gallery2: CompareGallery = None
remote_preview_path = "/temp/preview_augs.jpg"

augs_json_config = None
augs_py_preview = None
augs_config_path = None


def _load_template(json_path):
    config = sly.json.load_json_file(json_path)
    pipeline = sly.imgaug_utils.build_pipeline(config["pipeline"], random_order=config["random_order"])  # to validate
    py_code = sly.imgaug_utils.pipeline_to_python(config["pipeline"], config["random_order"])

    global augs_json_config, augs_py_preview
    augs_json_config = config
    augs_py_preview = py_code

    return pipeline, py_code


def get_aug_templates_list():
    pipelines_info = []
    name_to_py = {}
    for template in _templates:
        json_path = os.path.join(g.root_source_dir, template["config"])
        _, py_code = _load_template(json_path)
        pipelines_info.append({
            **template,
            "py": py_code
        })
        name_to_py[template["name"]] = py_code
    return pipelines_info, name_to_py


def get_template_by_name(name):
    for template in _templates:
        if template["name"] == name:
            json_path = os.path.join(g.root_source_dir, template["config"])
            pipeline, _ = _load_template(json_path)
            return pipeline
    raise KeyError(f"Template \"{name}\" not found")


def init(data, state):
    state["useAugs"] = True
    state["augsType"] = "template"
    templates_info, name_to_py = get_aug_templates_list()
    data["augTemplates"] = templates_info
    data["augPythonCode"] = name_to_py
    state["augsTemplateName"] = templates_info[0]["name"]

    data["pyViewOptions"] = {
        "mode": 'ace/mode/python',
        "showGutter": False,
        "readOnly": True,
        "maxLines": 100,
        "highlightActiveLine": False
    }

    state["customAugsPath"] = ""  # "/mmclass-heavy-no-fliplr.json"  # @TODO: for debug
    data["customAugsPy"] = None

    global gallery1, gallery2
    gallery1 = CompareGallery(g.task_id, g.api, "data.gallery1", g.project_meta)
    data["gallery1"] = gallery1.to_json()
    gallery2 = CompareGallery(g.task_id, g.api, "data.gallery2", g.project_meta)
    data["gallery2"] = gallery2.to_json()
    state["collapsed5"] = True
    state["disabled5"] = True
    data["done5"] = False


def restart(data, state):
    data["done5"] = False


@g.my_app.callback("load_existing_pipeline")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def load_existing_pipeline(api: sly.Api, task_id, context, state, app_logger):
    global _custom_pipeline_path, custom_pipeline

    api.task.set_field(task_id, "data.customAugsPy", None)

    remote_path = state["customAugsPath"]
    _custom_pipeline_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(remote_path))
    api.file.download(g.team_id, remote_path, _custom_pipeline_path)

    custom_pipeline, py_code = _load_template(_custom_pipeline_path)
    api.task.set_field(task_id, "data.customAugsPy", py_code)


@g.my_app.callback("preview_augs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def preview_augs(api: sly.Api, task_id, context, state, app_logger):
    global gallery1, gallery2
    image_info = td.get_random_image()
    if state["augsType"] == "template":
        gallery = gallery1
        augs_ppl = get_template_by_name(state["augsTemplateName"])
    else:
        gallery = gallery2
        augs_ppl = custom_pipeline

    img = api.image.download_np(image_info.id)
    ann_json = api.annotation.download(image_info.id).annotation
    ann = sly.Annotation.from_json(ann_json, g.project_meta)
    gallery.set_left("before", image_info.full_storage_url, ann)
    _, res_img, res_ann = sly.imgaug_utils.apply(augs_ppl, g.project_meta, img, ann)
    local_image_path = os.path.join(g.my_app.data_dir, "preview_augs.jpg")
    sly.image.write(local_image_path, res_img)
    if api.file.exists(g.team_id, remote_preview_path):
        api.file.remove(g.team_id, remote_preview_path)
    file_info = api.file.upload(g.team_id, local_image_path, remote_preview_path)
    gallery.set_right("after", file_info.full_storage_url, res_ann)
    gallery.update(options=False)


@g.my_app.callback("use_augs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_augs(api: sly.Api, task_id, context, state, app_logger):
    global augs_config_path

    if state["useAugs"] is True:
        augs_config_path = os.path.join(train_config.configs_dir, "augs_config.json")
        sly.json.dump_json_file(augs_json_config, augs_config_path)

        augs_py_path = os.path.join(train_config.configs_dir, "augs_preview.py")
        with open(augs_py_path, 'w') as f:
            f.write(augs_py_preview)
    else:
        augs_config_path = None

    fields = [
        {"field": "data.done5", "payload": True},
        {"field": "state.collapsed6", "payload": False},
        {"field": "state.disabled6", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]
    g.api.app.set_fields(g.task_id, fields)
