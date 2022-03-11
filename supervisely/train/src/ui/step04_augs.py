import os
import supervisely as sly
import sly_globals as g
from supervisely.app.v1.widgets.compare_gallery import CompareGallery
import step01_input_project
import step03_classes


_templates = [
    {
        "config": "supervisely/train/augs/seg_simple.json",
        "name": "Only Color",
    },
    {
        "config": "supervisely/train/augs/seg_simple_rotate.json",
        "name": "Light: color + rotate",
    },
    {
        "config": "supervisely/train/augs/seg_simple_rotate_crop.json",
        "name": "Medium: light + crop",
    },
    {
        "config": "supervisely/train/augs/seg_simple_rotate_crop_flipLR.json",
        "name": "Heavy: medium + flipLR",
    },
    {
        "config": "supervisely/train/augs/seg_simple_rotate_crop_flipLR_flipUD.json",
        "name": "Heavy + flipUD",
    }
]

_custom_pipeline_path = None
custom_pipeline = None
gallery1: CompareGallery = None
gallery2: CompareGallery = None
remote_preview_path = "/temp/unet/preview_augs.jpg"

augs_json_config = None
augs_py_preview = None
augs_config_path = os.path.join(g.info_dir, "augs_config.json")


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
    state["collapsed4"] = True
    state["disabled4"] = True
    data["done4"] = False


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

    ds_name, item_name = step01_input_project.get_random_item()
    image_info = step01_input_project.get_image_info_from_cache(ds_name, item_name)

    if state["augsType"] == "template":
        gallery = gallery1
        augs_ppl = get_template_by_name(state["augsTemplateName"])
    else:
        gallery = gallery2
        augs_ppl = custom_pipeline

    # old implementation - download from server
    #img = api.image.download_np(image_info.id)
    #ann_json = api.annotation.download(image_info.id).annotation
    #ann = sly.Annotation.from_json(ann_json, g.project_meta)
    #ann = ann.filter_labels_by_classes(keep_classes=step03_classes.selected_classes)

    # new implementation - read from local directory
    dataset_fs: sly.Dataset = step01_input_project.project_fs.datasets.get(ds_name)
    img_path = dataset_fs.get_img_path(item_name)
    ann_path = dataset_fs.get_ann_path(item_name)
    img = sly.image.read(img_path) # RGB
    ann = sly.Annotation.load_json_file(ann_path, step01_input_project.project_fs.meta)
    ann = ann.filter_labels_by_classes(keep_classes=step03_classes.selected_classes)

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
        sly.json.dump_json_file(augs_json_config, augs_config_path)
        augs_py_path = os.path.join(g.info_dir, "augs_preview.py")
        with open(augs_py_path, 'w') as f:
            f.write(augs_py_preview)

    fields = [
        {"field": "data.done4", "payload": True},
        {"field": "state.collapsed5", "payload": False},
        {"field": "state.disabled5", "payload": False},
        {"field": "state.activeStep", "payload": 5},
    ]
    g.api.app.set_fields(g.task_id, fields)


def restart(data, state):
    data["done4"] = False