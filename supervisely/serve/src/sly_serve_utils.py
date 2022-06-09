import os
import supervisely_lib as sly
import globals as g


@sly.timeit
def download_model_and_configs():
    if not g.remote_weights_path.endswith(".pth"):
        raise ValueError(f"Unsupported weights extension {sly.fs.get_file_ext(g.remote_weights_path)}. "
                         f"Supported extension: '.pth'")

    info = g.api.file.get_info_by_path(g.team_id, g.remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {g.remote_weights_path}")

    progress = sly.Progress("Downloading weights", info.sizeb, is_size=True, need_info_log=True)
    g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(g.remote_weights_path))
    g.api.file.download(
        g.team_id,
        g.remote_weights_path,
        g.local_weights_path,
        progress_cb=progress.iters_done_report
    )

    def _download_dir(remote_dir, local_dir):
        remote_files = g.api.file.list2(g.team_id, remote_dir)
        progress = sly.Progress(f"Downloading {remote_dir}", len(remote_files), need_info_log=True)
        for remote_file in remote_files:
            local_file = os.path.join(local_dir, sly.fs.get_file_name_with_ext(remote_file.path))
            if sly.fs.file_exists(local_file):  # @TODO: for debug
                pass
            else:
                g.api.file.download(g.team_id, remote_file.path, local_file)
            progress.iter_done_report()

    _download_dir(g.remote_configs_dir, g.local_configs_dir)
    _download_dir(g.remote_info_dir, g.local_info_dir)

    ui_state = sly.json.load_json_file(os.path.join(g.local_info_dir, "ui_state.json"))
    g.model_name = ui_state["selectedModel"]
    g.input_width = ui_state["imgSize"]["width"]
    g.input_height = ui_state["imgSize"]["height"]
    sly.logger.info("Model has been successfully downloaded")


def construct_model_meta():
    g.model_classes_json = sly.json.load_json_file(g.local_model_classes_path)

    obj_classes = []
    for obj_class_json in g.model_classes_json:
        obj_classes.append(sly.ObjClass.from_json(obj_class_json))
    g.model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))

