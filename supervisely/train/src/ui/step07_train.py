import os

import step01_input_project
import supervisely_lib as sly
from sly_progress_utils import init_progress, get_progress_cb, reset_progress
import sly_globals as g
import step03_classes

_open_lnk_name = "open_app.lnk"
project_dir_seg = None


def init(data, state):
    init_progress("Train1", data)
    # init_progress("Iter", data)
    # init_progress("UploadDir", data)
    # data["eta"] = None

    #init_charts(data, state)

    state["collapsed7"] = True
    state["disabled7"] = True
    state["done7"] = False

    state["started"] = False

    data["outputName"] = None
    data["outputUrl"] = None

    state["expName"] = g.project_info.name


def restart(data, state):
    data["done7"] = False


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        # convert project to segmentation masks
        global project_dir_seg
        project_dir_seg = os.path.join(g.my_app.data_dir, g.project_info.name + "_seg")

        if sly.fs.dir_exists(project_dir_seg) is False: # for debug, has no effect in production
            sly.fs.mkdir(project_dir_seg, remove_content_if_exists=True)
            progress_cb = get_progress_cb(
                index="Train1",
                message="Convert SLY annotations to segmentation masks",
                total=step01_input_project.project_fs.total_items
            )
            sly.Project.to_segmentation_task(
                g.project_dir, project_dir_seg,
                target_classes=step03_classes.selected_classes,
                progress_cb=progress_cb
            )
            reset_progress(index="Train1")






        #sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))


        # init_script_arguments(state)
        # mm_train()
        #
        # # hide progress bars and eta
        # fields = [
        #     {"field": "data.progressEpoch", "payload": None},
        #     {"field": "data.progressIter", "payload": None},
        #     {"field": "data.eta", "payload": None},
        # ]
        # g.api.app.set_fields(g.task_id, fields)
        #
        # remote_dir = upload_artifacts_and_log_progress()
        # file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        # api.task.set_output_directory(task_id, file_info.id, remote_dir)
        #
        # # show result directory in UI
        # fields = [
        #     {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
        #     {"field": "data.outputName", "payload": remote_dir},
        #     {"field": "state.done9", "payload": True},
        #     {"field": "state.started", "payload": False},
        # ]
        # g.api.app.set_fields(g.task_id, fields)
    except Exception as e:
        api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window

    # stop application
    g.my_app.stop()