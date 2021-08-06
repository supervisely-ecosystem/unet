import os
import sys

import step01_input_project
import step02_splits
import supervisely_lib as sly
from sly_progress_utils import init_progress, get_progress_cb, reset_progress
import sly_globals as g
import step03_classes
#from sly_seg_dataset import SlySegDataset


_open_lnk_name = "open_app.lnk"
project_dir_seg = None
model_classes_path = os.path.join(g.info_dir, "model_classes.json")

chart_lr: sly.app.widgets.Chart = None
chart_bce: sly.app.widgets.Chart = None
chart_dice: sly.app.widgets.Chart = None
chart_loss: sly.app.widgets.Chart = None


def init(data, state):
    init_progress("Train1", data)
    # init_progress("Iter", data)
    # init_progress("UploadDir", data)
    # data["eta"] = None

    init_charts(data, state)

    state["collapsed7"] = True
    state["disabled7"] = True
    state["done7"] = False

    state["started"] = False

    data["outputName"] = None
    data["outputUrl"] = None

    state["expName"] = g.project_info.name


def restart(data, state):
    data["done7"] = False


def init_charts(data, state):
    global chart_lr, chart_bce, chart_dice, chart_loss
    chart_lr = sly.app.widgets.Chart(g.task_id, g.api, "data.chartLR",
                                     title="LR", series_names=["LR"],
                                     yrange=[state["lr"] - state["lr"] / 2.0, state["lr"] + state["lr"] / 2.0],
                                     ydecimals=6, xdecimals=2)
    chart_bce = sly.app.widgets.Chart(g.task_id, g.api, "data.chartBCE",
                                      title="BCE", series_names=["train", "val"],
                                      smoothing=0.6, ydecimals=6, xdecimals=2)
    chart_dice = sly.app.widgets.Chart(g.task_id, g.api, "data.chartDICE",
                                      title="DICE", series_names=["train", "val"],
                                      smoothing=0.6, ydecimals=6, xdecimals=2)
    chart_loss = sly.app.widgets.Chart(g.task_id, g.api, "data.chartLoss",
                                       title="Total loss", series_names=["train", "val"],
                                       smoothing=0.6, ydecimals=6, xdecimals=2)
    state["smoothing"] = 0.6

    chart_lr.init_data(data)
    chart_bce.init_data(data)
    chart_dice.init_data(data)
    chart_loss.init_data(data)


def update_charts(phase, epoch, epoch_samples, metrics):
    fields = [
        chart_lr.get_append_field(epoch, metrics['lr']),
        chart_bce.get_append_field(epoch, metrics['bce'] / epoch_samples, phase),
        chart_dice.get_append_field(epoch, metrics['dice'] / epoch_samples, phase),
        chart_loss.get_append_field(epoch, metrics['loss']  / epoch_samples, phase),
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        # convert project to segmentation masks
        global project_dir_seg
        project_dir_seg = os.path.join(g.my_app.data_dir, g.project_info.name + "_seg")
        sly.fs.remove_dir(project_dir_seg)  # @TODO: for debug

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

        # model classes = selected_classes + bg
        project_seg = sly.Project(project_dir_seg, sly.OpenMode.READ)
        classes_json = project_seg.meta.obj_classes.to_json()

        # save model classes info + classes order. Order is used to convert model predictions to correct masks for every class
        sly.json.dump_json_file(classes_json, model_classes_path)

        set_train_arguments(state)
        import train
        train.main()

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


def set_train_arguments(state):
    # model
    sys.argv.extend(["--model", state["selectedModel"]])

    #for data loader
    sys.argv.extend(["--project-dir", project_dir_seg])
    sys.argv.extend(["--classes-path", model_classes_path])
    sys.argv.extend(["--train-set-path", step02_splits.train_set_path])
    sys.argv.extend(["--val-set-path", step02_splits.val_set_path])

    # basic hyperparameters
    sys.argv.extend(["--epochs", str(state["epochs"])])
    sys.argv.extend(["--input-size", str(state["imgSize"])])
    sys.argv.extend(["--batch-size", str(state["batchSizePerGPU"])])

    # # optimizer
    sys.argv.extend(["--optimizer", state["optimizer"]])
    sys.argv.extend(["--lr", str(state["lr"])])
    sys.argv.extend(["--momentum", str(state["momentum"])])
    sys.argv.extend(["--weight-decay", str(state["weightDecay"])])
    if state["nesterov"]:
        sys.argv.append("--nesterov")

    # lr schedule
    if state["lrPolicyEnabled"]:
        sys.argv.extend(["--lr-schedule", state["lrSchedule"]])
        sys.argv.extend(["--step-size", str(state["stepSize"])])
        sys.argv.extend(["--gamma-step", str(state["gammaStep"])])
        sys.argv.extend(["--milestones", str(state["milestones"])])
        sys.argv.extend(["--gamma-exp", str(state["gammaExp"])])
    else:
        sys.argv.extend(["--lr-schedule", ''])

    # system
    sys.argv.extend(["--gpu-id", f"cuda:{state['gpusId']}"])
    sys.argv.extend(["--num-workers", str(state['numWorkers'])])

    # logging
    sys.argv.extend(["--metrics-period", str(state['metricsPeriod'])])

    # checkpoints
    sys.argv.extend(["--val-interval", str(state['valInterval'])])
    sys.argv.extend(["--checkpoint-interval", str(state['checkpointInterval'])])
    if state["saveLast"]:
        sys.argv.append("--save-last")
    if state["saveBest"]:
        sys.argv.append("--save-best")

    sys.argv.append("--sly")

