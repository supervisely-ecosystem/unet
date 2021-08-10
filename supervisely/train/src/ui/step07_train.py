import os
import sys
import random
from functools import partial

import step02_splits
import step04_augs
import step05_models
import supervisely_lib as sly
#from sly_progress_utils import init_progress, get_progress_cb, reset_progress
import sly_globals as g
import step03_classes


_open_lnk_name = "open_app.lnk"
project_dir_seg = None
model_classes_path = os.path.join(g.info_dir, "model_classes.json")

chart_lr: sly.app.widgets.Chart = None
chart_loss: sly.app.widgets.Chart = None
chart_acc: sly.app.widgets.Chart = None

gallery: sly.app.widgets.PredictionsDynamicsGallery = None
train_vis_items_path = os.path.join(g.info_dir, "train_vis_items.json")
val_vis_items_path = os.path.join(g.info_dir, "val_vis_items.json")

progress_epoch: sly.app.widgets.ProgressBar = None
progress_iter: sly.app.widgets.ProgressBar = None
progress_other: sly.app.widgets.ProgressBar = None


def init(data, state):
    # data["eta"] = None

    init_charts(data, state)
    init_progress_bars(data)

    state["collapsed7"] = True
    state["disabled7"] = True
    state["done7"] = False

    state["started"] = False

    data["outputName"] = None
    data["outputUrl"] = None

    state["visEpoch"] = 0
    data["finishedEpoch"] = 0
    state["setTimeIndexLoading"] = False

    data["gallery"] = gallery
    state["visSets"] = ["train", "val"]


def restart(data, state):
    data["done7"] = False


def init_charts(data, state):
    global chart_lr, chart_loss, chart_acc
    chart_lr = sly.app.widgets.Chart(g.task_id, g.api, "data.chartLR",
                                     title="LR", series_names=["LR"],
                                     yrange=[0, state["lr"] + state["lr"]],
                                     ydecimals=6, xdecimals=2)
    chart_loss = sly.app.widgets.Chart(g.task_id, g.api, "data.chartLoss",
                                      title="Loss", series_names=["train", "val"],
                                      smoothing=0.6, ydecimals=6, xdecimals=2)
    chart_acc = sly.app.widgets.Chart(g.task_id, g.api, "data.chartAcc",
                                      title="Val Acc", series_names=["avg IoU", "avg Dice"],
                                      yrange=[0, 1],
                                      smoothing=0.6, ydecimals=6, xdecimals=2)
    state["smoothing"] = 0.6

    chart_lr.init_data(data)
    chart_loss.init_data(data)
    chart_acc.init_data(data)


def init_progress_bars(data):
    global progress_epoch
    progress_epoch = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressEpoch", "Epoch")
    global progress_iter
    progress_iter = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressIter", "Iterations (train + val)")
    global progress_other
    progress_other = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressOther", "Progress")

    progress_epoch.init_data(data)
    progress_iter.init_data(data)
    progress_other.init_data(data)


def sample_items_for_visualization(state):
    train_set = sly.json.load_json_file(step02_splits.train_set_path)
    val_set = sly.json.load_json_file(step02_splits.val_set_path)

    train_vis_items = random.sample(train_set, state['trainVisCount'])
    val_vis_items = random.sample(val_set, state['valVisCount'])

    sly.json.dump_json_file(train_vis_items, train_vis_items_path)
    sly.json.dump_json_file(val_vis_items, val_vis_items_path)


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts_and_log_progress(experiment_name):
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.app.widgets.ProgressBar):
        if progress.get_total() is None:
            progress.set_total(monitor.len)
        else:
            progress.set(monitor.bytes_read)
        progress.update()

    global progress_other
    progress_other = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressOther",
                                                 "Upload directory with training artifacts to Team Files",
                                                 is_size=True, min_report_percent=5)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress_other)

    remote_dir = f"/unet/{g.task_id}_{experiment_name}"
    res_dir = g.api.file.upload_directory(g.team_id, g.artifacts_dir, remote_dir, progress_size_cb=progress_cb)
    return res_dir


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
            global progress_other
            progress_other = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressOther",
                                                         "Convert SLY annotations to segmentation masks",
                                                         sly.Project(g.project_dir, sly.OpenMode.READ).total_items)
            sly.Project.to_segmentation_task(
                g.project_dir, project_dir_seg,
                target_classes=step03_classes.selected_classes,
                progress_cb=progress_other.increment
            )
            progress_other.reset_and_update()

        # model classes = selected_classes + __bg__
        project_seg = sly.Project(project_dir_seg, sly.OpenMode.READ)
        classes_json = project_seg.meta.obj_classes.to_json()

        # save model classes info + classes order. Order is used to convert model predictions to correct masks for every class
        sly.json.dump_json_file(classes_json, model_classes_path)

        # predictions improvement over time
        global gallery
        gallery = sly.app.widgets.PredictionsDynamicsGallery(g.task_id, g.api, "data.gallery", project_seg.meta)
        gallery.complete_update()

        sample_items_for_visualization(state)
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))

        set_train_arguments(state)
        import train
        train.main()

        progress_epoch.reset_and_update()
        progress_iter.reset_and_update()

        remote_dir = upload_artifacts_and_log_progress(experiment_name=state["expName"])
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        # show result directory in UI
        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done7", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)
    except Exception as e:
        api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window

    # stop application
    g.my_app.show_modal_window("Training is finished, app is still running and you can preview predictions dynamics over time."
                               "Please stop app manually once you are finished with it.")
    #g.my_app.stop()


def set_train_arguments(state):
    # model
    sys.argv.extend(["--model", state["selectedModel"]])

    #for data loader
    sys.argv.extend(["--project-dir", project_dir_seg])
    sys.argv.extend(["--classes-path", model_classes_path])
    sys.argv.extend(["--train-set-path", step02_splits.train_set_path])
    sys.argv.extend(["--val-set-path", step02_splits.val_set_path])
    if state["useAugs"]:
        sys.argv.extend(["--sly-augs-path", step04_augs.augs_config_path])
    else:
        sys.argv.extend(["--sly-augs-path", ''])

    # basic hyperparameters
    sys.argv.extend(["--epochs", str(state["epochs"])])
    #sys.argv.extend(["--input-size", str(state["imgSize"])])
    sys.argv.extend(["--input-height", str(state["imgSize"]["height"])])
    sys.argv.extend(["--input-width", str(state["imgSize"]["width"])])
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
    sys.argv.extend(["--checkpoints-dir", g.checkpoints_dir])
    if state["maxKeepCkptsEnabled"]:
        sys.argv.extend(["--max-keep-ckpts", str(state["maxKeepCkpts"])])
    else:
        sys.argv.extend(["--max-keep-ckpts", str(-1)])

    if state["weightsInitialization"] == "custom":
        sys.argv.extend(["--custom-weights", step05_models.local_weights_path])

    # visualization settings
    sys.argv.extend(["--train-vis-items-path", train_vis_items_path])
    sys.argv.extend(["--val-vis-items-path", val_vis_items_path])
    sys.argv.append("--sly")


@g.my_app.callback("set_gallery_time_index")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def set_gallery_time_index(api: sly.Api, task_id, context, state, app_logger):
    try:
        gallery.set_time_index(state["visEpoch"])
    except Exception as e:
        api.task.set_field(task_id, "state.setTimeIndexLoading", False)
        raise e
    finally:
        api.task.set_field(task_id, "state.setTimeIndexLoading", False)


@g.my_app.callback("stop")
@sly.timeit
def stop(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.done7", "payload": True},
        {"field": "state.started", "payload": False},
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("follow_latest_prediction")
@sly.timeit
def follow_latest_prediction(api: sly.Api, task_id, context, state, app_logger):
    gallery.follow_last_time_index()