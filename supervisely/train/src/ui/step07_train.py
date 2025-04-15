import os
import sys
import math
import random
from functools import partial
from dataclasses import asdict
from supervisely.nn.inference import SessionJSON

import step02_splits
import step04_augs
import step05_models
import supervisely as sly
from supervisely.app.v1.widgets.progress_bar import ProgressBar
from supervisely.app.v1.widgets.chart import Chart
from supervisely.app.v1.widgets.predictions_dynamics_gallery import (
    PredictionsDynamicsGallery,
)
from supervisely.nn.artifacts.artifacts import TrainInfo
import sly_globals as g
import step03_classes
from step02_splits import get_train_val_sets
import workflow as w
from sly_functions import get_bg_class_name, get_eval_results_dir_name
from supervisely.nn.utils import ModelSource

_open_lnk_name = "open_app.lnk"
project_dir_seg = None
model_classes_path = os.path.join(g.info_dir, "model_classes.json")

chart_lr: Chart = None
chart_loss: Chart = None
chart_acc: Chart = None


gallery: PredictionsDynamicsGallery = None
train_vis_items_path = os.path.join(g.info_dir, "train_vis_items.json")
val_vis_items_path = os.path.join(g.info_dir, "val_vis_items.json")

progress_epoch: ProgressBar = None
progress_iter: ProgressBar = None
progress_other: ProgressBar = None


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
    state["visStep"] = 0

    data["finishedEpoch"] = 0
    state["setTimeIndexLoading"] = False

    data["gallery"] = gallery
    state["visSets"] = ["train", "val"]
    
    data["benchmarkUrl"] = None
    state["benchmarkInProgress"] = False


def restart(data, state):
    data["done7"] = False


def init_charts(data, state):
    global chart_lr, chart_loss, chart_acc
    chart_lr = Chart(
        g.task_id,
        g.api,
        "data.chartLR",
        title="LR",
        series_names=["LR"],
        yrange=[0, state["lr"] + state["lr"]],
        ydecimals=6,
        xdecimals=2,
    )
    chart_loss = Chart(
        g.task_id,
        g.api,
        "data.chartLoss",
        title="Loss",
        series_names=["train", "val"],
        smoothing=0.6,
        ydecimals=6,
        xdecimals=2,
    )
    chart_acc = Chart(
        g.task_id,
        g.api,
        "data.chartAcc",
        title="Val Acc",
        series_names=["avg IoU", "avg Dice"],
        yrange=[0, 1],
        smoothing=0.6,
        ydecimals=6,
        xdecimals=2,
    )
    state["smoothing"] = 0.6

    chart_lr.init_data(data)
    chart_loss.init_data(data)
    chart_acc.init_data(data)


def init_progress_bars(data):
    global progress_epoch
    progress_epoch = ProgressBar(g.task_id, g.api, "data.progressEpoch", "Epoch")
    global progress_iter
    progress_iter = ProgressBar(
        g.task_id, g.api, "data.progressIter", "Iterations (train + val)"
    )
    global progress_other
    progress_other = ProgressBar(g.task_id, g.api, "data.progressOther", "Progress")

    progress_epoch.init_data(data)
    progress_iter.init_data(data)
    progress_other.init_data(data)


def external_update_callback(progress: sly.tqdm_sly, progress_name: str):
    percent = math.floor(progress.n / progress.total * 100)
    fields = []
    if hasattr(progress, "desc"):
        fields.append({"field": f"data.progress{progress_name}", "payload": progress.desc})
    elif hasattr(progress, "message"):
        fields.append({"field": f"data.progress{progress_name}", "payload": progress.message})
    fields += [
        {"field": f"data.progressCurrent{progress_name}", "payload": progress.n},
        {"field": f"data.progressTotal{progress_name}", "payload": progress.total},
        {"field": f"data.progressPercent{progress_name}", "payload": percent},
    ]
    g.api.app.set_fields(g.task_id, fields)


def external_close_callback(progress: sly.tqdm_sly, progress_name: str):
    fields = [
        {"field": f"data.progress{progress_name}", "payload": None},
        {"field": f"data.progressCurrent{progress_name}", "payload": None},
        {"field": f"data.progressTotal{progress_name}", "payload": None},
        {"field": f"data.progressPercent{progress_name}", "payload": None},
    ]
    g.api.app.set_fields(g.task_id, fields)

class TqdmBenchmark(sly.tqdm_sly):
    def update(self, n=1):
        super().update(n)
        external_update_callback(self, "Benchmark")

    def close(self):
        super().close()
        external_close_callback(self, "Benchmark")


class TqdmProgress(sly.tqdm_sly):
    def update(self, n=1):
        super().update(n)
        external_update_callback(self, "Tqdm")

    def close(self):
        super().close()
        external_close_callback(self, "Tqdm")

def sample_items_for_visualization(state):
    train_set = sly.json.load_json_file(step02_splits.train_set_path)
    val_set = sly.json.load_json_file(step02_splits.val_set_path)

    train_vis_items = random.sample(train_set, state["trainVisCount"])
    val_vis_items = random.sample(val_set, state["valVisCount"])

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

    def upload_monitor(monitor, api: sly.Api, task_id, progress: ProgressBar):
        if progress.get_total() is None:
            progress.set_total(monitor.len)
        else:
            progress.set(monitor.bytes_read)
        progress.update()

    model_dir = g.sly_unet.framework_folder
    remote_artifacts_dir = f"{model_dir}/{g.task_id}_{experiment_name}"
    remote_weights_dir = os.path.join(remote_artifacts_dir, g.sly_unet.weights_folder)
    remote_config_path = os.path.join(remote_weights_dir, g.sly_unet.config_file)

    total_size = sly.fs.get_directory_size(g.artifacts_dir)
    global progress_other
    progress_other = ProgressBar(
        task_id=g.task_id,
        api=g.api,
        v_model="data.progressOther",
        message="Upload directory with training artifacts to Team Files",
        total=total_size,
        is_size=True,
        min_report_percent=5,
    )
    progress_cb = partial(
        upload_monitor, api=g.api, task_id=g.task_id, progress=progress_other
    )
    res_dir = g.api.file.upload_directory(
        g.team_id, g.artifacts_dir, remote_artifacts_dir, progress_size_cb=progress_cb
    )

    # generate metadata
    g.sly_unet_generated_metadata = g.sly_unet.generate_metadata(
        app_name=g.sly_unet.app_name,
        task_id=g.task_id,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_dir,
        weights_ext=g.sly_unet.weights_ext,
        project_name=g.project_info.name,
        task_type=g.sly_unet.task_type,
        config_path=remote_config_path,
    )

    progress_other.reset_and_update()
    return res_dir

def create_experiment(
    model_name, remote_dir, report_id=None, eval_metrics=None, primary_metric_name=None
):
    train_info = TrainInfo(**g.sly_unet_generated_metadata)
    experiment_info = g.sly_unet.convert_train_to_experiment_info(train_info)
    experiment_info.experiment_name = f"{g.task_id}_{g.project_info.name}_{model_name}"
    experiment_info.model_name = model_name
    experiment_info.framework_name = f"{g.sly_unet.framework_name}"
    experiment_info.train_size = g.train_size
    experiment_info.val_size = g.val_size
    experiment_info.evaluation_report_id = report_id
    if report_id is not None:
        experiment_info.evaluation_report_link = f"/model-benchmark?id={str(report_id)}"
    experiment_info.evaluation_metrics = eval_metrics

    experiment_info_json = asdict(experiment_info)
    experiment_info_json["project_preview"] = g.project_info.image_preview_url
    experiment_info_json["primary_metric"] = primary_metric_name

    g.api.task.set_output_experiment(g.task_id, experiment_info_json)
    experiment_info_json.pop("project_preview")
    experiment_info_json.pop("primary_metric")

    experiment_info_path = os.path.join(g.artifacts_dir, "experiment_info.json")
    remote_experiment_info_path = os.path.join(remote_dir, "experiment_info.json")
    sly.json.dump_json_file(experiment_info_json, experiment_info_path)
    g.api.file.upload(g.team_id, experiment_info_path, remote_experiment_info_path)

def calc_visualization_step(epochs):
    total_visualizations_count = 20

    vis_step = (
        int(epochs / total_visualizations_count)
        if int(epochs / total_visualizations_count) > 0
        else 1
    )
    g.api.app.set_field(g.task_id, "state.visStep", vis_step)

    return vis_step

def run_benchmark(api: sly.Api, task_id, classes, state, remote_dir):
    global m

    api.app.set_field(task_id, "state.benchmarkInProgress", True)
    benchmark_report_template, report_id, eval_metrics, primary_metric_name = None, None, None, None
    try:
        from sly_unet import UNetModelBench
        import torch
        from pathlib import Path
        import asyncio

        dataset_infos = api.dataset.get_list(g.project_id, recursive=True)

        dummy_pbar = TqdmProgress
        with dummy_pbar(message="Preparing trained model for benchmark", total=1) as p:
            # 0. Find the best checkpoint
            best_filename = None
            best_checkpoints = []
            latest_checkpoint = None
            other_checkpoints = []
            for root, dirs, files in os.walk(g.checkpoints_dir):
                for file_name in files:
                    path = os.path.join(root, file_name)
                    if file_name.endswith(".pth"):
                        if file_name.endswith("best.pth"):
                            best_checkpoints.append(path)
                        elif file_name.endswith("last.pth"):
                            latest_checkpoint = path
                        elif file_name.startswith("model_"):
                            other_checkpoints.append(path)

            if len(best_checkpoints) > 1:
                best_checkpoints = sorted(best_checkpoints, key=lambda x: x, reverse=True)
            elif len(best_checkpoints) == 0:
                sly.logger.info("Best model checkpoint not found in the checkpoints directory.")
                if latest_checkpoint is not None:
                    best_checkpoints = [latest_checkpoint]
                    sly.logger.info(
                        f"Using latest checkpoint for evaluation: {latest_checkpoint!r}"
                    )
                elif len(other_checkpoints) > 0:
                    parse_epoch = lambda x: int(x.split("_")[-1].split(".")[0])
                    best_checkpoints = sorted(other_checkpoints, key=parse_epoch, reverse=True)
                    sly.logger.info(
                        f"Using the last epoch checkpoint for evaluation: {best_checkpoints[0]!r}"
                    )

            if len(best_checkpoints) == 0:
                raise ValueError("No checkpoints found for evaluation.")
            best_checkpoint = Path(best_checkpoints[0])
            sly.logger.info(f"Starting model benchmark with the checkpoint: {best_checkpoint!r}")
            best_filename = best_checkpoint.name
            workdir = best_checkpoint.parent

            # 1. Serve trained model
            m = UNetModelBench(model_dir=str(workdir), use_gui=False)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sly.logger.info(f"Using device: {device}")

            checkpoint_path = g.sly_unet.get_weights_path(remote_dir) + "/" + best_filename
            sly.logger.info(f"Checkpoint path: {checkpoint_path}")

            deploy_params = dict(
                device=device,
                model_source=ModelSource.CUSTOM,
                task_type=sly.nn.TaskType.SEMANTIC_SEGMENTATION,
                checkpoint_name=best_filename,
                checkpoint_url=checkpoint_path,
            )
            m._load_model(**deploy_params)
            asyncio.set_event_loop(asyncio.new_event_loop())
            m.serve()

            import requests
            import uvicorn
            import time
            from threading import Thread

            def run_app():
                uvicorn.run(m.app, host="localhost", port=8000)

            thread = Thread(target=run_app, daemon=True)
            thread.start()

            while True:
                try:
                    requests.get("http://localhost:8000")
                    print("✅ Local server is ready")
                    break
                except requests.exceptions.ConnectionError:
                    print("Waiting for the server to be ready")
                    time.sleep(0.1)

            session = SessionJSON(api, session_url="http://localhost:8000")
            if sly.fs.dir_exists(g.data_dir + "/benchmark"):
                sly.fs.remove_dir(g.data_dir + "/benchmark")

            # 1. Init benchmark (todo: auto-detect task type)
            benchmark_dataset_ids = None
            benchmark_images_ids = None
            train_dataset_ids = None
            train_images_ids = None

            split_method = state["splitMethod"]

            if split_method == "datasets":
                train_datasets = state["trainDatasets"]
                val_datasets = state["valDatasets"]
                benchmark_dataset_ids = [ds.id for ds in dataset_infos if ds.name in val_datasets]
                train_dataset_ids = [ds.id for ds in dataset_infos if ds.name in train_datasets]
                train_set, val_set = get_train_val_sets(g.project_dir, state)
            else:

                def get_image_infos_by_split(split: list):
                    ds_infos_dict = {ds_info.name: ds_info for ds_info in dataset_infos}
                    image_names_per_dataset = {}
                    for item in split:
                        name = item.name
                        if name[1] == "_":
                            name = name[2:]
                        image_names_per_dataset.setdefault(item.dataset_name, []).append(name)
                    image_infos = []
                    for dataset_name, image_names in image_names_per_dataset.items():
                        if "/" in dataset_name:
                            dataset_name = dataset_name.split("/")[-1]
                        ds_info = ds_infos_dict[dataset_name]
                        for batched_names in sly.batched(image_names, 200):
                            batch_infos = api.image.get_list(
                                ds_info.id,
                                filters=[
                                    {
                                        "field": "name",
                                        "operator": "in",
                                        "value": batched_names,
                                    }
                                ],
                            )
                            image_infos.extend(batch_infos)
                    return image_infos

                train_set, val_set = get_train_val_sets(g.project_dir, state)

                val_image_infos = get_image_infos_by_split(val_set)
                train_image_infos = get_image_infos_by_split(train_set)
                benchmark_images_ids = [img_info.id for img_info in val_image_infos]
                train_images_ids = [img_info.id for img_info in train_image_infos]

                p.update(1)

        pbar = TqdmBenchmark
        bm = sly.nn.benchmark.SemanticSegmentationBenchmark(
            api,
            g.project_info.id,
            output_dir=g.data_dir + "/benchmark",
            gt_dataset_ids=benchmark_dataset_ids,
            gt_images_ids=benchmark_images_ids,
            progress=pbar,
            progress_secondary=pbar,
            classes_whitelist=classes,
        )

        train_info = {
            "app_session_id": sly.env.task_id(),
            "train_dataset_ids": train_dataset_ids,
            "train_images_ids": train_images_ids,
            "images_count": len(train_set),
        }
        bm.train_info = train_info

        # 2. Run inference
        bm.run_inference(session)

        # 3. Pull results from the server
        gt_project_path, pred_project_path = bm.download_projects(save_images=False)

        # 4. Evaluate
        bm._evaluate(gt_project_path, pred_project_path)
        bm._dump_eval_inference_info(bm._eval_inference_info)

        # 5. Upload evaluation results
        eval_res_dir = get_eval_results_dir_name(api, sly.env.task_id(), g.project_info)
        bm.upload_eval_results(eval_res_dir + "/evaluation/")

        # # 6. Speed test
        if state["runSpeedTest"]:
            try:
                session_info = session.get_session_info()
                support_batch_inference = session_info.get("batch_inference_support", False)
                max_batch_size = session_info.get("max_batch_size")
                batch_sizes = (1, 8, 16)
                if not support_batch_inference:
                    batch_sizes = (1,)
                elif max_batch_size is not None:
                    batch_sizes = tuple([bs for bs in batch_sizes if bs <= max_batch_size])
                bm.run_speedtest(session, g.project_info.id, batch_sizes=batch_sizes)
                bm.upload_speedtest_results(eval_res_dir + "/speedtest/")
            except Exception as e:
                sly.logger.warning(f"Speedtest failed. Skipping. {e}")

        # 7. Prepare visualizations, report and
        bm.visualize()
        remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
        report_id = bm.report.id
        eval_metrics = bm.key_metrics
        primary_metric_name = bm.primary_metric_name

        # 8. UI updates
        benchmark_report_template = bm.report

        fields = [
            {"field": f"state.benchmarkInProgress", "payload": False},
            {"field": f"data.benchmarkUrl", "payload": bm.get_report_link()},
        ]
        api.app.set_fields(g.task_id, fields)
        sly.logger.info(
            f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
        )

        # 9. Stop the server
        try:
            m.app.stop()
        except Exception as e:
            api.app.set_field(task_id, "state.benchmarkInProgress", False)
            sly.logger.warning(f"Failed to stop the model app: {e}")
        try:
            thread.join()
        except Exception as e:
            api.app.set_field(task_id, "state.benchmarkInProgress", False)
            sly.logger.warning(f"Failed to stop the server: {e}")
    except Exception as e:
        api.app.set_field(task_id, "state.benchmarkInProgress", False)
        sly.logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
        try:
            if bm.dt_project_info:
                api.project.remove(bm.dt_project_info.id)
        except Exception as re:
            pass

    return benchmark_report_template, report_id, eval_metrics, primary_metric_name

@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    calc_visualization_step(state["epochs"])
    try:
        # convert project to segmentation masks
        global project_dir_seg
        project_dir_seg = os.path.join(g.my_app.data_dir, g.project_info.name + "_seg")

        if (
            sly.fs.dir_exists(project_dir_seg) is False
        ):  # for debug, has no effect in production
            sly.fs.mkdir(project_dir_seg, remove_content_if_exists=True)
            global progress_other
            progress_other = ProgressBar(
                g.task_id,
                g.api,
                "data.progressOther",
                "Convert SLY annotations to segmentation masks",
                sly.Project(g.project_dir, sly.OpenMode.READ).total_items,
            )
            sly.Project.to_segmentation_task(
                g.project_dir,
                project_dir_seg,
                target_classes=step03_classes.selected_classes,
                progress_cb=progress_other.increment,
            )
            progress_other.reset_and_update()

        # model classes = selected_classes + __bg__
        project_seg = sly.Project(project_dir_seg, sly.OpenMode.READ)
        classes_json = project_seg.meta.obj_classes.to_json()

        # save model classes info + classes order. Order is used to convert model predictions to correct masks for every class
        sly.json.dump_json_file(classes_json, model_classes_path)

        # predictions improvement over time
        global gallery
        gallery = PredictionsDynamicsGallery(
            g.task_id, g.api, "data.gallery", project_seg.meta
        )
        gallery.complete_update()

        sample_items_for_visualization(state)
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))

        set_train_arguments(state)
        import train

        train.main()

        progress_epoch.reset_and_update()
        progress_iter.reset_and_update()

        remote_dir = upload_artifacts_and_log_progress(experiment_name=state["expName"])
        file_info = api.file.get_info_by_path(
            g.team_id, os.path.join(remote_dir, _open_lnk_name)
        )
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
    
    # run benchmark
    benchmark_report_template, report_id, eval_metrics, primary_metric_name = (
            None,
            None,
            None,
            None,
        )
    
    # sly.logger.info(f"Run benchmark: {state['runBenchmark']}")
    if state["runBenchmark"]:
        classes = [obj_cls.name for obj_cls in project_seg.meta.obj_classes]
        benchmark_report_template, report_id, eval_metrics, primary_metric_name = run_benchmark(
            api, task_id, classes, state, remote_dir
        )

    try:
        sly.logger.info("Creating experiment info")
        create_experiment(state["selectedModel"], remote_dir, report_id, eval_metrics, primary_metric_name)
    except Exception as e:
        sly.logger.warning(
            f"Couldn't create experiment, this training session will not appear in experiments table. Error: {e}"
        )
    
    w.workflow_input(api, g.project_info, state)
    w.workflow_output(api, g.sly_unet_generated_metadata, state, benchmark_report_template)

    
    # stop application
    # g.my_app.show_modal_window("Training is finished, app is still running and you can preview predictions dynamics over time."
    #                           "Please stop app manually once you are finished with it.")
    g.my_app.stop()


def set_train_arguments(state):
    # model
    sys.argv.extend(["--model", state["selectedModel"]])

    # for data loader
    sys.argv.extend(["--project-dir", project_dir_seg])
    sys.argv.extend(["--classes-path", model_classes_path])
    sys.argv.extend(["--train-set-path", step02_splits.train_set_path])
    sys.argv.extend(["--val-set-path", step02_splits.val_set_path])
    if state["useAugs"]:
        sys.argv.extend(["--sly-augs-path", step04_augs.augs_config_path])
    else:
        sys.argv.extend(["--sly-augs-path", ""])

    # basic hyperparameters
    sys.argv.extend(["--epochs", str(state["epochs"])])
    # sys.argv.extend(["--input-size", str(state["imgSize"])])
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
        sys.argv.extend(["--lr-schedule", ""])

    # system
    sys.argv.extend(["--gpu-id", f"cuda:{state['gpusId']}"])
    sys.argv.extend(["--num-workers", str(state["numWorkers"])])

    # logging
    sys.argv.extend(["--metrics-period", str(state["metricsPeriod"])])

    # checkpoints
    sys.argv.extend(["--val-interval", str(state["valInterval"])])
    sys.argv.extend(["--checkpoint-interval", str(state["checkpointInterval"])])
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
