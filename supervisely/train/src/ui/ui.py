import supervisely as sly
import sly_globals as g
import step01_input_project
import step02_splits
import step03_classes
import step04_augs
import step05_models
import step06_hyperparameters
import step07_train


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None
    step01_input_project.init(data, state)
    step02_splits.init(g.project_info, g.project_meta, data, state)
    step03_classes.init(g.api, data, state, g.project_id, g.project_meta)
    step04_augs.init(data, state)
    step05_models.init(data, state)
    step06_hyperparameters.init(data, state)
    step07_train.init(data, state)


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}

    if restart_from_step <= 2:
        if restart_from_step == 2:
            step02_splits.restart(data, state)
        else:
            step02_splits.init(g.project_info, g.project_meta, data, state)
    if restart_from_step <= 3:
        if restart_from_step == 3:
            step03_classes.restart(data, state)
        else:
            step03_classes.init(g.api, data, state, g.project_id, g.project_meta)
    if restart_from_step <= 4:
        if restart_from_step == 4:
            step04_augs.restart(data, state)
        else:
            step04_augs.init(data, state)
    if restart_from_step <= 5:
        if restart_from_step == 5:
            step05_models.restart(data, state)
        else:
            step05_models.init(data, state)
    if restart_from_step <= 6:
        if restart_from_step == 6:
            step06_hyperparameters.restart(data, state)
        else:
            step06_hyperparameters.init(data, state)

    fields = [
        {"field": "data", "payload": data, "append": True, "recursive": False},
        {"field": "state", "payload": state, "append": True, "recursive": False},
        {"field": "state.restartFrom", "payload": None},
        {"field": f"state.collapsed{restart_from_step}", "payload": False},
        {"field": f"state.disabled{restart_from_step}", "payload": False},
        {"field": "state.activeStep", "payload": restart_from_step},
    ]
    g.api.app.set_fields(g.task_id, fields)
    g.api.app.set_field(task_id, "data.scrollIntoView", f"step{restart_from_step}")