import supervisely_lib as sly
import sly_globals as g
import step01_input_project


def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None
    step01_input_project.init(data, state)