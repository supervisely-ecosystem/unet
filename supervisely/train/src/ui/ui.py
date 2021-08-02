import supervisely_lib as sly
import sly_globals as g
import step01_input_project
import step02_splits
import step03_classes


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None
    step01_input_project.init(data, state)
    step02_splits.init(g.project_info, g.project_meta, data, state)
    step03_classes.init(g.api, data, state, g.project_id, g.project_meta)