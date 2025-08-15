# Description: This file contains functionality for workflow feature

import supervisely as sly


def workflow_input(api: sly.Api, checkpoint_url: str):   
    try:
        node_settings = sly.WorkflowSettings(title="Serve Custom Model")
        meta = sly.WorkflowMeta(node_settings=node_settings)
        sly.logger.debug(f"Workflow Input: Checkpoint URL - {checkpoint_url}")
        if checkpoint_url and api.file.exists(sly.env.team_id(), checkpoint_url):
            api.app.workflow.add_input_file(checkpoint_url, model_weight=True, meta=meta)
        else:
            sly.logger.debug(f"Checkpoint {checkpoint_url} not found in Team Files. Cannot set workflow input")
    except Exception as e:
        sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")


def workflow_output(api: sly.Api):
    raise NotImplementedError("Method is not implemented yet")