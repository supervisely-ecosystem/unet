import supervisely_lib as sly
import sly_globals as g
import ui


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    data = {}
    state = {}
    data["taskId"] = g.task_id

    ui.init(data, state)  # init data for UI widgets
    g.my_app.compile_template(g.root_source_dir)
    g.my_app.run(data=data, state=state)


#@TODO: change predefined augs
#@TODO: __background__
#@TODO: disable rotate reflect - iaa.geometric.Rotate
#@TODO: check restart step for all steps
#@TODO: validation metrics exclude background?? integrate BCE??
#@TODO: reimplement everything to ProgressBar widget
#@TODO: report metric every X
if __name__ == "__main__":
    #sly.main_wrapper("main", main)
    main()