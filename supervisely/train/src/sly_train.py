import supervisely_lib as sly
import sly_globals as g


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    data = {}
    state = {}
    data["taskId"] = g.task_id

    g.my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)