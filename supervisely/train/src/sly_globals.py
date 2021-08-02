import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import supervisely_lib as sly


root_source_dir = str(Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
sly.logger.info(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

# only for convenient debug
debug_env_path = os.path.join(root_source_dir, "supervisely/train", "debug.env")
secret_debug_env_path = os.path.join(root_source_dir, "supervisely/train", "secret_debug.env")
load_dotenv(debug_env_path)
load_dotenv(secret_debug_env_path, override=True)

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

project_info = api.project.get_info_by_id(project_id)
if project_info is None:  # for debug
    raise ValueError(f"Project with id={project_id} not found")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
project_dir = os.path.join(my_app.data_dir, "sly_project")
