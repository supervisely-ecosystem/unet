import os
from pathlib import Path
import sys
import supervisely as sly
from supervisely.app.v1.app_service import AppService
# from dotenv import load_dotenv


root_source_dir = str(Path(sys.argv[0]).parents[3])
print(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

models_source_dir = os.path.join(root_source_dir, "custom_net")
print(f"Models source directory: {models_source_dir}")
sys.path.append(models_source_dir)

source_path = str(Path(sys.argv[0]).parents[0])
print(f"App source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
print(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)

# only for convenient debug
# debug_env_path = os.path.join(root_source_dir, "supervisely/train", "debug.env")
# secret_debug_env_path = os.path.join(root_source_dir, "supervisely/train", "secret_debug.env")
# load_dotenv(debug_env_path)
# load_dotenv(secret_debug_env_path, override=True)

my_app = AppService()
api = my_app.public_api
task_id = my_app.task_id

#@TODO: for debug
#sly.fs.clean_dir(my_app.data_dir)

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

project_info = api.project.get_info_by_id(project_id)
if project_info is None:  # for debug
    raise ValueError(f"Project with id={project_id} not found")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
project_dir = os.path.join(my_app.data_dir, "sly_project")

artifacts_dir = os.path.join(my_app.data_dir, "artifacts")
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
sly.fs.mkdir(checkpoints_dir, remove_content_if_exists=True)  # remove content for debug, has no effect in production
visualizations_dir = os.path.join(artifacts_dir, "visualizations")
sly.fs.mkdir(visualizations_dir, remove_content_if_exists=True)  # remove content for debug, has no effect in production

