import os
import sys
import pathlib
import supervisely as sly
from supervisely.app.v1.app_service import AppService



root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

serve_source_path = os.path.join(root_source_path, "supervisely/serve/src")
sly.logger.info(f"Serve source directory: {serve_source_path}")
sys.path.append(serve_source_path)

model_source_path = os.path.join(root_source_path, "custom_net")
sly.logger.info(f"Model source directory: {model_source_path}")
sys.path.append(model_source_path)

# only for convenient debug
# from dotenv import load_dotenv
# debug_env_path = os.path.join(root_source_path, "supervisely/serve", "debug.env")
# secret_debug_env_path = os.path.join(root_source_path, "supervisely/serve", "secret_debug.env")
# load_dotenv(debug_env_path)
# load_dotenv(secret_debug_env_path, override=True)

my_app = AppService()
api = my_app.public_api
task_id = my_app.task_id

sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
remote_weights_path = os.environ['modal.state.slyFile']
device = os.environ['modal.state.device']

remote_exp_dir = str(pathlib.Path(remote_weights_path).parents[1])
remote_configs_dir = os.path.join(remote_exp_dir, "configs")
remote_info_dir = os.path.join(remote_exp_dir, "info")

local_weights_path = os.path.join(my_app.data_dir, sly.fs.get_file_name_with_ext(remote_weights_path))
local_configs_dir = os.path.join(my_app.data_dir, "configs")
sly.fs.mkdir(local_configs_dir)

local_info_dir = os.path.join(my_app.data_dir, "info")
sly.fs.mkdir(local_info_dir)
local_model_classes_path = os.path.join(local_info_dir, "model_classes.json")

model_name = None
model = None
model_classes_json = None
model_meta: sly.ProjectMeta = None  # list of classes and tags in Supervisely format
input_width = None
input_height = None
