import os
from dotenv import load_dotenv
import torch
import supervisely as sly
from unet_model import UNetModel
from pathlib import Path
from supervisely.nn.utils import ModelSource
from supervisely.nn.task_type import TaskType
from supervisely.io.fs import get_file_name_with_ext


root_source_path = str(Path(__file__).parents[3])
app_source_path = str(Path(__file__).parents[1])

if sly.is_development():
    load_dotenv(os.path.join(app_source_path, "local.env"))
    load_dotenv(os.path.expanduser("~/supervisely.env"))


checkpoint_url = os.environ['modal.state.slyFile']
device = os.environ['modal.state.device'] if 'cuda' in os.environ['modal.state.device'] and torch.cuda.is_available() else 'cpu'

m = UNetModel()
m.load_model(
    device=device,
    model_source=ModelSource.CUSTOM,
    task_type=TaskType.SEMANTIC_SEGMENTATION,
    checkpoint_name=get_file_name_with_ext(checkpoint_url),
    checkpoint_url=checkpoint_url,
)

if sly.is_production():
    m.serve()
else:
    image_path = "demo/image_01.jpg"
    results = m.predict(image_path, settings={})
    vis_path = "demo/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
