import supervisely
import custom_net

import os
from typing_extensions import Literal
from typing import List, Any, Dict
from pathlib import Path
import cv2
import json
from dotenv import load_dotenv
import torch
import supervisely as sly
from pathlib import Path

os.chdir('supervisely_sly/serve')
print(os.getcwd())
import sys
print(sys.path)

import utils

load_dotenv("debug.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

is_prod = os.environ['ENV'] == 'production'

if is_prod:
    weights_path = os.environ['modal.state.weightsPath']
    model_dir = Path(weights_path).parents[1]
    ui_state_path = str(model_dir / "info" / "ui_state.json").replace('\\', '/')
    model_classes_path = str(model_dir / "info" / "model_classes.json").replace('\\', '/')
else:
    weights_path = 'data/model/model_004_best.pth'
    ui_state_path = 'data/model/ui_state.json'
    model_classes_path = 'data/model/model_classes.json'

device = os.environ['modal.state.device'] if 'cuda' in os.environ['modal.state.device'] and torch.cuda.is_available() else 'cpu'


class UNetModel(sly.nn.inference.SemanticSegmentation):
    def load_on_device(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        weights_path, ui_state_path, model_classes_path = self.location
        ui_state = sly.json.load_json_file(ui_state_path)
        self.model_classes = sly.json.load_json_file(model_classes_path)
        self.model_name = ui_state["selectedModel"]
        self.input_width = ui_state["imgSize"]["width"]
        self.input_height = ui_state["imgSize"]["height"]

        obj_classes = []
        for obj_class_json in self.model_classes:
            obj_classes.append(sly.ObjClass.from_json(obj_class_json))
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
    
        model = utils.load_model(weights_path, len(self.model_classes), self.model_name, device)
        self.model = model
        self.class_names = [x['title'] for x in self.model_classes]

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self):
        info = super().get_info()
        info["model_name"] = "UNet"
        info["checkpoint_name"] = self.model_name
        info["device"] = device
        info["pretrained_on_dataset"] = "custom"
        info["sliding_window_support"] = self.sliding_window_mode
        return info

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionSegmentation]:

        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. DETECTRON2) #######
        image = sly.image.read(image_path)  # RGB
        input = utils.prepare_image_input(image, self.input_width, self.input_height)
        input = torch.unsqueeze(input, 0)
        input = utils.cuda(input, device)
        with torch.no_grad():
            output = self.model(input)
        image_height, image_width = image.shape[:2]
        predicted_classes_indices = output.data.cpu().numpy().argmax(axis=1)[0]
        result = cv2.resize(predicted_classes_indices, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. DETECTRON2)  ########

        return [sly.nn.PredictionSegmentation(result)]


m = UNetModel(location=[weights_path, ui_state_path, model_classes_path])
m.load_on_device(device)

image_path = "demo/IMG_0748.jpeg"
results = m.predict(image_path, settings={})
vis_path = "demo/image_01_prediction.jpg"
m.visualize(results, image_path, vis_path)
print(f"predictions and visualization have been saved: {vis_path}")
