import os
from typing_extensions import Literal
from typing import List, Any, Dict
from pathlib import Path
import cv2
from dotenv import load_dotenv
import torch
import supervisely as sly
from pathlib import Path
from torchvision import transforms

from model_list import model_list

root_source_path = str(Path(__file__).parents[3])
app_source_path = str(Path(__file__).parents[1])
load_dotenv(os.path.join(app_source_path, "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))


weights_path = os.environ['modal.state.weightsPath']
model_dir = Path(weights_path).parents[1]
ui_state_path = str(model_dir / "info" / "ui_state.json")
model_classes_path = str(model_dir / "info" / "model_classes.json")

device = os.environ['modal.state.device'] if 'cuda' in os.environ['modal.state.device'] and torch.cuda.is_available() else 'cpu'


class UNetModel(sly.nn.inference.SemanticSegmentation):
    
    transforms_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
    ])

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
    
        self.model = self.load_model(weights_path, device)
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
        image = sly.image.read(image_path)  # RGB
        input = self.prepare_image_input(image, device)
        with torch.no_grad():
            output = self.model(input)
        image_height, image_width = image.shape[:2]
        predicted_classes_indices = output.data.cpu().numpy().argmax(axis=1)[0]
        result = cv2.resize(predicted_classes_indices, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        return [sly.nn.PredictionSegmentation(result)]
    
    def load_model(self, weights_path, device):
        num_classes = len(self.model_classes)
        model_class = model_list[self.model_name]["class"]
        model = model_class(num_classes=num_classes)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    def prepare_image_input(self, image, device):
        # RGB -> Normalized Tensor
        input = cv2.resize(image, (self.input_width, self.input_height))
        input = self.transforms_img(input)  # totensor + normalize
        input = torch.unsqueeze(input, 0)
        input = input.to(device)
        return input


m = UNetModel(location=[weights_path, ui_state_path, model_classes_path])
m.load_on_device(device)

if sly.is_production():
    m.serve()
else:
    image_path = "demo/image_01.jpg"
    results = m.predict(image_path, settings={})
    vis_path = "demo/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
