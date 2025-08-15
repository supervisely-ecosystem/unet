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
from supervisely.nn.artifacts.unet import UNet

from model_list import model_list
import workflow as w

team_id = sly.env.team_id()
api = sly.Api.from_env()

class UNetModel(sly.nn.inference.SemanticSegmentation):
    in_train = False
    transforms_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
    ])

    def load_model(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        model_source: Literal["Pretrained models", "Custom models"],
        task_type: Literal["semantic segmentation"],
        checkpoint_name: str,
        checkpoint_url: str,
    ):
        self.device = device
        self.model_source = model_source
        self.task_type = task_type
        self.checkpoint_name = checkpoint_name
        self.checkpoint_url = checkpoint_url
        
        model_dir = Path(checkpoint_url).parents[1]
        ui_state_path = str(model_dir / "info" / "ui_state.json")
        model_classes_path = str(model_dir / "info" / "model_classes.json")
        remote_files = (checkpoint_url, ui_state_path, model_classes_path)

        weights_path, ui_state_path, model_classes_path = [self.download(p) for p in remote_files]
        ui_state = sly.json.load_json_file(ui_state_path)
        self.model_classes = sly.json.load_json_file(model_classes_path)
        self.model_name = ui_state["selectedModel"]
        self.input_width = ui_state["imgSize"]["width"]
        self.input_height = ui_state["imgSize"]["height"]

        obj_classes = []
        for obj_class_json in self.model_classes:
            obj_classes.append(sly.ObjClass.from_json(obj_class_json))
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        
        num_classes = len(self.model_classes)
        self.class_names = [x['title'] for x in self.model_classes]
        model_class = model_list[self.model_name]["class"]
        self.model = model_class(num_classes=num_classes)
        state = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        
        # -------------------------------------- Add Workflow Input -------------------------------------- #
        if not self.in_train:
            sly.logger.debug("Workflow: Start processing Input")
            if self.model_source == "Custom models":
                sly.logger.debug("Workflow: Custom model detected")
                w.workflow_input(api, self.checkpoint_url)
            else:
                sly.logger.debug("Workflow: Pretrained model detected. No need to set Input")
            sly.logger.debug("Workflow: Finish processing Input")
        # ----------------------------------------------- - ---------------------------------------------- #
        
        self.model.eval()
        self.checkpoint_info = sly.nn.inference.CheckpointInfo(
            checkpoint_name=self.checkpoint_name,
            model_name=self.model_name,
            architecture=None,
            checkpoint_url=self.checkpoint_url,
            custom_checkpoint_path=self.checkpoint_url,
            model_source=self.model_source,
        )

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self):
        info = super().get_info()
        info["model_name"] = "UNet"
        info["checkpoint_name"] = self.model_name
        info["device"] = self.device
        info["pretrained_on_dataset"] = "custom"
        info["sliding_window_support"] = self.sliding_window_mode
        return info

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionSegmentation]:
        image = sly.image.read(image_path)  # RGB
        input = self.prepare_image_input(image, self.device)
        with torch.no_grad():
            output = self.model(input)
        image_height, image_width = image.shape[:2]
        predicted_classes_indices = output.data.cpu().numpy().argmax(axis=1)[0]
        result = cv2.resize(predicted_classes_indices, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        return [sly.nn.PredictionSegmentation(result)]
    
    def prepare_image_input(self, image, device):
        # RGB -> Normalized Tensor
        input = cv2.resize(image, (self.input_width, self.input_height))
        input = self.transforms_img(input)  # totensor + normalize
        input = torch.unsqueeze(input, 0)
        input = input.to(device)
        return input