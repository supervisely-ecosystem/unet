import sys
import os
from pathlib import Path

root_dir = Path(__file__).absolute().parents[3]
serve_src_dir = os.path.join(root_dir, "supervisely", "serve", "src")

sys.path.append(str(serve_src_dir))
model_list_dir = serve_src_dir
sys.path.append(str(model_list_dir))

from unet_model import UNetModel


class UNetModelBench(UNetModel):
    in_train = True
