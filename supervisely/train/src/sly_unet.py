import sys
import os
from pathlib import Path

root_dir = Path(__file__).absolute().parents[3]
serve_src_dir = os.path.join(root_dir, "supervisely", "serve", "src")
sys.path.append(str(serve_src_dir))

import unet_model

class UNetModelBench(unet_model.UNetModel):
    in_train = True
