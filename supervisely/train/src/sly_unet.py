import sys
import os
from pathlib import Path

root_dir = Path(__file__).absolute().parents[3]  # Navigate up to the 'unet' directory
sys.path.append(str(root_dir))

from supervisely.serve.src.unet_model import UNetModel

# For local debug
# from serve.src.unet_model import UNetModel


class UNetModelBench(UNetModel):
    in_train = True
