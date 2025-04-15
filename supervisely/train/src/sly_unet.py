import sys
import os
from pathlib import Path

# Add parent directory to path to find serve module
root_dir = Path(__file__).absolute().parents[3]  # Navigate up to the project root
serve_dir = os.path.join(root_dir, "supervisely", "serve")
sys.path.append(str(serve_dir))

# Import directly from the src directory
from src.unet_model import UNetModel
# from serve.src.unet_model import UNetModel


class UNetModelBench(UNetModel):
    in_train = True
