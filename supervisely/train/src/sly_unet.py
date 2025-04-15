import sys

sys.path.insert(0, "../")

from serve.src.unet_model import UNetModel


class UNetModelBench(UNetModel):
    in_train = True
