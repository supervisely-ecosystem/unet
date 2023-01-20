from custom_net.models import UNet11, LinkNet34, UNet, UNet16, AlbuNet

# Pytorch can automatically download pretrained weights, we use direct download to show progress bar in UI at step05
from torchvision.models.vgg import model_urls as vgg_urls
from torchvision.models.resnet import model_urls as resnet_urls

model_list = {
    'UNet': {
        "class": UNet,
        "description": "Vanilla UNet with random weights"
    },
    'UNet11': {
        "class": UNet11,
        "description": "Initialized from vgg-11 pretrained on ImageNet",
        "pretrained": vgg_urls['vgg11']
    },
    'UNet16': {
        "class": UNet16,
        "description": "Initialized from vgg-16 pretrained on ImageNet",
        "pretrained": vgg_urls['vgg16']
    },
    'AlbuNet': {
        "class": AlbuNet,
        "description": "UNet modification with resnet34 encoder pretrained on ImageNet",
        "pretrained": resnet_urls['resnet34']
    },
    'LinkNet34': {
        "class": LinkNet34,
        "description": "LinkNet with resnet34 encoder pretrained on ImageNet",
        "pretrained": resnet_urls['resnet34']
    }
}
