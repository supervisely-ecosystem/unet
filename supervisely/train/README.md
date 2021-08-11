<div align="center" markdown>
<img src="https://i.imgur.com/vh6d26z.png"/>

# Semantic segmentation: Train UNet-based models 

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#For-Developers">For developers</a> •
  <a href="#References">References</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/unet/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/unet)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/train&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/train&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Training dashboard for several UNet-based models:
- Convert annotations in Sypervisely format to segmentation masks
- Define Train / Validation splits
- Select classes for training
- Define Training Augmentations (on the fly)
- 5 UNet-based architectures
- Tune hyperparameters
- Monitor Metrics charts
- Preview model predictions over time
- All training artifacts will be uploaded to the Team Files

Supports following architectures: 
- Vanilla UNet
- UNet11 (based on vgg11)
- UNet16 (based on vgg16)
- AlbuNet (UNet modification with resnet34 encoder)
- LinkNet34


# How to Run
1. Add app to your team from Ecosystem
2. Be sure that you connected computer with GPU to your team by running Supervisely Agent on it ([how-to video](https://youtu.be/aDqQiYycqyk))
3. Run app from context menu of project with annotations: `Context menu` -> `Neural Networks` -> `UNet` -> `Train UNet`
4. Open Training Dashboard (app UI) and follow instructions provided in the video above


Watch [how-to video](https://youtu.be/R9sbH3biCmQ) for more details:

<a data-key="sly-embeded-video-link" href="https://youtu.be/R9sbH3biCmQ" data-video-code="R9sbH3biCmQ">
    <img src="https://i.imgur.com/O47n1S1.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

# For Developers

Developers can use this app as a template to integrate their own custom Semantic 
Segmentation models.

We recommend to follow several tutorials and documentation to get the basics before you start develop your custom app
- Supervisely platform
- Annotations in Supervisely format
- Supervisely SDK for Python
- IPython notebooks that explain how to communicate with the platform
- Quickstart APP development guide - how to create and add your custom apps to Supervisely Platfom
  - explains basic principles (python examples + videos)
  - shows how to configure development environment (PyCharm)
  - shows how to debug apps
  - how to add private repositories (in Enterprise Edition)
- Use other apps as an examples - sources for all our Apps in Supervisely Ecosystem are available on github

**Notice**: Documentation is constantly improving, if you find some topics are missing or in case of questions, please 
contact our tech support: 
- for Community Edition (CE) we have public Slack [![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
- for Enterprise Edition (EE) use your private Slack workspace

**Benefits for developers**:

Most of the features for Semantic Segmentations are already implemented:
1. **step-01** - download selected project from server to local app directory
2. **step-02** - choose how to prepare train / validation splits and save them to JSON files 
3. **step-03** - select classes for training. Sometimes you may want to train on a subset of classes. Background class `__bg__` will be added automatically to cover all unlabeled pixels
4. **step-04** - (optional) select template augmentations or provide your custom ones (use App `ImgAug Studio`), 
preview augmentations on random image from your project for only selected classes
5. **step-05** - select model architecture to train from scratch or provide path to your custom weights. App will automatically download them from server to local directory
6. **step-06** - define hyperparameters - optimized, learning rate schedule, epochs, batch size, how to save checkpoints, how to log metrics and so on ...
7. **step-07** - annotations from downloaded supervisely project will be converted to segmentations masks and validated, 
training loop will be started with real-time metrics and visualization of predictions after every epoch, after training 
all artifacts will be uploaded to the Team Files. 
8. Working **serving app** out of the box - you can send inference requests to your model from python, also other inference 
apps will be able to communicate with your model within platform. For example: apply model for unlabeled images, 
or apply your model right in labeling interface.
9. Apps for **model performance analysis** out of the box

## How app is spawned in Supervisely Platform. High-level explanation
1. You connect your computer with GPU to the platform by running Supervisely Agent on your machine ([how to video](https://youtu.be/aDqQiYycqyk)). 
It is our task manager that recieves task messages from platform. You can run it almost on any computer even without 
static IP address (it is a client, but works like server. Dropbox app works with the similar principles). This is a way 
how you can connect unlimited computational resources (servers) to the Platform
2. When you click `Run App` button, agent recieves task message, spawns app's docker container 
(docker image is defined in app's `config.json`) and mounts app sources inside
3. Inside this container it starts command `python /a/b/c/my_main_script.py`. Main python script is also defined in app's `config.json`
4. Now you scripts has its own isolated environment and can communicate with the platform using REST API 
(for 99.99% of cases you can use Supervisely Python SDK to do it in a convenient way)
5. Once app is started, agent parces logs (`stdour` and `stderr`) from runnning docker container and submits them to the paltform. 
It means that if you has line `print('hello!!!')` in your script, you will see this message in task logs.
6. Also agent can stop or kill app by request.

## Train dashboard: UI is separated from the main implementation
UI is mostly separated from the training loop. When you press `Train` button in UI, the state (values) from all widgets 
(selectors, sliders, input fields, ...) are added to the sys.args and your default `train.py` scripts is started. 
It means that you can really quickly integrate scripts you use locally to the UI dashboard, because thay are 98% separated.


## Light integration (Recommended)
For whom:
- if your model is has classic structure - RGB image as input and tensor with probabilities for every class as output  
- if you are satisfied with our training loop / loss functions / validation metrics
- if you don't want to significantly change training dashboard 

What to do:
1. Add model definition to `/custom_net/model.py`
2. Change variable `model_list` in `/custom_net/train.py`. This variable is used to show available models in UI

    For example:
    ```python
   from models import MyNet1, MyNet2
    model_list = {
    'MyNet1': {
        "class": MyNet1,
        "description": "My super model #1"
    },
   'MyNet1': {
        "class": MyNet2,
        "description": "My super model #2"
    },
   }
    ```
3. Already implemented, just check: how to apply model to a local image - function `inference` in `custom_net/inference.py`  
4. Already implemented, just check: how to convert model predictions to annotations in Supervisely format - function 
`convert_prediction_to_sly_format` in `custom_net/inference.py`

That's it. Not you can start debug and testing.   



## Deep Integration (for advanced users)
For whom:
- if you model has custom structure (complex input, several output branches, custom data preprocessing, ...)
- if you want to change training dashboard UI
- if you want to change DataLoaders and training loop
- if you want to define custom loss functions or calculate custom validation metrics
- and so on ... (feel free to change everything you need)

Main files with explanation:
- `custom_net/model.py` - models definition
- `custom_net/loss.py` - used loss functions
- `custom_net/sly_seg_dataset.py` - pytorch Dataset that loads directory with images and segmentation masks in 
Supervisely format. Augmentations (optional) are performed on the fly
- `custom_net/train.py` - main training script, taskes a lot of arguments as input (all this arguments will be passed from UI).
If you don't modify UI, you just need to parse these arguments and use all or some of them
- `custom_net/utils.py` - training loop
- `custom_net/validation.py` - metrics used during validation
- `custom_net/inference.py` - how to apply model to the image in local directory
- `custom_net/sly_integration.py` - function `_convert_prediction_to_sly_format` converts model predictions to 
annotations in Supervisely format. Other methods are used to log progress to UI: epoch / iteration / train and val metrics / preview predictions dynamics
- `docker/Dockerfile` - in app config you should define dockerimage (from public or private docker registry). 
- `supervisely/train` - train dashboard app
  - `supervisely/train/src/ui` - all steps from UI are in separate `html` nd `py` files
  - `supervisely/train/src/sly_train.py` main script for train app, defined in app config
  - `supervisely/train/src/gui.html` - main UI template for train app, defined in app config
- `remote_dev` directory is an example how to setup remote development and debugging using PyCharm. 
For example, if you prefer to develop NN on your Macbook by connecting to remote server with GPU. Learn more in docs 
or contact tech support.  

# References

- implementation of model architectures is from [this repo](https://github.com/ternaus/robot-surgery-segmentation)
- Image credits: [link1](https://arxiv.org/abs/1505.04597) and [link2](https://tariq-hasan.github.io/concepts/computer-vision-semantic-segmentation/)
