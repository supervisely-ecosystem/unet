<div align="center" markdown>
<img src="https://i.imgur.com/XfzccQT.png"/>

# Serve UNet

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#For-Developers">For Developers</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/unet/supervisely/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/unet)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

App deploys UNet model trained in Supervisely as REST API service. Serve app is the simplest way how any model 
can be integrated into Supervisely. Once model is deployed, user gets the following benefits:

1. Use out of the box apps for inference
   - used directly in [labeling interface](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) (images, videos)
   - apply to [images project or dataset](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset)
   - apply to videos (coming soon, check in Ecosystem)
2. Apps from Supervisely Ecosystem can use NN predictions: for visualization, for analysis, performance evaluation, etc ...
3. Communicate with NN in custom python script (see section <a href="#For-developers">for developers</a>)
4. App illustrates how to use NN weights. For example: you can train model in Supervisely, download its weights and use them the way you want.


# How To Run

1. Go to the directory with weights in `Team Files`. Training app saves results to the 
   directory: `/unet/<session id>_<experiment_name>/checkpoints`. Then right click to weights `.pth` file,
   for example: `/unet/7777_lemons_demo/checkpoints/model_47_best.pth`
   
<img src="https://i.imgur.com/cmEzYGr.gif"/>

2. Run `Serve UNet` app from context menu

3. Select device, both `gpu` and `cpu` are supported. Also in advanced section you can 
change what agent should be used for deploy.

4. Press `Run` button.

5. Wait until you see following message in logs: `Model has been successfully deployed`

<img src="https://i.imgur.com/AAKToCb.png" width="800"/>

6. All deployed models are listed in `Team Apps`. You can view logs and stop them from this page.

<img src="https://i.imgur.com/7eVkiIm.png"/>


# For Developers

1. Serving app allows developers to send direct inference request to the deployed model from their python script. 
Please, see YOLOv5 Serve app (**for developers** section) for more info (API is equal, the only difference is that yolo model produces rectangles in Supervisely 
format, but this model producess segmentation masks for every class in Supervisely format)
2. also you can use serving app as an example - how to use downloaded NN weights outside Supervisely 
(how to load model and how to get predictions from it).
3. Other apps can use serving app to communicate with NN and get predictions from it. Once you implemented it, apps 
from multiple categories become available with zero codding: inference apps, model performance analysis and much more.
Learn more in Ecosystem and in corresponding app collections.
4. If you want to integrate custom segmentation model, you don't need to modify sources of this serving app. Please, learn more 
in UNet training app readme. If you implemented steps from `Light integration` secsion, this serving app will work out of the box
