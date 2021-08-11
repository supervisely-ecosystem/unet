# 🎉 UNet collection for semantic segmentation

# Overview 
🚀 This collection is designed to cover semantic segmenetation task in 
[**Supervisely**](https://supervise.ly/) and has two main purposes:
1. Demonstrate apps that can be useful for training semantic segmentation models
2. Provide UNet train / serve apps that can be used as a templates for integration of custom segmentation neural network 

Before using these apps 
we recommend to try end-to-end working demo - data and explanations are provided, get more details from apps readmes 
and how-to videos.

# Table of Contents

1. [About Supervisely](#-about-supervisely)
2. [Prerequisites](#Prerequisites)
3. [Apps Collection for Semantic Segmentation](#-apps-collection-for-semantic-segmentation)
    - [Demo data and synthetic data](#demo-data-and-synthetic-data)
    - [Data Exploration](#data-exploration)
    - [Data manipulation](#data-manipulation) - convert, merge, rasterize
    - [UNet](#unet-apps)
    - [Integration into labeling tool](#integration-into-labeling-tool)
4. [Contact & Questions & Suggestions](#contact--questions--suggestions)

# 🔥 About Supervisely

You can think of [Supervisely](https://supervise.ly/) as an Operating System available via Web Browser to help you solve Computer Vision tasks. The idea is to unify all the relevant tools that may be needed to make the development process as smooth and fast as possible. 

More concretely, Supervisely includes the following functionality:
 - Data labeling for images, videos, 3D point cloud and volumetric medical images (dicom)
 - Data visualization and quality control
 - State-Of-The-Art Deep Learning models for segmentation, detection, classification and other tasks
 - Interactive tools for model performance analysis
 - Specialized Deep Learning models to speed up data labeling (aka AI-assisted labeling)
 - Synthetic data generation tools
 - Instruments to make it easier to collaborate for data scientists, data labelers, domain experts and software engineers

One challenge is to make it possible for everyone to train and apply SOTA Deep Learning models directly from the Web Browser. To address it, we introduce an open sourced Supervisely Agent. All you need to do is to execute a single command on your machine with the GPU that installs the Agent. After that, you keep working in the browser and all the GPU related computations will be performed on the connected machine(s).


# Prerequisites
You should connect computer with GPU to your Supervisely account. If you already have Supervisely Agent running on your computer, you can skip this step.

 Several tools have to be installed on your computer:

- Nvidia drives + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

Once your computer is ready just add agent to your team and execute automatically generated running command in terminal. Watch how-to video:

<a data-key="sly-embeded-video-link" href="https://youtu.be/aDqQiYycqyk" data-video-code="aDqQiYycqyk">
    <img src="https://i.imgur.com/X9NTc5X.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="width:50%;">
</a>

# 🎉 UNet apps collection

To learn more about how to use every app, please go to app's readme page (links are provided). Just add the apps to your team to start using them.

Collection consists of the following apps (find more apps in Ecosystem): 

## Demo data and synthetic data

- [Lemons Annotated](https://ecosystem.supervise.ly/projects/lemons-annotated) - 6 images with two labeled
classes: lemons and kiwis

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/lemons-annotated" src="https://i.imgur.com/1as5W1L.png" width="350px"/>

- [Lemons (Test)](https://ecosystem.supervise.ly/projects/lemons-test) - images 
  with products on shelves, will be used to test classification model on real data

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/lemons-test" src="https://i.imgur.com/DsO08qM.png" width="350px"/>

- [Flying objects](https://ecosystem.supervise.ly/apps/flying-objects) - 
  app generates synthetic images for segmentation / detection / instance segmentation tasks from labeled examples 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/flying-objects" src="https://i.imgur.com/i5rCve6.png" width="380px"/>

## Data Exploration

- [Classes stats for images](https://ecosystem.supervise.ly/apps/classes-stats-for-images) - classes stats and detailed per image stats 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/classes-stats-for-images" src="https://i.imgur.com/ltTtIKT.png" width="380px"/>

- [Object size stats](https://ecosystem.supervise.ly/apps/object-size-stats) - classes stats and detailed per image stats 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/object-size-stats" src="https://i.imgur.com/1wx1G8F.png" width="380px"/>

- [Classes co-occurrence matrix](https://ecosystem.supervise.ly/apps/classes-co-occurrence-matrix) - discover edge cases and find “suspicious annotations” 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/classes-co-occurrence-matrix" src="https://i.imgur.com/O2O9J3k.png" width="380px"/>

## Data Manipulation

- [Convert class shape](https://ecosystem.supervise.ly/apps/convert-class-shape)

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/convert-class-shape" src="https://i.imgur.com/hFuVWOA.png" width="380px"/>

- [Merge classes](https://ecosystem.supervise.ly/apps/merge-classes)

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/convert-class-shape" src="https://i.imgur.com/S7GUdYv.png" width="380px"/>

- [Rasterize objects on images](https://ecosystem.supervise.ly/apps/rasterize-objects-on-images)

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/rasterize-objects-on-images" src="https://i.imgur.com/zoDWjP6.png" width="380px"/>

- [Export as masks](https://ecosystem.supervise.ly/apps/export-as-masks)

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/export-as-masks" src="https://i.imgur.com/NcI3tTs.png" width="380px"/>


## UNet apps

- [ImgAug studio](https://ecosystem.supervise.ly/apps/imgaug-studio) - prepare and visualize custom augmentation pipeline for training, augs are applied on the fly during training

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/imgaug-studio" src="https://i.imgur.com/l9kNWs9.png" width="380px"/>

- [Train UNet](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Funet%252Fsupervisely%252Ftrain) - training dashboard - customize hyperparameters and monitor metrics in real time

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/unet/supervisely/train" src="https://i.imgur.com/GjgRJre.png" width="350px"/>

- [Serve UNet](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Funet%252Fsupervisely%252Fserve) - deploy your model as Rest-API service and connect it with other apps from Ecosystem

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/unet/supervisely/serve" src="https://i.imgur.com/ATVxEk5.png" width="350px"/>

## Integration into labeling tool

- [Apply NN to images project ](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" width="350px"/> 

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image. 
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" width="350px"/>
  

# Contact & Questions & Suggestions

- for technical support please leave issues, questions or suggestions in our [repo](https://github.com/supervisely-ecosystem/unet). Our team will try to help.
- also we can chat in community slack channel [![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack) or in private workspaces if Enterprise Edition
- if you are interested in Supervisely Enterprise Edition (EE) please send us a [request](https://supervise.ly/enterprise/?demo) or email Yuri Borisov at [sales@supervise.ly](sales@supervise.ly)
