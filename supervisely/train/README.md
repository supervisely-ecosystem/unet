<div align="center" markdown>
<img src="https://i.imgur.com/vh6d26z.png"/>

# Semantic segmentation: Train UNet-based models 

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/unet/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/unet)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/train&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/unet/supervisely/train&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Training dashboard for UNet:
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



# References

- implementation of model architectures is from [this repo](https://github.com/ternaus/robot-surgery-segmentation)
- Image credits: [link1](https://arxiv.org/abs/1505.04597) and [link2](https://tariq-hasan.github.io/concepts/computer-vision-semantic-segmentation/)
