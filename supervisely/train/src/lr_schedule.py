# by default there is no learning rate schedule
# enable it by uncommenting one of the following examples and modify its settings
# learn more in official pytorch docs:
# https://pytorch.org/docs/1.9.0/optim.html#how-to-adjust-learning-rate
# Note 1: `step` argument should be less than the number of epochs

# ***********************************************
# Examples (PLEASE, UNCOMMENT ONLY ONE LINE):
# ***********************************************
import torch.optim.lr_scheduler as lr_scheduler

optimizer = None
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

