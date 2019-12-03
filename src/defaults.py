"""Defaults to be shared across modules"""
from functools import partial
import torch.optim as optim

lr = 3e-3
wd = 1e-2
seed = 100
optim_fn = optim.SGD #partial(optim.SGD, lr=lr)
batch_size = 32 #for vae

# Log
save_every = 10 # in unit of epoch
# Runner.validate will use this as the default integer flag to pass down to
# Runner._do_epoch_validate method
val_epoch = -1


# VAE
z_dim=10
ignore_gt = None # ground truth label to ignore from IOU and accuracy computation
# ^ look at metrics.running_metrics class
