"""
Code to demonstrate some simple things about normalising flows and show the
refactoring has been done correctly

"""

import os
import torch
import torchvision
import numpy as onp

import jax
import jax.numpy as np
import math
import matplotlib.pyplot as plt

import urllib.request
from urllib.error import HTTPError
from layers import *
from DataLoader import DataLoader
from TrainerModule import TrainerModule
from flows import ImageFlow


def show_imgs(imgs, title=None, row_size=4):
    # Form a grid of pictures (we use max. 8 columns)
    imgs = np.copy(jax.device_get(imgs))
    num_imgs = imgs.shape[0]
    is_int = (imgs.dtype == np.int32)
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs / nrow))
    imgs_torch = torch.from_numpy(onp.array(imgs)).permute(0, 3, 1, 2)
    imgs = torchvision.utils.make_grid(imgs_torch, nrow=nrow, pad_value=128 if is_int else 0.5)
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


checkpoint_path = "./saved_models/simple_example"

# Use pretrained model
# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial11/"
# Files to download
pretrained_files = ["MNISTFlow_simple.ckpt", "MNISTFlow_vardeq.ckpt", "MNISTFlow_multiscale.ckpt",
                    "MNISTFlow_simple_results.json", "MNISTFlow_vardeq_results.json",
                    "MNISTFlow_multiscale_results.json"]
# Create checkpoint path if it doesn't exist yet
os.makedirs(checkpoint_path, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(checkpoint_path, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please contact the author with the full output including the"
                  "following error:\n", e)

train_set, val_set, test_set = DataLoader.load_MNIST()
train_exmp_loader, train_data_loader, \
    val_loader, test_loader = DataLoader.generate_data_loaders(train_set, val_set, test_set)

# show_imgs(np.stack([train_set[i][0] for i in range(8)], axis=0))

def create_checkerboard_mask(h, w, invert=False):
    x, y = np.arange(h, dtype=np.int32), np.arange(w, dtype=np.int32)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    mask = np.fmod(xx + yy, 2)
    mask = mask.astype(np.float32).reshape(1, h, w, 1)
    if invert:
        mask = 1 - mask
    return mask

def create_channel_mask(c_in, invert=False):
    mask = np.concatenate([
                np.ones((c_in//2,), dtype=np.float32),
                np.zeros((c_in-c_in//2,), dtype=np.float32)
            ])
    mask = mask.reshape(1, 1, 1, c_in)
    if invert:
        mask = 1 - mask
    return mask

def create_multiscale_flow():
    flow_layers = []

    vardeq_layers = [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=16),
                                   mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
                                   c_in=1) for i in range(4)]
    flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]

    flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=32),
                                  mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
                                  c_in=1) for i in range(2)]
    flow_layers += [SqueezeFlow()]
    for i in range(2):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=8, c_hidden=48),
                                      mask=create_channel_mask(c_in=4, invert=(i%2==1)),
                                      c_in=4)]
    flow_layers += [SplitFlow(),
                    SqueezeFlow()]
    for i in range(4):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=16, c_hidden=64),
                                      mask=create_channel_mask(c_in=8, invert=(i%2==1)),
                                      c_in=8)]

    flow_model = ImageFlow(flow_layers)
    return flow_model


flow_dict = {"simple": {}, "vardeq": {}, "multiscale": {}}
flow_dict["multiscale"]["model"], flow_dict["multiscale"]["result"] = TrainerModule.train_flow(create_multiscale_flow(), checkpoint_path, model_name="MNISTFlow")
