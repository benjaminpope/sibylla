from NormFlow import NormFlow
import torchvision.utils
import math
import torch
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

class FlowEvaluator:
    def __init__(self, config) -> None:
        self.config = config

        self.create_model = NormFlow.get_create_model(config)


    def show_img_grid(imgs, row_size=4):
        """
        class helper method to show images in a compact grid
        """
        num_imgs = imgs.shape[0]
        nrow = min(num_imgs, row_size)
        ncol = int(math.ceil(num_imgs / nrow))
        imgs_torch = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2)
        imgs = torchvision.utils.make_grid(imgs_torch, nrow=nrow)
        np_imgs = imgs.cpu().numpy()
        plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
        plt.imshow(np.transpose(np_imgs, (1, 2, 0)))
        plt.axis('off')
        plt.show()

    
