import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.backward_diffusion import sample
from src.save_load_model import load_model_img_gen


def grid_plot(checkpoint_path, nrow=4, batch_size=16):  
    # Load model, config parameters, and diffusion variables
    model, config_params, diffusion_variables = load_model_img_gen(checkpoint_path)
    
    timesteps = config_params['timesteps']
    
    # Set image shape for reverse diffusion process
    x_0_shape = (batch_size, 1, 256, 256)
    
    print(f" --> Generating images GRID")

    # Perform reverse diffusion to generate images
    imgs_batch = sample(model, x_0_shape, timesteps, diffusion_variables)
  
    # Arrange images in a grid
    grid_img = make_grid(imgs_batch[-1], nrow=nrow, normalize=True, pad_value=1)

    return grid_img, config_params

