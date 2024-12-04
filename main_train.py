
import os, sys
import numpy as np
import torch
import subprocess
import wandb
from torch.optim import Adam
from pprint import pprint

from src.data_loader import create_dataloader
from src.forward_diffusion import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from src.helper import diffusion_variables
from src.u_net import Unet
from src.train import train_model


# =========== DEFINE PARAMETERS =============
config_params = {
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 15,
    "timesteps": 1000
}

learning_rate = config_params["learning_rate"]
batch_size = config_params["batch_size"]
epochs = config_params["epochs"]
timesteps = config_params["timesteps"]
# =============================================

# =========== DEFINE U-NET ====================
model_config = {
    'dim': 64,                  # Base dimension for U-Net layers
    'dim_mults': (1, 2, 4, 8),  # Downsampling/upsampling configuration
    'resnet_block_groups': 4,   # Number of groups in GroupNorm within ResNet blocks
    'self_condition': False,    # Whether to use self-conditioning
    'init_kernel' : 7,          # input kernel size
    'init_padding' : 3          # input padding
}
# =============================================

# ===== DEFINE TIMESCHEDULER FOR FORWARD DIFFUSION  =====
# # linear
# beta_params = {"scheduler": "linear", "beta_start": 0.0001, "beta_end": 0.01}
# betas = linear_beta_schedule(timesteps, beta_start = beta_params['beta_start'], beta_end = beta_params['beta_end'])

# cos
beta_params = {"scheduler": "cos", "s": 0.0001, "beta_end" : 0.99}
betas = cosine_beta_schedule(timesteps, s = beta_params["s"], beta_end= beta_params["beta_end"])

# # sigmoid
# beta_params = {"scheduler": "sigmoid", "beta_start": 0.0001, "beta_end": 0.01}
# betas = sigmoid_beta_schedule(timesteps, beta_start = beta_params['beta_start'], beta_end = beta_params['beta_end'])  


# Callculates the diffusion variables
diffusion_variables = diffusion_variables(betas)
# =============================================

# ============== NORM =========================
#  scaleclip [-0.2,1], norm1_scaleclip [-1,1], norm2_scaleclip [0,1]
norm_params = {"norm": 'norm1_scaleclip'}
# =============================================

# ========= Initialize WandB logging ==========
combined_config = {**config_params, **model_config, **beta_params, **norm_params}

wandb.init(
    project='scheduler',
    entity='carlospeckert-eth-z-rich',
    name='cos_beta_end099_norm1',  
    notes='**',
    config=combined_config
)
# =============================================


# =========== LOAD DATA ========================
# OADAT location
oadat_dir = 'C:/Users/carlo/OADAT'

file_name = 'SWFD_semicircle_RawBP.h5'
key = 'sc_BP'

prng = np.random.RandomState(42)

indices = None # use whole dataset
# indices = np.arange(xxx) # Use only the first xxx

dataloader = create_dataloader(oadat_dir, file_name, key, norm_params['norm'], batch_size, 
                               shuffle=True, num_workers=0, prng=prng, indices=indices)
# =============================================


# =========== info print  =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("+--+---+----+--      --+----+---+--+")
print(f"{device}")
pprint(combined_config, width=60)
print("============== RUN =================")
# =============================================


# =========== MODEL SETUP =====================  
# Initialize a new model from scratch
model = Unet(**model_config).to(device)

# Initialize the optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)
# =============================================

# Move diffusion variables to the appropriate device (CPU/GPU)
for key in diffusion_variables:
    diffusion_variables[key] = diffusion_variables[key].to(device)



# ************** TRAIN *****************
train_model(model, dataloader, optimizer, 
            diffusion_variables, timesteps, epochs, device, 
            config_params, model_config, loss_type="huber") 
#  *************************************


# ========= Finalize the wandb run ============
wandb.finish()
# =============================================


