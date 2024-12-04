import os
import subprocess
import wandb
import numpy as np
import torch
from pprint import pprint

from torch.optim import Adam
from torchvision.utils import save_image

from src.data_loader import create_dataloader 
from src.train import continuing_training
from src.save_load_model import save_model, load_model_continuing


# Continuing parameters
start_epoch = 1 # same as last epoch that finished
epochs = 2 
batch_size = 10 
project = 'test'
name = 'init_conv_9-4'

# Load model configuration parameters from checkpoint
print(">>> load checkpoint")
checkpoint_path = f'../external_storage/poly_carlo/checkpoint/{project}/{name}/checkpoint.pth'
_, _, config_params, model_config, _ = load_model_continuing(checkpoint_path)

# ============== NORM =========================
norm = 'norm1_scaleclip'
#  scaleclip [-0.2,1], norm1_scaleclip [-1,1], norm2_scaleclip [0,1]
norm_params = {"norm": norm}
# =============================================

# ========= Initialize WandB logging ==========
combined_config = {**config_params, **model_config, **norm_params}
wandb.init(
    project=project,
    entity='carlospeckert-eth-z-rich',
    name=name,  
    notes='**',
    config=combined_config
)
# =============================================




# =========== LOAD DATA ========================
file_name = 'SWFD_semicircle_RawBP.h5'
key = 'sc_BP'

# Synchronize data file using rsync
subprocess.run(
    ["rsync", "-avzhP", f"/home/jovyan/work/oadat/{file_name}", "/home/jovyan/work/oadat-ddpm/data"], 
    check=True
)
oadat_dir = '/home/jovyan/work/oadat-ddpm/data'
prng = np.random.RandomState(42)
indices = None  # Use full dataset or specify subset

# Initialize DataLoader
dataloader = create_dataloader(
    oadat_dir, file_name, key, norm_params['norm'], batch_size, 
    shuffle=True, num_workers=0, prng=prng, indices=indices
)
# =============================================

# =========== DEVICE SETUP ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("+--+---+----+--      --+----+---+--+")
print(f"Device: {device}")
pprint(combined_config, width=60)
print("Continuous Training Initialized")
print("============== RUN =================")
# =============================================

# Run the training function
continuing_training(checkpoint_path, dataloader, start_epoch, epochs, 
                    device, loss_type = 'huber')

# ========= Finalize the wandb run ============
wandb.finish()
# =============================================
