import wandb
import numpy as np
import torch
from pprint import pprint

from src.data_loader import create_dataloader 
from src.train import continuing_training
from src.save_load_model import load_model_continuing


# Continuing parameters
start_epoch     = 15 # same as last epoch that finished
epochs          = 30 
batch_size      = 5
model_name      = "Cosine_05-test"
wandb_project   = "test"

# Load model configuration parameters from checkpoint
print(">>> load checkpoint")
checkpoint_path = f"results/models/{model_name}/{model_name}.pth"
_, _, config_params, model_config, _ = load_model_continuing(checkpoint_path)


# ============== NORM =========================
norm = "norm1_scaleclip"
#  scaleclip [-0.2,1], norm1_scaleclip [-1,1], norm2_scaleclip [0,1]
norm_params = {"norm": norm}
# =============================================


# ========= Initialize WandB logging ==========
combined_config = {**config_params, **model_config, **norm_params}
wandb.init(
    project = wandb_project,
    entity  = "carlospeckert-eth-z-rich",
    name    = model_name,  
    notes   = "**",
    config  = combined_config
)
# =============================================


# =========== LOAD DATA ========================
# OADAT location
oadat_dir = "C:/Users/carlo/OADAT"

file_name = "SWFD_semicircle_RawBP.h5"
key = "sc_BP"

prng = np.random.RandomState(42)
indices = None  # Use full dataset or specify subset
# indices = np.arange(xxx) # Use only the first xxx

# Initialize DataLoader
dataloader = create_dataloader(oadat_dir, file_name, key, norm_params["norm"], batch_size, 
                                shuffle=True, num_workers=0, prng=prng, indices=indices)
# =============================================

# =========== DEVICE SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===================== INFO =========================")
print(f"Device: {device}")
pprint(combined_config, width=60)
print("Continuous Training Initialized")
print("============== CONTINUING TRAINING =================")
# =============================================


# ************** TRAIN *****************
continuing_training(checkpoint_path, dataloader, start_epoch, epochs, 
                    device, loss_type = "huber")
#  *************************************
print("=============== TRAINING FINISHED =================")

# ========= Finalize the wandb run ============
wandb.finish()
# =============================================
