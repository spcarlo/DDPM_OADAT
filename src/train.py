import os
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import wandb
from tqdm import tqdm

from src.forward_diffusion import forward_diffusion_sample
from src.save_load_model import save_model, load_model_continuing
from src.image_generation import grid_plot



def p_losses(denoise_model, x_0, t, diffusion_variables, noise=None, loss_type="l1"):
    """
    Compute the loss between the predicted noise and the actual noise for the diffusion model.
    
    denoise_model: The U-Net model that predicts noise.
    x_0: Clean images (original).
    t: Timestep (integer).
    diffusion_variables: Precomputed diffusion variables.
    noise: The actual noise added to the images (if None, random noise will be generated).
    loss_type: Type of loss to compute ('l1', 'l2', 'huber').
    """
    
    if noise is None:
        noise = torch.randn_like(x_0)  # Generate Gaussian noise

    # Get the noisy images by performing the forward diffusion
    x_noisy, _ = forward_diffusion_sample(x_0, t=t, diffusion_variables=diffusion_variables, noise=noise)

    # Use the model to predict the noise at timestep t
    predicted_noise = denoise_model(x_noisy, t)

    # Choose the loss type
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)  # L1 loss (robust to outliers, slower convergence)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)  # L2 loss (sensitive to all errors)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)  # Huber loss (combo of L1 and L2)
    else:
        raise NotImplementedError(f"Unknown loss type: {loss_type}")

    return loss


def sample_timesteps(batch_size, timesteps):
    return torch.randint(0, timesteps, (batch_size,))



def train_model(model, dataloader, optimizer, diffusion_variables, timesteps, epochs, device, 
                config_params, model_config, loss_type):
    
    model.train()

    # ** CHECKPOINT DIRECTORY **
    # ===========================
    save_dir = f'results/checkpoint/{wandb.run.project}/{wandb.run.name}'
    os.makedirs(save_dir, exist_ok=True)
    # **************************


    total_batches = len(dataloader)

    for epoch in range(epochs):
        
        # log interval for wandb
        log_interval = 500 // dataloader.batch_size

        for batch_idx, batch in enumerate(dataloader):
            x_0 = batch.to(device)
            optimizer.zero_grad()

            # Sample random timesteps
            t = sample_timesteps(x_0.shape[0], timesteps).to(device)

            # ====================================================================
            loss = p_losses(model, x_0, t, diffusion_variables, loss_type=loss_type)
            loss.backward()
            optimizer.step()
            # ============================================================================

            # === Log to WandB every log_interval ===
            if batch_idx % log_interval == 0:
                try:
                    wandb.log({"epoch": epoch+1, "loss": loss.item()})
                except Exception as e:
                    print(f"Warning: Failed to log to WandB")
            #  ======================================

            # consol print
            if batch_idx % 10 == 0:
                sys.stdout.write(f'\rEpoch {epoch+1}/{epochs}, Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}')
                sys.stdout.flush() 

        # ============== SAVE CHECKPOINT EVERY EPOCH =====================
        model_name = "checkpoint.pth"
        image_name = f'grid_epoch-{epoch+1}'

        save_model(epoch + 1, model, optimizer, 
                    config_params, diffusion_variables, model_config, 
                    save_dir, model_name)

        # Generate and save images
        checkpoint_path = os.path.join(save_dir, model_name)
        grid_img, _ = grid_plot(checkpoint_path, nrow=2, batch_size=4)
        save_image(grid_img, f'{save_dir}/{image_name}.png')

        torch.cuda.empty_cache()
        # ============== ============ ============= ======================






def continuing_training(checkpoint_path, dataloader, start_epoch, epochs, 
                        device, loss_type = 'huber'):
    
    model, optimizer, config_params, model_config, diffusion_variables = load_model_continuing(checkpoint_path)
    model.train()
    
    timesteps = config_params['timesteps']

    # ** CHECKPOINT DIRECTORY **
    # ===========================
    save_dir = f'results/checkpoint/{wandb.run.project}/{wandb.run.name}'
    os.makedirs(save_dir, exist_ok=True)
    # **************************


    total_batches = len(dataloader)
    
    for epoch in range(start_epoch, epochs):
        
        # log interval for wandb
        log_interval = 500 // dataloader.batch_size
        
        for batch_idx, batch in enumerate(dataloader):
            x_0 = batch.to(device)
            optimizer.zero_grad()

            # Sample random timesteps
            t = sample_timesteps(x_0.shape[0], timesteps).to(device)

            # ==========================================
            loss = p_losses(model, x_0, t, diffusion_variables, loss_type=loss_type)
            loss.backward()
            optimizer.step()
            # ==========================================
            
            # === Log to WandB every log_interval ===
            if batch_idx % log_interval == 0:
                try:
                    wandb.log({"epoch": epoch+1, "loss": loss.item()})
                except Exception as e:
                    print(f"Warning: Failed to log to WandB")
            #  ======================================
            
            # consol print
            if batch_idx % 10 == 0:
                sys.stdout.write(f'\rEpoch {epoch+1}/{epochs}, Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}')
                sys.stdout.flush() 

        
        # ============== SAVE CHECKPOINT EVERY EPOCH =====================
        model_name = "checkpoint.pth"
        image_name = f'grid_epoch-{epoch + 1}'


        # Attempt to save to the primary directory
        save_model(epoch + 1, model, optimizer, config_params, diffusion_variables, model_config, 
                    save_dir, model_name)
        
        # Generate and save images
        checkpoint_path = os.path.join(save_dir, model_name)
        grid_img, _ = grid_plot(checkpoint_path, nrow=2, batch_size=4)
        save_image(grid_img, f'{save_dir}/{image_name}.png')       
        # ================================================================
            
        torch.cuda.empty_cache()


