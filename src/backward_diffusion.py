import torch
from tqdm import tqdm
import torch.nn.functional as F



def extract(a, t, x_shape):
    """
    Extract values at specific timesteps t for each sample in the batch.
    
    a: Tensor of diffusion variables (e.g., sqrt_alphas_cumprod).
    t: Tensor of timesteps (batch_size,).
    x_shape: Shape of the input tensor x_0 to ensure correct broadcasting.
    
    Returns: Tensor reshaped for broadcasting.
    """
    batch_size = t.shape[0]
    
    # Ensure the diffusion variable tensor a is on the same device as t
    a = a.to(t.device)
    
    out = a.gather(-1, t)  # Gather values along the last dimension
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # Reshape for broadcasting



@torch.no_grad()
def p_sample(model, x, t, t_index, timesteps, diffusion_variables):
    """
    Perform one reverse diffusion step using precomputed variables.
    """
    # Extract diffusion variables
    betas = 1 - diffusion_variables['alphas']
    sqrt_one_minus_alphas_cumprod = diffusion_variables['sqrt_one_minus_alphas_cumprod']
    sqrt_recip_alphas = diffusion_variables['sqrt_recip_alphas']
    posterior_variance = diffusion_variables['posterior_variance']
    
    # Extract values for the current timestep
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Use the model to predict the noise (denoising)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    # If it's the final step (t=0), return the predicted clean image
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, x_shape, timesteps, diffusion_variables, x_start=None, denoise_steps=None):
    """
    Perform the reverse diffusion process.

    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine if starting from pure noise or from a provided noisy image
    if x_start is None:
        # Start from pure noise if no starting image is provided
        img = torch.randn(x_shape, device=device)
        denoise_steps = timesteps - 1
    else:
        # Use the provided noisy image and start from the specified step
        img = x_start.clone().to(device)
        if denoise_steps is None:
            raise ValueError("If x_start is provided, denoise_steps must be specified.")
    
    imgs = []

    # Perform reverse diffusion from `denoise_steps` back to 0
    for i in reversed(range(0, denoise_steps + 1)):
        img = p_sample(
            model, img, torch.full((x_shape[0],), i, device=device, dtype=torch.long), 
            i, timesteps, diffusion_variables
        )
        imgs.append(img.cpu())  # Store images at each time step for visualization or further processing

    return imgs

@torch.no_grad()
def sample(model, x_shape, timesteps, diffusion_variables, x_start=None, denoise_steps=None):
    """
    Wrapper function to sample images from the reverse diffusion process.
    Automatically infer shape from x_0.
    """
    
    model.eval()
    return p_sample_loop(
        model, x_shape=x_shape, timesteps=timesteps, diffusion_variables=diffusion_variables, 
        x_start=x_start, denoise_steps=denoise_steps
    )
