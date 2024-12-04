import torch



# ==================== TIMESCHEDULERS ====================
# Smooth and balanced, Carful parameter tuning of s
def cosine_beta_schedule(timesteps, s=0.008, beta_end=0.9999):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas / betas[-1] * beta_end # NEW VERSION!!!
    return torch.clip(betas, 0.0001, 0.9999)

# Fast and simple. good baseline
def linear_beta_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# better for high detail, tuning...
def quadratic_beta_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

# middle ground ... tuning. more control but carfull tuning compare to cosine.
def sigmoid_beta_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):  
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# =======================================================================



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



# ==================== FORWARD DIFFUSION ====================
def forward_diffusion_sample(x_0, t, diffusion_variables, noise=None):
    """
    Perform forward diffusion
    
    x_0: Original images.
    t: Integer timestep (0 to timesteps - 1) or batch of timesteps.
    diffusion_variables: Precomputed diffusion variables (alphas, etc.).
    """
    # Extract precomputed variables
    sqrt_alphas_cumprod = diffusion_variables['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = diffusion_variables['sqrt_one_minus_alphas_cumprod']

    if noise is None:
        noise = torch.randn_like(x_0)  # Generate Gaussian noise

    if isinstance(t, int):
        # If t is an integer (single timestep), directly index
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]

        # Reshape to broadcast across x_0
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(1, 1, 1, 1)
    else:
        # If t is a tensor (batch of timesteps), use extract function
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # Apply forward diffusion
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise
# =======================================================================


