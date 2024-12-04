import torch
import os

from src.u_net import Unet


def save_model(epoch, model, optimizer, config_params, diffusion_variables, model_config, save_dir, model_name):
    """
    Saved model, optimizer, and other necessary configurations.
    """
    save_path = os.path.join(save_dir, model_name)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config_params': config_params,
        'diffusion_variables': diffusion_variables,
        'epoch': epoch,
        'model_config': model_config
    }, save_path)



def load_model_img_gen(checkpoint_path):
    """
    Load the saved model, optimizer, and other necessary configurations 
    for image generation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration and model parameters
    config_params = checkpoint['config_params']
    model_config = checkpoint['model_config']
    # print(model_config)
    # print(config_params)
    
    # Initialize the U-Net using model_config parameters
    model = Unet(**model_config).to(device)

    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval() 
    

    return model, config_params, checkpoint['diffusion_variables']



def load_model_continuing(checkpoint_path):
    """
    Load the saved model, optimizer, and other necessary configurations 
    for continuing training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config_params = checkpoint['config_params']
    model_config = checkpoint['model_config']
    diffusion_variables = checkpoint['diffusion_variables']
    
    model = Unet(**model_config).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config_params['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.train()
    
    return model, optimizer, config_params, model_config, diffusion_variables




