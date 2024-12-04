# OADAT_DDPM

This repository implements **Denoising Diffusion Probabilistic Models (DDPM)** for generative modeling on the [OADAT dataset](https://www.research-collection.ethz.ch/handle/20.500.11850/551512). The codebase is adapted from [lucidrains / denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).

---

## Code Structure

### Core Scripts
- **`main_train.py`**: Configure parameters and train the model.
- **`main_train_cont.py`**: Resume training from checkpoints.

### Directory Structure
- **`src/`**: Contains core modules:
  - `train.py`: Training pipeline.
  - `data_loader.py`: Data preparation and loading.
  - `forward_diffusion.py`: Implements the forward diffusion process.
  - `backward_diffusion.py`: Implements the backward diffusion process.
  - `image_generation.py`: Generates and visualizes samples (grid plots).
  - `save_load_model.py`: Handles saving and loading of model checkpoints.
  - `u_net_blocks.py`: Building blocks for the U-Net architecture.
  - `helper.py`: Utility functions.
  - `u_net.py`: U-Net model architecture.
- **`notebooks/`**: Jupyter notebooks for exploration:
  - `data_expo.ipynb`: Data exploration.
  - `timescheduler.ipynb`: Time-scheduling insights.
  - `forward.ipynb`: Forward process analysis.
  - `backward.ipynb`: Random sampling from the model.
  - `backward_conditional.ipynb`: Conditional sampling based on SCD_vc.
  - `backward_conditional_sketch.ipynb`: Conditional sampling based on sketches.

---

## Usage

### Training

Run **`main_train.py`**. Parameters can be set in this file. Adjust dataset location and checkpoint location accordingly.

### Sampling

Use `backward.ipynb` for random sampling, and `backward_conditional.ipynb` for conditional sampling a trained model.

---

