# One-Step Generation

## Overview

This application implements fast generative modeling using flow matching with linearized transformations. It enables both one-step and multi-step generation, along with inversion capabilities which we demonstrate though interpolation in the latent space.

## Training a Model

### For MNIST Dataset

```bash
python train_one_step.py mnist 
```

### For CelebA Dataset

```bash
python train_one_step.py celeba 
```

### Optional Training Parameters

- `--batch_size`: Training batch size (default: 64 for MNIST, 32 for CelebA)
- `--epc`: Number of training epochs (default: 200)
- `--img_size`: Image resolution (32 for MNIST, 64 for CelebA)
- `--steps`: Number of sampling steps (default: 100)
- `--sampling_method`: Sampling method ('euler' or 'rk')
- `--save_folder`: Output directory for models and artifacts

### Training output
The training will output:
1. Artifacts of samples across the training process of one step and multiple steps each 10 epochs.
You can change the code if you want output less/more frquenctly.
2. The model '.pth' state dict
3. A json containing the arguments used for the training

## Testing and Inference

After training, use the test script to generate samples and perform inversion interpolations:



```bash
python test_one_step.py --model_path /path/to/you/model.pth
```
 Note: the script automatically scan the folder structure and load the arguments json file.


### What the Test Script Does

1. **Sample Generation**: 
   - Generates 24 samples using both one-step and multi-step methods
   - Saves results as 4x6 grids in PDF format
   - Compares quality using PSNR and LPIPS metrics

![Example](../figs/one_step_example.png)


2. **Interpolation Experiments**:
   - Performs latent space interpolation between image pairs from the data.
   - Evaluates reconstruction quality

![Example](../figs/inersion_interpolation_example.png)


### Expected Outputs

- `generated_samples_{dataset}_one_step.pdf`: One-step generated samples
- `generated_samples_{dataset}_multi_step.pdf`: Multi-step generated samples  
- `interpolation_result_{pairs}.pdf`: Interpolation results
- Console output with PSNR and LPIPS metrics
