# Linearizer Project

## Overview

![Linearizer Architecture](figs/the_linearizer.png)

Neural networks are famously nonlinear, but linearity is defined relative to vector spaces f:X→Y. This project introduces **Linearizers**—architectures that sandwich a linear operator A between two invertible neural networks: f(x) = g⁻¹ᵧ(Agₓ(x)).

This framework makes linear algebra tools (SVD, pseudo-inverse, projections) applicable to nonlinear mappings, enabling:

- **One-Step Generation**: Collapse diffusion model sampling from hundreds of steps to one
- **Style Transfer**: Modular artistic style transfer using linear transformations
3. IGN - TODO 

We provide, in this repository, for all of these 3 applications full implementations.


## Project Structure

```
linearizer/
├── one_step/           # One-step generation application
│   ├── data/          # Data loading utilities
│   ├── modules/       # Core model components
│   ├── utils/         # Training and sampling utilities
│   ├── train_one_step.py
│   └── test_one_step.py
├── style_transfer/    # Style transfer application
│   ├── modules/       # Style transfer models
│   ├── utils/         # Visualization utilities
│   ├── train_style_transfer.py
│   └── style_intrepolations.py
└── common/           # Shared components
    └── song__unet.py # UNet architecture
```

## Installation Requirements

### Python Environment

1. **Create conda environment**:
   ```bash
   conda create -n linearizer python=3.9
   conda activate linearizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

Navigate to the specific application directories (`one_step/`, `ign/` or `style_transfer/`) for detailed usage instructions.
