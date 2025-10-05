import argparse

# CelebA dataset configuration
CELEBA_CONFIG = {
    'dataset': 'celeba',
    'img_size': 64,
    'batch_size': 16,
    'batch_size_val': 128,
    'epc': 201,
    'lr': 1e-4,
    'noise_level': 0.3,
    'steps': 1000,
    'sampling_method': 'rk',
    'linear_module': 'TimeDependentLoRALinearLayer',
    'g': 'InverseUnet',
    'num_of_layers': 3,
    'in_ch': 3,
    'out_ch': 3,
    'linear_lora_features': 8,
    'save_folder': './outputs_new'
}


def get_celeba_parser():
    parser = argparse.ArgumentParser(description='CelebA Flow Matching Training')

    # Artifacts
    parser.add_argument('--save_folder', type=str, default=CELEBA_CONFIG['save_folder'])

    # Model parameters
    parser.add_argument('--linear_module', type=str, default=CELEBA_CONFIG['linear_module'],
                        choices=['TimeDependentLoRALinearLayer', ])
    parser.add_argument('--g', type=str, default=CELEBA_CONFIG['g'], choices=['InverseUnet'])
    parser.add_argument('--num_of_layers', type=int, default=CELEBA_CONFIG['num_of_layers'])
    parser.add_argument('--in_ch', type=int, default=CELEBA_CONFIG['in_ch'])
    parser.add_argument('--out_ch', type=int, default=CELEBA_CONFIG['out_ch'])

    # linear module
    parser.add_argument('--linear_lora_features', type=int, default=CELEBA_CONFIG['linear_lora_features'])

    # Data and training
    parser.add_argument('--dataset', type=str, default=CELEBA_CONFIG['dataset'])
    parser.add_argument('--epc', type=int, default=CELEBA_CONFIG['epc'])
    parser.add_argument('--batch_size', type=int, default=CELEBA_CONFIG['batch_size'])
    parser.add_argument('--batch_size_val', type=int, default=CELEBA_CONFIG['batch_size_val'])
    parser.add_argument('--img_size', type=int, default=CELEBA_CONFIG['img_size'])

    # Loss and training parameters
    parser.add_argument('--noise_level', type=float, default=CELEBA_CONFIG['noise_level'])
    parser.add_argument('--lr', type=float, default=CELEBA_CONFIG['lr'])

    # Sampling parameters
    parser.add_argument('--steps', type=int, default=CELEBA_CONFIG['steps'])
    parser.add_argument('--sampling_method', type=str, default=CELEBA_CONFIG['sampling_method'])

    return parser
