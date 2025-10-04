from common.song__unet import creat_song_unet
from one_step.modules.invertable_network import InverseUnet
from one_step.modules.linear_network import TimeDependentLoRALinearLayer


def get_linear_network(linear_module, in_ch=1, linear_lora_features=128, t_size=128, img_size=32):
    """Create linear module based on type"""
    if linear_module == 'TimeDependentLoRALinearLayer':
        model = TimeDependentLoRALinearLayer(img_size ** 2 * in_ch, linear_lora_features, t_size)
    else:
        raise NotImplementedError(f'Linear module {linear_module} not implemented')
    return model


def get_g(g_type, num_of_layers, in_ch, img_resolution):
    if g_type == 'InverseUnet':
        g = InverseUnet(num_of_layers, in_ch, img_resolution, creat_song_unet)
    else:
        raise NotImplementedError(f'Projector {g_type} not implemented')
    return g


def get_latent_size(dataset, size):
    """Calculate latent size based on dataset and image size"""
    if 'mnist' in dataset:
        return size ** 2
    elif 'celeb' in dataset:
        return size ** 2 * 3
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
