import os.path as osp
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor, CenterCrop, Normalize, Compose
import imageio
import glob
import random

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


def set_seeds(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data(config, take=1, sampling='random', sampling_list=None, device=torch.device('cuda'), seed=1234):
    files = sorted(glob.glob(osp.join(config['data_dir'], "*")))

    # select
    if sampling == 'random':
        sample = random.sample(range(len(files)), take)
    elif sampling == 'custom' and sampling_list is not None:
        sample = sampling_list
    else:
        raise ValueError(f"Invalid sampling method: {sampling} and sampling_list: {sampling_list}")

    with open(f"{config['log_dir']}/config_{seed}.txt", 'a') as f:
        flag = 'Randomly' if sampling == 'random' else 'Custom'
        
        if take > 1:
            # train시에만 verbose
            f.write(f'{flag} selected {take} images from {len(files)} images: \n{sample}')
            print(f'{flag} selected {take} images from {len(files)} images: \n{sample}')        

    files = [files[i] for i in sample]
    images = []
    for fname in files:
    
        pilmode='RGB' if config['out_channels'] == 3 else 'L'
        img = np.array(imageio.imread(fname, pilmode=pilmode), dtype=np.float32) / 255.   # [H, W, C], [0, 1]
        if img.ndim == 2:
            # 그레이스케일 이미지일 때
            img = np.expand_dims(img, axis=-1)  # C 차원을 추가하여 (1, H, W)로 만듦
            
        H, W, _ = img.shape

        aug_list = [
                ToTensor(),
                CenterCrop(min(H, W)),
                Resize(config['image_size']),
        ]
        if config['zero_mean']:
                aug_list.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))

        transform = Compose(aug_list)
        img = transform(img).permute(1, 2, 0)

        H, W, C = img.shape
        images.append(img)

    return torch.stack(images).float().to(device)

def get_coords(H, W, T=None, device=torch.device('cuda')):
    if T is None:
        x = torch.linspace(-1, 1, W).to(device)
        y = torch.linspace(-1, 1, H).to(device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    else:
        X, Y, Z = np.meshgrid(np.linspace(-1, 1, W),
                              np.linspace(-1, 1, H),
                              np.linspace(-1, 1, T))
        coords = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))
        coords = torch.tensor(coords.astype(np.float32)).to(device)
    return coords

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def preprocess_raw_image(x, enc_type, img_size):
    '''
    input x는 get_data를 통해 얻은 입력으로, (B, HW, C) shape
    Encoder(DINOvw) 입력은  B, nc, w, h 
    '''
    x = x.transpose(1, 2)                       # B x C x WH
    x = x.reshape(-1, 3, img_size, img_size)    # B x C x W x H
    resolution = x.shape[-1]    # H or W
    if 'dinov2' in enc_type:
        # Already x is in [0, 1] in get_data function
        # x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        enc_input_sidelen = 224 * (resolution // 256)
        x = torch.nn.functional.interpolate(x, enc_input_sidelen, mode='bicubic')
    else:
        raise NotImplementedError("You must specify the correct encoder type: [dinov2]")
    return x

def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)

        if 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
            
        else:
            raise NotImplementedError(f"Encoder type {encoder_type} is not implemented.")

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures