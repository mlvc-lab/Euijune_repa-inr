import os.path as osp
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor, CenterCrop, Normalize, Compose
import imageio
import glob
import random


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