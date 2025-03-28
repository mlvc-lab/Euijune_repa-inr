import os.path as osp
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor, CenterCrop, Normalize, Compose
import imageio
import glob
import random


def get_train_data(log_dir, path, zero_mean=True, sidelen=256, out_feature=3, take=10, device=torch.device('cuda'), seed=1234):
    files = sorted(glob.glob(osp.join(path, "*")))

    # Randomly select
    sample = random.sample(range(len(files)), take)

    with open(f'{log_dir}/config_{seed}.txt', 'w') as f:
        f.write(f'Randomly selected {take} images from {len(files)} images: \n{sample}')
    print(f'Randomly selected {take} images from {len(files)} images: \n{sample}')        

    files = [files[i] for i in sample]
    images = []
    for fname in files:
    
        pilmode='RGB' if out_feature == 3 else 'L'
        img = np.array(imageio.imread(fname, pilmode=pilmode), dtype=np.float32) / 255.   # [H, W, C], [0, 1]
        if img.ndim == 2:
            # 그레이스케일 이미지일 때
            img = np.expand_dims(img, axis=-1)  # C 차원을 추가하여 (1, H, W)로 만듦
            
        H, W, _ = img.shape

        aug_list = [
                ToTensor(),
                CenterCrop(min(H, W)),
                Resize((sidelen, sidelen)),
        ]
        if zero_mean:
                aug_list.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))

        transform = Compose(aug_list)
        img = transform(img).permute(1, 2, 0)

        H, W, C = img.shape

        #gt = img.view(-1, C)
        images.append(img)
        #print(img.shape)

    return torch.stack(images).float().to(device)

def get_test_data(path, zero_mean=True, sidelen=256, out_feature=3, idx=0, device=torch.device('cuda')):
    files = sorted(glob.glob(osp.join(path, "*")))
    file = files[idx]
    images = []
    
    pilmode='RGB' if out_feature == 3 else 'L'
    img = np.array(imageio.imread(file, pilmode=pilmode), dtype=np.float32) / 255.   # [H, W, C], [0, 1]
    if img.ndim == 2:  # 그레이스케일 이미지일 때
        img = np.expand_dims(img, axis=-1)  # C 차원을 추가하여 (1, H, W)로 만듦
        
    H, W, _ = img.shape

    aug_list = [
            ToTensor(),
            CenterCrop(min(H, W)),
            Resize((sidelen, sidelen)),
    ]
    if zero_mean:
            aug_list.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))

    transform = Compose(aug_list)
    img = transform(img).permute(1, 2, 0)

    H, W, C = img.shape

    #gt = img.view(-1, C)
    images.append(img)
    #print(img.shape)

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