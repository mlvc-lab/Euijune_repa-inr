import torch
import torch.nn as nn

seed = 1234
torch.manual_seed(1234)

import os, os.path as osp
import numpy as np
import skimage, skimage.transform, skimage.io, skimage.filters
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
from collections import defaultdict, OrderedDict
import math
import cv2
import random
from copy import deepcopy

from torchvision.transforms import Resize, ToTensor, CenterCrop, Normalize, Compose
import imageio

import glob
device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.

        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.

        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class INR(nn.Module):
    def __init__(self, in_features, hidden_features,
                 hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True, no_init=False):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = SineLayer

        self.net = []
        if hidden_layers != 0:
        # append first sine layer
            self.net.append(self.nonlin(in_features, hidden_features,
                                    is_first=True, omega_0=first_omega_0,
                                    scale=scale, init_weights=(not no_init)))
        hidden_layers = hidden_layers -1 if (hidden_layers > 0 and outermost_linear is True) else hidden_layers
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale, init_weights=(not no_init)))

        if outermost_linear or (hidden_layers == 0):
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)

            if not no_init:
                with torch.no_grad():
                    const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                    final_linear.weight.uniform_(-const, const)

            self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)

        output = self.net(coords)

        return output

class SharedINR(nn.Module):


    def __init__(self, in_features, hidden_features,
                 hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True, shared_encoder_layers=None, num_decoders=None, no_init=False):

        super().__init__()
        assert shared_encoder_layers is not None, "Please mention shared_encoder_layers. Use 0 if none are shared"
        assert hidden_layers > shared_encoder_layers, "Total hidden layers must be greater than number of layers in shared encoder"
        self.shared_encoder_layers = shared_encoder_layers
        self.num_decoders = num_decoders

        self.encoderINR = INR(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=self.shared_encoder_layers - 1, # input is a layer
            out_features=hidden_features,
            outermost_linear=False,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=scale,
            pos_encode=pos_encode,
            sidelength=sidelength,
            fn_samples=fn_samples,
            use_nyquist=use_nyquist,
            no_init=no_init
        )

        self.num_decoder_layers = hidden_layers - self.shared_encoder_layers
        assert self.num_decoder_layers >= 1 , "Num decoder layers must be more than 1"
        self.decoderINRs = nn.ModuleList([
                                            INR(
                                                in_features=hidden_features,
                                                hidden_features=hidden_features,
                                                hidden_layers=self.num_decoder_layers - 1,
                                                out_features=out_features,
                                                outermost_linear=outermost_linear,
                                                first_omega_0=first_omega_0,
                                                hidden_omega_0=hidden_omega_0,
                                                scale=scale,
                                                pos_encode=pos_encode,
                                                sidelength=sidelength,
                                                fn_samples=fn_samples,
                                                use_nyquist=use_nyquist,
                                                no_init=no_init

                                            ) for i in range(self.num_decoders)])

    def forward(self, coords):
        encoded_features = self.encoderINR(coords)
        outputs = []
        for _idx, _decoder in enumerate(self.decoderINRs):
            output = _decoder(encoded_features)
            outputs.append(output)

        return outputs

    def load_encoder_weights_from(self, fellow_model):
        self.encoderINR.load_state_dict(deepcopy(fellow_model.encoderINR.state_dict()))

    def load_weights_from_file(self, file, key="encoderINR"):
        weights = torch.load(file)
        self.encoderINR.load_state_dict(deepcopy(weights['encoder_weights']))

IMG_SIZE = (256,256)
POS_ENCODE = False

config = defaultdict()
config['epochs']=2000
#config['epochs_train_strainer'] = 501
config['learning_rate'] = 1e-4
config['plot_every'] = 100
config['image_size'] = IMG_SIZE

# INR params
config['num_layers'] = 5
config['hidden_features'] = 256
config['in_channels'] = 2
config['out_channels'] = 3
config['shared_encoder_layers'] = 4
config['num_decoders'] = 1
config['nonlin'] = 'siren'

TRAINING_PATH = "/local_dataset/DIV2K"
TESTING_PATH = "/local_dataset/Urban_100"

def get_train_data(path, zero_mean=True, sidelen=256, out_feature=3, take=10):
    files = sorted(glob.glob(osp.join(path, "*")))

    # Randomly select
    sample = random.sample(range(len(files)), take)

    with open(f'logs_STRAINER/config_{seed}.txt', 'w') as f:
        f.write(f'Randomly selected {take} images from {len(files)} images: \n{sample}')
    print(f'Randomly selected {take} images from {len(files)} images: \n{sample}')        

    files = [files[i] for i in sample]
    images = []
    for fname in files:
    
        pilmode='RGB' if out_feature == 3 else 'L'
        img = np.array(imageio.imread(fname, pilmode=pilmode), dtype=np.float32) / 255.   # [H, W, C], [0, 1]
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

def get_test_data(path, zero_mean=True, sidelen=256, out_feature=3, idx=0):
    files = sorted(glob.glob(osp.join(path, "*")))
    file = files[idx]
    images = []
    
    print(file)
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

def get_coords(H, W, device=torch.device('cuda'), T=None):
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

def fit_inr(coords, data, model, optim, config={}, mlogger=None, name=None):
    assert name is not None, "`name` must be provided as metric logger needs it"
    gt_tensor = data['gt']

    best_loss = np.inf
    best_epoch = 0

    tbar = tqdm(range(config['epochs']))
    psnr_vals = []
    out_mins = []
    out_maxs = []
    for epoch in tbar:
        outputs = model(coords) # 10 x 1 x (HxW) x 3
        output = outputs[0] if isinstance(outputs, list) else outputs
        n, p, ldim = output.shape

        loss = ((output - gt_tensor)**2).mean() # works per image
        optim.zero_grad()
        loss.backward()
        optim.step()

        # PSNR 계산을 위해 추가
        out = output/2+0.5
        gt = gt_tensor/2+0.5

        psnr = -10*torch.log10(((out - gt)**2).mean()) #  config['image_size'][0],  config['image_size'][1])
        psnr_vals.append(float(psnr))

        tbar.set_description(f"Iter {epoch}/{config['epochs']} Loss = {loss.item():6f} PSNR = {psnr:.4f}")
        tbar.refresh()

    return {
        "psnr" : psnr_vals,
        "state_dict" : model.state_dict()
    }

def shared_encoder_training(coords, data, model, optim, config={}, mlogger=None, name=None):
    assert name is not None, "`name` must be provided as metric logger needs it"
    gt_tensor = data['gt']

    best_loss = np.inf
    best_epoch = 0

    tbar = tqdm(range(config['epochs']))
    psnr_vals = []

    for epoch in tbar:
        outputs = model(coords) # 10 x 1 x (HxW) x 3
        stacked_outputs = torch.stack(outputs, dim=0)
        stacked_gt = torch.stack(gt_tensor, dim=0)
        loss = ((stacked_outputs - stacked_gt)**2).mean(dim=[1,2,3]).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        tbar.set_description(f"Iter {epoch}/{config['epochs']} Loss = {loss.item():6f}")
        tbar.refresh()

    return {
        "psnr" : psnr_vals,
        "state_dict" : model.state_dict()
    }




if __name__ == '__main__':
    coords = get_coords(*IMG_SIZE, device=device)
    print(coords.min(), coords.max())

    im_tensor_train = get_train_data(TRAINING_PATH, take=10)

    data_dict_train_strainer = {'image_size':IMG_SIZE, 'gt':[x.reshape(1, -1, 3) for x in im_tensor_train]}
    data_dict_1 = {'image_size':IMG_SIZE, 'gt':im_tensor_train[1].reshape(1, -1, 3)}

    #  STRAINER 1 decoder 학습
    inr_siren_vanilla1_random_train = SharedINR(in_features=config['in_channels'],
                                hidden_features=config['hidden_features'], hidden_layers=config['num_layers'],
                                shared_encoder_layers = config['shared_encoder_layers'],
                                num_decoders=config['num_decoders'],
                                out_features=config['out_channels']).to(device)
    optim_siren_vanilla1_random_train = torch.optim.Adam(lr=config['learning_rate'], params=inr_siren_vanilla1_random_train.parameters())

    config_ft = deepcopy(config)
    config_ft['epochs'] = 5000

    print('\nTraining STRAINER 1 decoder\n')
    _ = fit_inr(coords=coords, data=data_dict_1,
                model=inr_siren_vanilla1_random_train,
                optim=optim_siren_vanilla1_random_train,
                config=config_ft, mlogger=None,name="random_vanilla_train")
    
    # STRAINER 1 decoder 테스트 (Urban-100)    
    ret_strainer_1decoder_test = {}

    print('\nTesting STRAINER 1 decoder\n')
    for idx in range(100):
        im_tensors = get_test_data(TESTING_PATH, idx=idx)
        data_dict_test_strainer_1 = {'image_size':IMG_SIZE, 'gt':im_tensors[0].reshape(1, -1, 3)}  # for 10 decoder STRAINER

        inr_strainer_1decoder = SharedINR(in_features=config['in_channels'],
                                hidden_features=config['hidden_features'], hidden_layers=config['num_layers'],
                                shared_encoder_layers = config['shared_encoder_layers'],
                                num_decoders=config['num_decoders'],
                                out_features=config['out_channels']).to(device)

        inr_strainer_1decoder.load_encoder_weights_from(inr_siren_vanilla1_random_train)
        optim_siren_strainer1decoder = torch.optim.Adam(lr=config['learning_rate'], params=inr_strainer_1decoder.parameters())    
        ret_strainer_1decoder_test[str(idx+1).zfill(2)] = fit_inr(coords=coords, data=data_dict_test_strainer_1,
                                        model=inr_strainer_1decoder,
                                        optim=optim_siren_strainer1decoder,
                                        config=config, mlogger=None,name="strainer_encoder_only_1decoder")
        
    torch.save(ret_strainer_1decoder_test,f'./logs_STRAINER/ret_strainer_1decoder_test_seed{seed}.pt')
    

    # STRAINER 10 decoder 학습
    inr_strainer_10decoders_train = SharedINR(in_features=config['in_channels'],
                                hidden_features=config['hidden_features'], hidden_layers=config['num_layers'],
                                shared_encoder_layers = config['shared_encoder_layers'],
                                num_decoders=10,
                                out_features=config['out_channels']).to(device)
    optim_siren_strainer10decoder_train = torch.optim.Adam(lr=config['learning_rate'], params=inr_strainer_10decoders_train.parameters())

    print('\nTraining STRAINER 10 decoder\n')
    config_train = deepcopy(config)
    config_train['epochs'] = 5000
    ret_strainer10decoder_train = shared_encoder_training(coords=coords, data=data_dict_train_strainer,
                                                            model=inr_strainer_10decoders_train,
                                                            optim=optim_siren_strainer10decoder_train,
                                                            config=config_train, mlogger=None,name="strainer_encoder_only_10decoder")
    
    # STRAINER 10 decoder 테스트 (Urban-100)
    ret_strainer_10decoder_test = {}

    print('\nTesting STRAINER 10 decoder\n')

    for idx in range(100):
        im_tensors = get_test_data(TESTING_PATH, idx=idx)
        data_dict_test_strainer_10 = {'image_size':IMG_SIZE, 'gt':im_tensors[0].reshape(1, -1, 3)}  # for 10 decoder STRAINER

        inr_strainer_test = SharedINR(in_features=config['in_channels'],
                                hidden_features=config['hidden_features'], hidden_layers=config['num_layers'],
                                shared_encoder_layers = config['shared_encoder_layers'],
                                num_decoders=config['num_decoders'],
                                out_features=config['out_channels']).to(device)

        inr_strainer_test.load_encoder_weights_from(inr_strainer_10decoders_train)
        optim_siren_strainer_test = torch.optim.Adam(lr=config['learning_rate'], params=inr_strainer_test.parameters())

        ret_strainer_10decoder_test[str(idx+1).zfill(2)] = fit_inr(coords=coords, data=data_dict_test_strainer_10,
                                                                model=inr_strainer_test,
                                                                optim=optim_siren_strainer_test,
                                                                config=config, mlogger=None,name=f"strainer_test_{idx}img")
        
    
    torch.save(ret_strainer_10decoder_test,f'./logs_STRAINER/ret_strainer_10decoder_test_seed{seed}.pt')
    print(f"successfully saved in /logs_STRAINER/ret_strainer_10decoder_test_seed{seed}.pt")

