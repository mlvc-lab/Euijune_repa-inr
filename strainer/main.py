import torch
import datetime
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
from model import STRAINER
from collections import defaultdict
from train import fit_inr
from utils import get_data, get_coords, set_seeds

seed=1234
set_seeds(seed=seed)
device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
base_dir = '/home/choah76/workspace2/Team_repa_inr_neurips_2025'


config = defaultdict()
# 'train', 'eval'
config['mode'] = 'eval'

# Logging
config['exp_name'] = 'strainer_encoder_train'
config['log_dir'] = os.path.join(base_dir, 'logs',
                                 config['exp_name'], 
                                 datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
config['data_dir'] = "/local_dataset/DIV2K" if config['mode'] == 'train' else "/local_dataset/Urban_100"
os.makedirs(config['log_dir'], exist_ok=True)

# Fitting params
config['epochs']=3   # Train: 5000, Eval: 2000
#config['epochs_train_strainer'] = 501
config['learning_rate'] = 1e-4
config['plot_every'] = 100
config['image_size'] = (256, 256)
config['pos_encode'] = False 
config['zero_mean'] = True

# INR params
config['num_layers'] = 6
config['hidden_features'] = 256
config['in_channels'] = 2
config['out_channels'] = 3
config['shared_encoder_layers'] = 5
config['num_decoders'] = 10
config['nonlin'] = 'siren'
config['weight_path'] = os.path.join(base_dir, 'logs', 
                                     'strainer_encoder_train', '20250328_212436', 
                                     'ret_strainer_10decoder_test_seed1234.pt')


if __name__ == '__main__':
    print(f"Logging to {config['log_dir']}")
    print(f"Using device {device}, data from {config['data_dir']}")
    
    with open(f"{config['log_dir']}/config_{seed}.txt", 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    
    # STRAINER Training (DIV2K)
    if config['mode'] == 'train':
        coords = get_coords(*config['image_size'], device=device)

        # STRAINER 10 decoder 데이터셋
        im_tensor_train = get_data(config, take=config['num_decoders'], sampling='random', device=device, seed=seed)
        data_dict_train = {'image_size':config['image_size'], 'gt':[x.reshape(1, -1, 3) for x in im_tensor_train]}
        
        # STRAINER 10 decoder 학습
        strainer_train = STRAINER(in_features=config['in_channels'],
                                    hidden_features=config['hidden_features'], hidden_layers=config['num_layers'],
                                    shared_encoder_layers = config['shared_encoder_layers'],
                                    num_decoders=10,
                                    out_features=config['out_channels']).to(device)
        optim_siren_strainer10decoder_train = torch.optim.Adam(lr=config['learning_rate'], params=strainer_train.parameters())

        print('\nTraining STRAINER 10 decoder\n')
        config_train = deepcopy(config)
        ret_strainer10decoder_train = fit_inr(coords=coords, data=data_dict_train,
                                                                model=strainer_train,
                                                                optim=optim_siren_strainer10decoder_train,
                                                                config=config_train, mlogger=None,name="strainer_encoder_only_10decoder")

        torch.save(ret_strainer10decoder_train,f"{config['log_dir']}/ret_strainer_10decoder_test_seed{seed}.pt")
        
    # STRAINER Evaluation (Urban100)
    elif config['mode'] == 'eval':
        coords = get_coords(*config['image_size'], device=device)
        ret_strainer_test = {}

        print('\nTesting STRAINER 10 decoder\n')

        for idx in range(len(os.listdir(config['data_dir']))):
            im_tensor = get_data(config, take=1, sampling='custom', sampling_list=[idx], device=device)
            data_dict_test = {'image_size':config['image_size'], 'gt':[x.reshape(1, -1, 3) for x in im_tensor]}  # for 10 decoder STRAINER

            strainer_test = STRAINER(in_features=config['in_channels'],
                                    hidden_features=config['hidden_features'], hidden_layers=config['num_layers'],
                                    shared_encoder_layers = config['shared_encoder_layers'],
                                    num_decoders=config['num_decoders'],
                                    out_features=config['out_channels']).to(device)

            if config['weight_path'] is not None:
                strainer_test.load_weights_from_file(config['weight_path'])
            optim_siren_strainer_test = torch.optim.Adam(lr=config['learning_rate'], params=strainer_test.parameters())

            ret_strainer_test[str(idx+1).zfill(2)] = fit_inr(coords=coords, data=data_dict_test,
                                                                    model=strainer_test,
                                                                    optim=optim_siren_strainer_test,
                                                                    config=config, mlogger=None,name=f"strainer_test_{idx}img")
            
        
        torch.save(ret_strainer_test,f"{config['log_dir']}/ret_strainer_10decoder_test_seed{seed}.pt")
        
    else:
        if os.path.exists(config['log_dir']):
            try:
                shutil.rmtree(config['log_dir'])  # 디렉토리와 내용 모두 삭제
                print(f"Removed temporary directory: {config['log_dir']}")
            except Exception as e:
                print(f"Failed to remove temporary directory: {e}")
        raise ValueError("You must choose either 'train' or 'eval' mode")