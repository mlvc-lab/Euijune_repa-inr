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


base_dir = '/home/choah76/workspace2/Team_repa_inr_neurips_2025'

# Other hyperparameters
config = defaultdict()
config['seed'] = 1234
set_seeds(config['seed'])
config['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
config['mode'] = 'train' # 'train' or 'eval' 

# Loss
config['loss_fn'] = 'repair'                           # 'mse' or 'repair'
config['encoder_depth'] = 2
config['enc_type'] = 'dinov2-vit-b'                 # 'dinov2-vit-b' or ??
config['proj_coeff'] = 0.5
config['learning_rate'] = 1e-4
config['epochs']=10                               # Train: 5000, Eval: 2000

# Logging
config['exp_name'] = 'strainer_encoder_train'
config['log_dir'] = os.path.join(base_dir, 'logs',
                                 config['exp_name'], 
                                 datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(config['log_dir'], exist_ok=True)
config['plot_every'] = 100

# Data
config['image_size'] = (256, 256)
config['data_dir'] = "/local_dataset/DIV2K" if config['mode'] == 'train' else "/local_dataset/Urban_100"
config['sampling'] = 'random' if config['mode'] == 'train' else 'custom'

# INR params
config['num_layers'] = 6
config['hidden_features'] = 256
config['in_channels'] = 2
config['out_channels'] = 3
config['shared_encoder_layers'] = 5
config['num_decoders'] = 10 if config['mode'] == 'train' else 1
config['nonlin'] = 'siren'
config['weight_path'] = None
# config['weight_path'] = os.path.join(base_dir, 'logs', 
#                                      'strainer_encoder_train', '20250401_190721', 
#                                      'ret_strainer_10decoder_test_seed1234.pt')
config['pos_encode'] = False 
config['zero_mean'] = True


if __name__ == '__main__':
    print(f"Logging to {config['log_dir']}")
    print(f"Using device {config['device']}, data from {config['data_dir']}")
    
    with open(f"{config['log_dir']}/config_{config['seed']}.txt", 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
            
    print(f"\n{config['mode'].upper()} STRAINER {config['num_decoders']} decoder\n")
    
    # STRAINER Training (DIV2K)
    if config['mode'] == 'train':
        result = fit_inr(config=config, name="strainer_encoder_only_10decoder")
        torch.save(result, f"{config['log_dir']}/ret_strainer_10decoder_test_seed{config['seed']}.pt")
        
    # STRAINER Evaluation (Urban100)
    elif config['mode'] == 'eval':
        result = {}
        for idx in range(len(os.listdir(config['data_dir']))):
            result[str(idx+1).zfill(2)] = fit_inr(config=config, name=f"strainer_test_{idx}img", idx=[idx])
        torch.save(result,f"{config['log_dir']}/ret_strainer_10decoder_test_seed{config['seed']}.pt")
        
    else:
        if os.path.exists(config['log_dir']):
            try:
                shutil.rmtree(config['log_dir'])  # 디렉토리와 내용 모두 삭제
                print(f"Removed temporary directory: {config['log_dir']}")
            except Exception as e:
                print(f"Failed to remove temporary directory: {e}")
        raise ValueError("You must choose either 'train' or 'eval' mode")