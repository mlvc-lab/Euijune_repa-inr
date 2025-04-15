import torch
import datetime
import os
import shutil
import warnings
warnings.filterwarnings("ignore")
import configargparse

from collections import defaultdict
from train import fit_inr
from utils import set_seeds



BASE_DIR = '/home/choah76/workspace2/Team_repa_inr_neurips_2025'

def get_configs():
    config = configargparse.ArgumentParser()
    config.add_argument('--seed', type=int, default=1234, help='random seed')
    config.add_argument('--device', type=str, default='cuda:0', help='device to use for training')
    config.add_argument('--mode', type=str, default='train', help='train or eval')
    
    # Loss
    config.add_argument('--loss_fn', type=str, default='mse', help='loss function to use (mse or repair)')
    config.add_argument('--encoder_depth', type=int, default=None, help='depth of the encoder')
    config.add_argument('--enc_type', type=str, default=None, help='encoder type (dinov2-vit-b only)')
    config.add_argument('--proj_coef', type=float, default=0.0, help='projection coefficient for repair loss')
    
    # Training
    config.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    config.add_argument('--epochs', type=int, default=5000, help='number of epochs to train for')
    
    # Data
    config.add_argument('--img_size', type=tuple, default=(256, 256), help='image size')
    config.add_argument('--data_dir', type=str, default='/local_dataset/DIV2K', help='directory to dataset')
    config.add_argument('--sampling', type=str, default='random', help='sampling method (random or custom)')
    
    # Model
    config.add_argument('--num_layers', type=int, default=8, help='number of layers in the model')
    config.add_argument('--hidden_features', type=int, default=256, help='number of hidden features')
    config.add_argument('--in_channels', type=int, default=2, help='number of input channels')
    config.add_argument('--out_channels', type=int, default=3, help='number of output channels')
    config.add_argument('--shared_encoder_layers', type=int, default=7, help='number of shared encoder layers')
    config.add_argument('--num_decoders', type=int, default=10, help='number of decoders')
    config.add_argument('--nonlin', type=str, default='siren', help='nonlinearity to use (siren or other)')
    config.add_argument('--weight_path', type=str, default=None, required=False, help='path to pretrained weights')
    config.add_argument('--pos_encode', action='store_true', help='use positional encoding')
    config.add_argument('--zero_mean', action='store_true', help='use zero mean normalization')
    
    # Logging
    config.add_argument('--exp_name', type=str, default=None, help='아래 __main__에서 설정')
    config.add_argument('--log_dir', type=str, default='logs', help='directory to save logs')
    config.add_argument('--save_dir', type=str, default=None, help='아래 __main__에서 설정')
    config.add_argument('--plot_every', type=list, default=[1, 10, 20, 30, 50, 100, 200, 300, 500, 1000], help='plotting frequency')
    
    return config.parse_args()


if __name__ == '__main__':
    config = get_configs()
    set_seeds(config.seed)
    
    # Set experiment name
    config.exp_name =  os.path.join(
        'strainer',
        config.mode,
        config.data_dir.split('/')[-1],
        f"{config.num_decoders}img_{config.loss_fn}" \
            if config.mode == 'train' 
            else \
                f"from_{config.weight_path.split('/')[-2]}_depth_{config.weight_path.split('/')[-1].split('_')[4]}",
    )
    
    # Set logfile(.pt) name
    fname = f"{'%s'}_{'%s'}_enc_depth_{'%s'}_seed_{'%s'}.pt" % (
            config.shared_encoder_layers,
            config.num_layers - config.shared_encoder_layers,
            config.encoder_depth,
            config.seed
        )
        
    
    #if config.mode == 'eval':
    #    config.exp_name += '_from_repair_loss'
        
    save_dir = os.path.join(BASE_DIR, config.log_dir, config.exp_name)
    config.save_dir = save_dir
    os.makedirs(config.save_dir, exist_ok=True)
    
    print(f"Logging to {config.save_dir}")
    print(f"Using device {config.device}, data from {config.data_dir}")
    
    with open(f"{config.save_dir}/config_{fname.split('.')[0]}.txt", 'w') as f:
        for k, v in vars(config).items():
            f.write(f"{k}: {v}\n")
            
    print(f"\n{config.mode.upper()} STRAINER {config.num_decoders} decoder\n")
    
    # STRAINER Training (DIV2K)
    if config.mode == 'train':
        result = fit_inr(config=config)
        torch.save(result, os.path.join(config.save_dir, fname))
        
    # STRAINER Evaluation (Urban100, Chest_CT, CelebA_HQ)
    elif config.mode == 'eval':
        result = {}
        #for idx in range(len(os.listdir(config['data_dir']))):
        for idx in range(20):
            print(f"Evaluating {idx+1}th image...")
            result[str(idx+1).zfill(2)] = fit_inr(config=config, idx=[idx])
        torch.save(result, os.path.join(config.save_dir, fname))