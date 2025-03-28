import torch
import datetime
import os

from copy import deepcopy
from model import STRAINER
from collections import defaultdict
from train import fit_inr, shared_encoder_training
from utils import get_train_data, get_test_data, get_coords


seed = 1234
torch.manual_seed(1234)
device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")



IMG_SIZE = (256,256)
POS_ENCODE = False

config = defaultdict()
config['epochs']=2000
#config['epochs_train_strainer'] = 501
config['learning_rate'] = 1e-4
config['plot_every'] = 100
config['image_size'] = IMG_SIZE

# INR params
config['num_layers'] = 6
config['hidden_features'] = 256
config['in_channels'] = 2
config['out_channels'] = 3
config['shared_encoder_layers'] = 5
config['num_decoders'] = 1
config['nonlin'] = 'siren'


# Logging
config['exp_name'] = 'strainer_reproduce'
config['log_dir'] = os.path.join('/home/choah76/workspace2/Team_repa_inr_neurips_2025/logs',
                                 config['exp_name'], 
                                 datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
config['train_dir'] = "/local_dataset/DIV2K"
config['test_dir'] = "/local_dataset/Urban_100"
os.makedirs(config['log_dir'], exist_ok=True)


if __name__ == '__main__':
    coords = get_coords(*IMG_SIZE, device=device)
    #print(coords.min(), coords.max())

    # STRAINER 10 decoder 데이터셋
    im_tensor_train = get_train_data(config['log_dir'], config['train_dir'], take=10, device=device, seed=seed)
    data_dict_train_strainer = {'image_size':IMG_SIZE, 'gt':[x.reshape(1, -1, 3) for x in im_tensor_train]}
    
    # STRAINER 10 decoder 학습
    inr_strainer_10decoders_train = STRAINER(in_features=config['in_channels'],
                                hidden_features=config['hidden_features'], hidden_layers=config['num_layers'],
                                shared_encoder_layers = config['shared_encoder_layers'],
                                num_decoders=10,
                                out_features=config['out_channels']).to(device)
    optim_siren_strainer10decoder_train = torch.optim.Adam(lr=config['learning_rate'], params=inr_strainer_10decoders_train.parameters())

    print('\nTraining STRAINER 10 decoder\n')
    config_train = deepcopy(config)
    config_train['epochs'] = 200
    ret_strainer10decoder_train = shared_encoder_training(coords=coords, data=data_dict_train_strainer,
                                                            model=inr_strainer_10decoders_train,
                                                            optim=optim_siren_strainer10decoder_train,
                                                            config=config_train, mlogger=None,name="strainer_encoder_only_10decoder")
    
    # STRAINER 10 decoder 테스트 (Urban-100)
    ret_strainer_10decoder_test = {}

    print('\nTesting STRAINER 10 decoder\n')

    for idx in range(len(os.listdir(config['test_dir']))):
        im_tensors = get_test_data(config['test_dir'], idx=idx, device=device)
        data_dict_test_strainer_10 = {'image_size':IMG_SIZE, 'gt':im_tensors[0].reshape(1, -1, 3)}  # for 10 decoder STRAINER

        inr_strainer_test = STRAINER(in_features=config['in_channels'],
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
        
    
    torch.save(ret_strainer_10decoder_test,f"{config['log_dir']}/ret_strainer_10decoder_test_seed{seed}.pt")
    print(f"{config['log_dir']}/ret_strainer_10decoder_test_seed{seed}.pt")