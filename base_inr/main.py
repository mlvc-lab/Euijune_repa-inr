import os
import torch
import matplotlib
import configargparse

import warnings

from utils import setup_seed, get_train_data, get_model, find_image_file
from train_image import train

setup_seed(3407)
device = torch.device('cuda:0')
matplotlib.use('agg')
warnings.filterwarnings("ignore") 



def get_opts():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs', help='logdir')
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    
    # dataset
    parser.add_argument('--dataset_dir', type=str, default= '../data/setA', help='dataset')
    parser.add_argument('--img_id', type=int, default=0, help='id of image')
    parser.add_argument('--not_zero_mean', action='store_true') 
    parser.add_argument('--sidelen', type=int, default=256, help='Image resolution to resize (sidelen x sidelen x channel)')
    
    # training options
    parser.add_argument('--num_epochs', type=int, default=3000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--steps_til_summary', type=int, default=1, help='steps_til_summary')
    
    # network 
    parser.add_argument('--model_type', type=str, default='siren', required=['siren', 'finer', 'wire', 'gauss', 'pemlp'])
    parser.add_argument('--hidden_layers', type=int, default=3, help='hidden_layers, total_layer = first layer (1) + hidden_layers + Final_layer (1)') 
    parser.add_argument('--hidden_features', type=int, default=256, help='hidden_features')
    # parser.add_argument('--load_path', type=str, default=None, help='load model states from path')
    # parser.add_argument('--pretrain_epochs', type=int, default=500)
    parser.add_argument('--out_feature', type=int, default=3, help='output channel dimension')
    
    # model hyp param
    parser.add_argument('--first_omega', type=float, default=30, help='(siren, wire, finer)')    
    parser.add_argument('--hidden_omega', type=float, default=30, help='(siren, wire, finer)')    
    # parser.add_argument('--scale', type=float, default=30, help='simga (wire, guass)')    
    # parser.add_argument('--N_freqs', type=int, default=10, help='(PEMLP)')    

    # finer
    # parser.add_argument('--first_bias_scale', type=float, default=None, help='bias_scale of the first layer')    
    # parser.add_argument('--scale_req_grad', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opts = get_opts()
    
    ## logdir 
    logdir = os.path.join(opts.logdir, opts.exp_name)
    os.makedirs(logdir, exist_ok=True)
    
    config_path = f'{logdir}/config.txt'
    if not os.path.exists(config_path) or os.stat(config_path).st_size == 0:
        with open(config_path, 'a') as ff:
            for k, v in vars(opts).items():
                ff.write(f"{k}: {v}\n")
    
    # image path
    coords, gt, size = get_train_data(find_image_file(opts.dataset_dir, opts.img_id), not opts.not_zero_mean, opts.sidelen, opts.out_feature)
    coords = coords.to(device)      # (H, W, C) [-1, 1]
    gt = gt.to(device)              # (H, W, C) [-1, 1]
    
    # model 
    model, is_scheduler = get_model(opts)
    model = model.to(device)

    # load state dict
    if opts.load_path is not None:
        try:
            # 새로운 버전: pt파일에는 Logger Class가 저장되어 있으며, ckpt.state_dict와 같은 방식으로 접근 가능
            ckpt = torch.load(opts.load_path)
            model.load_state_dict(ckpt.state_dicts[f'{opts.pretrain_epochs}step'])
        except AttributeError:
            # 이전 버전: pt파일은 dictionary 형태로 저장되어 있음
            model.load_state_dict(ckpt['model_state'])
        except:
            raise(f"Failed loading state from {opts.load_path}")
    
    # train
    log = train(opts, model, is_scheduler, coords, gt, size, not opts.not_zero_mean, loss_fn='mse')
    
    # save 
    torch.save(log, os.path.join(logdir, 'outputs_%02d.pt'%((opts.img_id))))
    
    # verbose
    print('Train PSNR: %.5f,    SSIM: %.5f,    LPIPS: %.7f'%(log.psnr[-1], log.ssim[-1], log.lpips[-1]))