import torch
from tqdm import tqdm
from utils import load_encoders, preprocess_raw_image, get_coords, get_data
from loss import REPAIRLoss, MSELoss
from model import STRAINER


def fit_inr(config={}, name=None, idx=None):    
    im_tensors = get_data(config, 
                          take=config['num_decoders'], 
                          sampling=config['sampling'], 
                          device=config['device'], 
                          seed=config['seed'], 
                          sampling_list=idx)
    data = {'image_size':config['image_size'], 'gt':[x.reshape(1, -1, 3) for x in im_tensors]}
    gt_tensors = data['gt']
    
    if config['enc_type'] != None:
        encoders, encoder_types, architectures = load_encoders(
            config['enc_type'], config['device'], resolution=config['image_size'][0]
            )
    else:
        raise NotImplementedError()
    
    assert name is not None, "`name` must be provided as metric logger needs it"
    
    # get representation from encoders
    if config['loss_fn'] == 'repair':
        loss_fn = REPAIRLoss(proj_coeff=config['proj_coeff'])
        z_dims = [encoder.embed_dim for encoder in encoders] if config['enc_type'] != 'None' else [0]
        zs = []
        for gt_tensor in gt_tensors:
            z = None
            with torch.no_grad():
                for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                    gt_tensor_ = preprocess_raw_image(gt_tensor, encoder_type, img_size=config['image_size'][0])
                    z = encoder.forward_features(gt_tensor_)    # (1, 256, 768)
                    if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                    zs.append(z)
                    
    elif config['loss_fn'] == 'mse':
        loss_fn = MSELoss()
        zs = None
        z_dims=None
    
    coords = get_coords(*config['image_size'], device=config['device'])
    model = STRAINER(in_features=config['in_channels'],
                    hidden_features=config['hidden_features'], hidden_layers=config['num_layers'],
                    shared_encoder_layers = config['shared_encoder_layers'],
                    num_decoders=config['num_decoders'],
                    out_features=config['out_channels'],
                    encoder_depth=config['encoder_depth'],
                    z_dims=z_dims).to(config['device'])
    model = model.train()
    if config['weight_path'] is not None:
                model.load_weights_from_file(config['weight_path'])
                
    optim = torch.optim.Adam(lr=config['learning_rate'], params=model.parameters())
    
    
    tbar = tqdm(range(config['epochs']))
    psnr_vals = []
    for epoch in tbar:
        outputs, zs_tilde = model(coords) # (10, 1, HW, 3), (10, 1, HW, 256)
        stacked_outputs = torch.stack(outputs, dim=0)
        stacked_gt = torch.stack(gt_tensors, dim=0)
        loss = loss_fn(stacked_outputs, stacked_gt, zs_tilde, zs)
            
        ## optimization
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        
        # PSNR 계산을 위해 추가
        stacked_outs = stacked_outputs/2+0.5
        stacked_gt = stacked_gt/2+0.5

        psnr = -10*torch.log10(((stacked_outs - stacked_gt)**2).mean(dim=[1,2,3]).sum()) #  config['image_size'][0],  config['image_size'][1])
        psnr_vals.append(float(psnr))

        tbar.set_description(f"Iter {epoch}/{config['epochs']} Loss = {loss.item():6f} PSNR = {psnr:.4f}")
        tbar.refresh()
        
    result = {
        "psnr" : psnr_vals,
        "state_dict" : model.state_dict()
    }
    
    return result
    