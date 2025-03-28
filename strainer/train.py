import torch
from tqdm import tqdm


def fit_inr(coords, data, model, optim, config={}, mlogger=None, name=None):
    assert name is not None, "`name` must be provided as metric logger needs it"
    gt_tensor = data['gt']

    #best_loss = np.inf
    #best_epoch = 0

    tbar = tqdm(range(config['epochs']))
    psnr_vals = []
    #out_mins = []
    #out_maxs = []
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

    #best_loss = np.inf
    #best_epoch = 0

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