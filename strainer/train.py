import torch
from tqdm import tqdm


def fit_inr(coords, data, model, optim, config={}, mlogger=None, name=None):
    assert name is not None, "`name` must be provided as metric logger needs it"
    gt_tensor = data['gt']

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

        # PSNR 계산을 위해 추가
        stacked_outs = stacked_outputs/2+0.5
        stacked_gt = stacked_gt/2+0.5

        psnr = -10*torch.log10(((stacked_outs - stacked_gt)**2).mean(dim=[1,2,3]).sum()) #  config['image_size'][0],  config['image_size'][1])
        psnr_vals.append(float(psnr))

        tbar.set_description(f"Iter {epoch}/{config['epochs']} Loss = {loss.item():6f} PSNR = {psnr:.4f}")
        tbar.refresh()

    return {
        "psnr" : psnr_vals,
        "state_dict" : model.state_dict()
    }