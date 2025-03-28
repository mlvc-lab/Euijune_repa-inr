import torch
import time
import warnings

from tqdm import trange
from utils import mse_fn, psnr_fn, ssim_fn, lpips_fn, waveloss_fn, Logger


warnings.filterwarnings("ignore") 


def train(opts, model, is_scheduler, coords, gt, size, zero_mean=True, loss_fn='mse'):
    if loss_fn == 'mse':
        loss_fn = mse_fn
    else:
        raise NotImplementedError(f"Loss function {loss_fn} is not implemented")

    optimizer = torch.optim.Adam(lr=opts.lr, params=model.parameters())
    
    if is_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opts.num_epochs, 1))

    
    logger = Logger()
    for epoch in trange(1, opts.num_epochs + 1):      
        time_start = time.time()

        pred = model(coords)        # [-1, 1], [H, W, C]
        loss = loss_fn(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_scheduler:
            scheduler.step()

        torch.cuda.synchronize()
        logger.total_time += time.time() - time_start

        # Calculate metrics & Save intermediate results
        if not epoch % opts.steps_til_summary:
            with torch.no_grad():
                if zero_mean:
                    # [-1, 1] -> [0, 1] 작업을 수행하고 metric 계산.
                    psnr_val = psnr_fn(pred.reshape(size)/2+0.5, gt.reshape(size)/2+0.5).item()
                    ssim_val = ssim_fn(pred.reshape(size)/2+0.5, gt.reshape(size)/2+0.5).item()
                    lpips_val = lpips_fn(pred.reshape(size)/2+0.5, gt.reshape(size)/2+0.5).item()
                else:
                    psnr_val = psnr_fn(pred.reshape(size), gt.reshape(size)).item()
                    ssim_val = ssim_fn(pred.reshape(size), gt.reshape(size)).item()
                    lpips_val = lpips_fn(pred.reshape(size), gt.reshape(size)).item()

                # Calculate Wavelet coff difference
                #waveloss = waveloss_fn(pred.reshape(size).detach().cpu(), gt.reshape(size).detach().cpu())
            logger._append(waveloss=waveloss, psnr=psnr_val, ssim=ssim_val, lpips=lpips_val)

        # Save intermediate pred image
        if (epoch-1) in [1, 5, 10, 20, 50, 100, 2000]:
            if zero_mean:
                out = pred.reshape(size)/2+0.5
            else:
                out = pred.reshape(size)
            logger._adddict(pred_imgs={
                f'{epoch-1}step': out
            })

        # Save intermediate model state
        if (epoch-1) in [500, 1000, 1500, 2000]:
            logger._adddict(state_dicts={
                f'{epoch-1}step': model.state_dict()
            })             
        

    # 마지막 epoch에 대한 결과 저장
    with torch.no_grad():
        if zero_mean:
            # [-1, 1] -> [0, 1]
            out = model(coords).reshape(size)/2+0.5
        else:
            out = model(coords).reshape(size)
        logger._adddict(
            pred_imgs={
                f'{opts.num_epochs}step': out}, 
            state_dicts={
                f'{opts.num_epochs}step': model.state_dict()}
            )


    return logger