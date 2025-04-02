import torch
import torch.nn as nn

from utils import mean_flat


class REPAIRLoss(nn.Module):
    def __init__(self, proj_coeff=0.5):
        super().__init__()
        self.proj_coeff = proj_coeff

    def __call__(self, pred, gt, zs_tilde, zs):
        recon_loss = mean_flat((pred - gt) ** 2)

        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            # for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
            #     z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
            #     z_j = torch.nn.functional.normalize(z_j, dim=-1) 
            #     proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            z_tilde = torch.nn.functional.normalize(z_tilde[0], dim=-1) 
            z = torch.nn.functional.normalize(z, dim=-1) 
            proj_loss += mean_flat(-(z * z_tilde).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)
        
        return recon_loss + proj_loss * self.proj_coeff
    
class MSELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, gt, zs_tilde=None, zs=None):
        # In MSELoss, zs_tilde and zs are not used.
        loss_val = mean_flat((pred - gt) ** 2).sum()
        return loss_val