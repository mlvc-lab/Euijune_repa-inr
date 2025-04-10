import torch
import torch.nn as nn

from utils import mean_flat


class REPAIRLoss(nn.Module):
    def __init__(self, proj_coeff=0.5):
        super().__init__()
        self.proj_coeff = proj_coeff

    def __call__(self, pred, gt, zs_tilde, zs):
        '''
        zs_tilde: INR의 m번째 encoder layer hidden representation
        zs: Visual encoder의 
        '''
        recon_loss = mean_flat((pred - gt) ** 2).sum()

        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]    # (10, 1, 256, j768)
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            # i = visual_encoder_idx
            # REPA github에서는 Dinov2말고도 다른 encoder들도 같이 함. 
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                # j = patch_idx
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)
        
        alignment_loss = (1 - proj_loss) * self.proj_coeff
        return recon_loss, alignment_loss
    
class MSELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, gt, zs_tilde=None, zs=None):
        # In MSELoss, zs_tilde and zs are not used.
        recon_loss = mean_flat((pred - gt) ** 2).sum()
        alignment_loss = None   # MSE Loss에서는 alignment 계산이 불필요요
        return recon_loss, alignment_loss