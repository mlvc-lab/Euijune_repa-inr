import numpy as np
#import imageio.v2 as imageio
import imageio
import torch
import random
import lpips
import glob
import pywt
import os
import warnings

from skimage.metrics import structural_similarity
from models import Finer, Siren, Gauss, PEMLP, Wire
from torchvision.transforms import Resize, ToTensor, CenterCrop, Normalize, Compose
from skimage.color import rgb2gray


warnings.filterwarnings("ignore") 


class Logger:
    def __init__(self):
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.waveloss = []
        self.pred_imgs = {}
        self.state_dicts = {}
        self.total_time = 0.0

    # list에 값을 추가하는 함수
    def _append(self, **kwargs):
        for key, value in kwargs.items():
            getattr(self, key).append(value)

    # dictionary에 값을 추가하는 함수
    def _adddict(self, **kwargs):
        for key, value in kwargs.items():
            getattr(self, key).update(value)    # value: dict

    def __str__(self):
        return str(self.__dict__.keys())


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mse_fn(pred, gt):
    # input shape must be [H, W, C]
    return ((pred - gt) ** 2).mean()

def psnr_fn(pred, gt):
    # input shape must be [H, W, C]
    return -10. * torch.log10(mse_fn(pred, gt))

def ssim_fn(pred, gt):
    '''
    input shape
        image: (W, H, C)
        video: (B, W*H, C)  and B is almost 1.
    '''
    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    if pred.shape[2] == 3:
        ssims = []
        diffs = []
        for i in range(3):
            try:
                # for image
                score, diff = structural_similarity(pred[:,:,i], gt[:,:,i], full=True, data_range=1.)
            except ValueError:
                # for video
                score, diff = structural_similarity(pred[0, :,i], gt[0,:,i], full=True, data_range=1.)
            ssims.append(score)
            diffs.append(diff)
        #return np.array(ssims).mean(), np.array(diffs).mean()
        return np.array(ssims).mean()
    else:
        # gray image
        score, _ = structural_similarity(np.squeeze(pred), np.squeeze(gt), full=True, data_range=1.)
        return score
        #return score, _
    
# input shape must be [C, H*W]
calcluate_LPIPS = lpips.LPIPS(net='alex', verbose=False).cuda()
def lpips_fn(pred, gt):
    # input shape must be [H, W, C]
    return calcluate_LPIPS(pred.permute(2,0,1), gt.permute(2,0,1)).squeeze().cpu().detach().numpy()

def waveloss_fn(pred, gt):
    if pred.shape[2] == 3:
        p_gray = rgb2gray(pred)
        trgt_gray = rgb2gray(gt)
    elif pred.shape[2] == 1:
        p_gray = pred
        trgt_gray = gt

    p_coeffs2 = pywt.dwt2(p_gray, 'bior1.3')
    trgt_coeffs2 = pywt.dwt2(trgt_gray, 'bior1.3')
    LL_p, (LH_p, HL_p, HH_p) = p_coeffs2
    LL_trgt, (LH_trgt, HL_trgt, HH_trgt) = trgt_coeffs2

    ret = []
    for coef_p, coef_trgt in zip([LL_p, LH_p, HL_p, HH_p], [LL_trgt, LH_trgt, HL_trgt, HH_trgt]):
        wave_loss = ((coef_p - coef_trgt)**2).mean()
        ret.append(wave_loss)

    return ret  # LL, LH, HL, HH



def get_train_data(img_path, zero_mean=True, sidelen=256, out_feature=3):
    
    pilmode='RGB' if out_feature == 3 else 'L'
    img = np.array(imageio.imread(img_path, pilmode=pilmode), dtype=np.float32) / 255.   # [H, W, C], [0, 1]
    if img.ndim == 2:  # 그레이스케일 이미지일 때
        img = np.expand_dims(img, axis=-1)  # C 차원을 추가하여 (1, H, W)로 만듦
        
    H, W, _ = img.shape

    aug_list = [
            ToTensor(),
            CenterCrop(min(H, W)),
            Resize((sidelen, sidelen)),
    ]
    if zero_mean:
            aug_list.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))

    transform = Compose(aug_list)
    img = transform(img).permute(1, 2, 0)

    H, W, C = img.shape

    gt = img.view(-1, C)
    coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)], indexing='ij'), dim=-1).view(-1, 2)
    return coords, gt, [H, W, C]


def get_model(opts):
    scheduler = False
    if opts.model_type == 'finer':
        model = Finer(in_features=2, out_features=opts.out_feature, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega, first_bias_scale=opts.first_bias_scale, scale_req_grad=opts.scale_req_grad)
        scheduler = True
    elif opts.model_type == 'siren':
        model = Siren(in_features=2, out_features=opts.out_feature, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega)
    elif opts.model_type == 'wire':
        model = Wire(in_features=2, out_features=opts.out_feature, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                     first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega, scale=opts.scale)
        scheduler = True
    elif opts.model_type == 'gauss':
        model = Gauss(in_features=2, out_features=opts.out_feature, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      scale=opts.scale)
    elif opts.model_type == 'pemlp':
        model = PEMLP(in_features=2, out_features=opts.out_feature, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      N_freqs=opts.N_freqs)
    return model, scheduler


def find_image_file(dataset_dir, img_id):
    # 검색할 파일 패턴을 정의
    image_patterns = [
        os.path.join(dataset_dir, f'{img_id:02d}.jpg'),
        os.path.join(dataset_dir, f'{img_id:03d}.jpg'),
        os.path.join(dataset_dir, f'{img_id:02d}.png'),
        os.path.join(dataset_dir, f'{img_id:03d}.png'),
    ]

    # 패턴에 맞는 파일 찾기
    image_files = [f for pattern in image_patterns for f in glob.glob(pattern)]
    
    if len(image_files) > 0:
        return image_files[0]  # 첫 번째로 찾은 파일 반환
    else:
        raise FileNotFoundError(f'Image not found in {dataset_dir} with id {img_id}')