import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class Projector(nn.Module):
    '''
    input: tensor (B, H*W, INR_hidden_dim)
    output: tensor (B, p1*p2, z_dim) where p1*p2 = kernel_size (16x16)
    '''
    def __init__(self, hidden_size, projector_dim, z_dim, kernel_size=16, stride=16):
        super(Projector, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.linear1 = nn.Linear(hidden_size, projector_dim)
        self.silu1 = nn.SiLU()
        self.linear2 = nn.Linear(projector_dim, projector_dim)
        self.silu2 = nn.SiLU()
        self.linear3 = nn.Linear(projector_dim, z_dim)

    def forward(self, x):
        # x.shape = (B, H*W, INR_hidden_dim)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], 256, 256)
        # x.shape = (B, INR_hidden_dim, H, W)
        x = self.avgpool(x)
        # x.shape = (B, INR_hidden_dim, p1, p2), p1, p2 = kernel_size (16x16)
        x = x.view(1, 256, -1)
        x = x.permute(0, 2, 1)
        # x.shape = (B, p1*p2, INR_hidden_dim)
        x = self.linear1(x)
        x = self.silu1(x)
        x = self.linear2(x)
        x = self.silu2(x)
        x = self.linear3(x)
        # x.shape = (B, p1*p2, z_dim)
        return x

def build_projector(hidden_size, projector_dim, z_dim, kernel_size=16, stride=16):
    return Projector(hidden_size, projector_dim, z_dim, kernel_size, stride)

class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.

        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.

        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, outermost_linear=True, first_omega_0=30, 
                 hidden_omega_0=30., pos_encode=False, no_init=False,
                 z_dims=[768], projector_dim=2048, encoder_depth=-1):
        '''
        z_dims: list of dimensions of the latent space for each encoder(ex. DINOv2)
        projector_dim: dimension of the projector network
        encoder_depth: index of the encoder layer to get the activations (1부터 시작)
        '''
        
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = SineLayer

        self.net = []
        if hidden_layers != 0:
            # append first sine layer
            self.net.append(self.nonlin(in_features, hidden_features,
                            is_first=True, omega_0=first_omega_0, init_weights=(not no_init)))
            
        hidden_layers = hidden_layers -1 if (hidden_layers > 0 and outermost_linear is True) else hidden_layers
        
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features,
                            is_first=False, omega_0=hidden_omega_0, init_weights=(not no_init)))

        if outermost_linear or (hidden_layers == 0):
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)

            if not no_init:
                with torch.no_grad():
                    const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                    final_linear.weight.uniform_(-const, const)

            self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)
        self.encoder_depth = encoder_depth
        
        # For REPAIRLoss
        if self.encoder_depth > -1:
            self.projectors = nn.ModuleList([
                build_projector(hidden_features, 
                                projector_dim, 
                                z_dim, 
                                kernel_size=16, 
                                stride=16) for z_dim in z_dims
                ])

    def forward(self, coords):
        zs = None
        x = self.positional_encoding(coords) if self.pos_encode else coords
        
        for i , layer in enumerate(self.net):
            x = layer(x)
            if self.encoder_depth is not None and i == self.encoder_depth:
                zs = [projector(x) for projector in self.projectors]
        
        return x, zs

class STRAINER(nn.Module):
    def __init__(self, in_features, hidden_features,
                 hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.,
                 pos_encode=False,
                 shared_encoder_layers=None, num_decoders=None, no_init=False,
                 z_dims=[768], projector_dim=2048, encoder_depth=None):
        '''
        z_dims: list of dimensions of the latent space for each encoder(ex. DINOv2)
        projector_dim: dimension of the projector network
        encoder_depth: index of the encoder layer to get the activations (1부터 시작)
        '''

        super().__init__()
        assert shared_encoder_layers is not None, "Please mention shared_encoder_layers. Use 0 if none are shared"
        assert hidden_layers > shared_encoder_layers, "Total hidden layers must be greater than number of layers in shared encoder"
        if encoder_depth is not None:
            assert encoder_depth <= shared_encoder_layers, "Encoder depth must be equal or less than shared encoder layers"
        
        self.shared_encoder_layers = shared_encoder_layers
        self.num_decoders = num_decoders
        self.encoder_depth = encoder_depth

        self.encoderINR = SIREN(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=self.shared_encoder_layers - 1, # input is a layer
            out_features=hidden_features,
            outermost_linear=False,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            pos_encode=pos_encode,
            no_init=no_init,
            z_dims=z_dims,
            projector_dim=projector_dim,
            encoder_depth=self.encoder_depth
        )

        self.num_decoder_layers = hidden_layers - self.shared_encoder_layers
        assert self.num_decoder_layers >= 1 , "Num decoder layers must be more than 1"
        self.decoderINRs = nn.ModuleList([
                                            SIREN(
                                                in_features=hidden_features,
                                                hidden_features=hidden_features,
                                                hidden_layers=self.num_decoder_layers - 1,
                                                out_features=out_features,
                                                outermost_linear=outermost_linear,
                                                first_omega_0=first_omega_0,
                                                hidden_omega_0=hidden_omega_0,
                                                pos_encode=pos_encode,
                                                no_init=no_init
                                            ) for i in range(self.num_decoders)])
        

    def forward(self, coords):
        encoded_features, zs_tilde = self.encoderINR(coords)
        outputs = []
        for _idx, _decoder in enumerate(self.decoderINRs):
            output = _decoder(encoded_features)  # zs는 encoder에서만 계산됨. decoder에서는 항상 None
            outputs.append(output[0])
        return outputs, [zs_tilde] * 10

    def load_encoder_weights_from(self, fellow_model):
        if fellow_model is not None:
            self.encoderINR.load_state_dict(deepcopy(fellow_model.encoderINR.state_dict()))
        else:
            raise ValueError("Fellow model is None")

    def load_weights_from_file(self, file, prefix="encoderINR"):
        ckpt = torch.load(file)
        encoder_state_dict = {}
        for k, v in ckpt['state_dict_last'].items():
            # encoderINR로 시작하는 레이어만 선택, encoder에 같이 있는 projector는 eval에서 안쓰므로 제외
            if k.startswith(prefix) and k.find('projectors') == -1:  
                encoder_state_dict[k.replace(f'{prefix}.', '')] = v
            
        self.encoderINR.load_state_dict(encoder_state_dict)