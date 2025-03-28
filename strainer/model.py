import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


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
                 hidden_omega_0=30., pos_encode=False, no_init=False):
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

    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
        return self.net(coords)

class STRAINER(nn.Module):
    def __init__(self, in_features, hidden_features,
                 hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.,
                 pos_encode=False,
                 shared_encoder_layers=None, num_decoders=None, no_init=False):

        super().__init__()
        assert shared_encoder_layers is not None, "Please mention shared_encoder_layers. Use 0 if none are shared"
        assert hidden_layers > shared_encoder_layers, "Total hidden layers must be greater than number of layers in shared encoder"
        self.shared_encoder_layers = shared_encoder_layers
        self.num_decoders = num_decoders

        self.encoderINR = SIREN(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=self.shared_encoder_layers - 1, # input is a layer
            out_features=hidden_features,
            outermost_linear=False,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            pos_encode=pos_encode,
            no_init=no_init
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
        encoded_features = self.encoderINR(coords)
        outputs = []
        for _idx, _decoder in enumerate(self.decoderINRs):
            output = _decoder(encoded_features)
            outputs.append(output)

        return outputs

    def load_encoder_weights_from(self, fellow_model):
        if fellow_model is not None:
            self.encoderINR.load_state_dict(deepcopy(fellow_model.encoderINR.state_dict()))
        else:
            raise ValueError("Fellow model is None")

    def load_weights_from_file(self, file, prefix="encoderINR"):
        model_state_dict = self.state_dict()
        weights = torch.load(file)['state_dict']
        encoder_state_dict = {k: v for k, v in weights.items() if k.startswith(prefix)}
        for name, param in encoder_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name] = deepcopy(param)