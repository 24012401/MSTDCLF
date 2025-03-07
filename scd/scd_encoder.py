import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange
import math

from model.ctrgcn import Model

class Encoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer) -> None:
        super().__init__()
        self.d_model = hidden_size

        hidden_size = 64
        self.gcn_t = Model(hidden_size)
        self.gcn_s = Model(hidden_size)

        self.channel_t = nn.Sequential(
            nn.Linear(50*hidden_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
        )

        self.channel_s = nn.Sequential(
            nn.Linear(64 * hidden_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
        )

        self.t_encoder = nn.LSTM(input_size=self.d_model,hidden_size=self.d_model//2,num_layers=num_layer,batch_first=True,bidirectional=True)
        self.s_encoder = nn.LSTM(input_size=self.d_model,hidden_size=self.d_model//2,num_layers=num_layer,batch_first=True,bidirectional=True)


    def forward(self, x):

        self.t_encoder.flatten_parameters()
        self.s_encoder.flatten_parameters()
        
        vt = self.gcn_t(x)
        vt = rearrange(vt, '(B M) C T V -> B T (M V C)', M=2)
        vt = self.channel_t(vt)

        vs = self.gcn_s(x)
        vs = rearrange(vs, '(B M) C T V -> B (M V) (T C)', M=2)
        vs = self.channel_s(vs)

        vt, _ = self.t_encoder(vt) # B T C
        vs, _ = self.s_encoder(vs)

        vt = vt.amax(dim=1)
        vs = vs.amax(dim=1)

        return vt, vs


class PretrainingEncoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer,
                 num_class=60,
                 ):
        super(PretrainingEncoder, self).__init__()

        self.d_model = hidden_size

        self.encoder = Encoder(
            hidden_size, num_head, num_layer,
        )

        self.t_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.s_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.i_proj = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

    def forward(self, x):

        vt, vs = self.encoder(x)

        zt = self.t_proj(vt)
        zs = self.s_proj(vs)

        vi = torch.cat([vt, vs], dim=1)

        zi = self.i_proj(vi)

        return zt, zs, zi


class DownstreamEncoder(nn.Module):
    def __init__(self, 
                 hidden_size, num_head, num_layer,
                 num_class=60,
                 ):
        super(DownstreamEncoder, self).__init__()

        self.d_model = hidden_size

        self.encoder = Encoder(
            hidden_size, num_head, num_layer,
        )

        self.fc = nn.Linear(2 * self.d_model, num_class)

    def forward(self, x, knn_eval=False):

        vt, vs = self.encoder(x)

        vi = torch.cat([vt, vs], dim=1)

        if knn_eval:
            return vi
        else:
            return self.fc(vi)