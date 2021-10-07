from torch import nn
import torch
import math


class PositionalEncodingLearned(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncodingLearned, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.register_parameter('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x)
    
#taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncodingVaswani(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncodingVaswani, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
#Adaptive two-dimensional encoding
#taken from https://github.com/Media-Smart/vedastr/blob/master/vedastr/models/bodies/sequences/transformer/position_encoder/adaptive_2d_encoder.py
def generate_encoder(in_channels, max_len):
    pos = torch.arange(max_len).float().unsqueeze(1)

    i = torch.arange(in_channels).float().unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / in_channels)

    position_encoder = pos * angle_rates
    position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
    position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

    return position_encoder

class PositionalEncoding1D(nn.Module):
    def __init__(self, in_channels, max_len=250, dropout=0.1):
        super(PositionalEncoding1D, self).__init__()

        position_encoder = generate_encoder(in_channels, max_len)
        position_encoder = position_encoder.unsqueeze(0)
        self.register_buffer('position_encoder', position_encoder)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = x + self.position_encoder[:, :x.size(1), :]
        out = self.dropout(out)

        return out

class Adaptive2DPositionEncoder(nn.Module):
    def __init__(self, in_channels, max_h=250, max_w=250, dropout=0.1):
        super(Adaptive2DPositionEncoder, self).__init__()

        h_position_encoder = generate_encoder(in_channels, max_h)
        h_position_encoder = h_position_encoder.transpose(0, 1).view(1, in_channels, max_h, 1)

        w_position_encoder = generate_encoder(in_channels, max_w)
        w_position_encoder = w_position_encoder.transpose(0, 1).view(1, in_channels, 1, max_w)

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        self.h_scale = self.scale_factor_generate(in_channels)
        self.w_scale = self.scale_factor_generate(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def scale_factor_generate(self, in_channels):
        scale_factor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        return scale_factor

    def forward(self, x):
        b, c, h, w = x.size()

        avg_pool = self.pool(x)

        h_pos_encoding = self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        out = x + h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out
    
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    