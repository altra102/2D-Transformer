from torch import nn
from blocks import NormLayer, FeedForward1D, FeedForward2D
from attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_layers, dropout=.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([Transformer2DEncoderLayer(hidden_dim=hidden_dim,
                                                               n_heads=n_heads, 
                                                               dropout=dropout) 
                                    for _ in range(num_layers)])
        
    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
            
        return src


class Transformer2DEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=.1):
        super(Transformer2DEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(in_channels=hidden_dim,
                                            k_channels=hidden_dim//n_heads,
                                            v_channels=hidden_dim//n_heads,
                                            n_head=n_heads,
                                            dropout=dropout)
        
        self.feedforward2D = FeedForward2D(hidden_dim=hidden_dim,
                                         dropout=dropout)

        self.feedforward1D = FeedForward1D(hidden_dim=hidden_dim,
                                         dropout=dropout)
        
        self.norm_layer = NormLayer(hidden_dim=hidden_dim)

    def norm(self, x):
        b, c, h, w = x.size()
        out = x.view(b, c, h * w).transpose(1, 2)
        out = self.norm_layer(out)
        out = out.transpose(1, 2).contiguous().view(b, c, h, w)
        return out

    def forward(self, src, src_mask=None):
        size = src.size()
        if len(size) == 4:
            b, c, h, w = size
            src = src.view(b, c, h * w).transpose(1, 2)
        else:
            b, s, c = size
        if src_mask is not None:
            if len(size) == 4:
                src_mask = src_mask.view(b, 1, h * w)
            else:
                if src_mask is not None:
                    src_mask = src_mask.view(b, 1, s)
        
        attn_out, _ = self.attention(src, src, src, src_mask)
        out1 = src + attn_out
        if len(size) == 4:
            out1 = out1.transpose(1, 2).contiguous().view(b, c, h, w)
            out1 = self.norm(out1)          
            ff_out = self.feedforward2D(out1)
            out2 = self.norm(out1 + ff_out)
        else:
            out1 = self.norm_layer(out1)
            ff_out = self.feedforward1D(out1)
            out2 = self.norm_layer(out1+ff_out)

        return out2
