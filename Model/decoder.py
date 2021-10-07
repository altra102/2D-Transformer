from torch import nn
from blocks.py import NormLayer, FeedForward1D
from attention.py import MultiHeadAttention


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_layers, dropout=.1):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([Transformer1DDecoderLayer(hidden_dim=hidden_dim,
                                                               n_heads=n_heads,
                                                               dropout=dropout)
                                    for _ in range(num_layers)])
        
    def forward(self, trg, src, trg_mask=None, src_mask=None):
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        
        return trg



class Transformer1DDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=.1):
        super(Transformer1DDecoderLayer, self).__init__()

        self.masked_attention = MultiHeadAttention(in_channels=hidden_dim,
                                            k_channels=hidden_dim//n_heads,
                                            v_channels=hidden_dim//n_heads,
                                            n_head=n_heads,
                                            dropout=dropout)
        
        self.attention = MultiHeadAttention(in_channels=hidden_dim,
                                            k_channels=hidden_dim//n_heads,
                                            v_channels=hidden_dim//n_heads,
                                            n_head=n_heads,
                                            dropout=dropout)
        
        self.feedforward = FeedForward1D(hidden_dim=hidden_dim,
                                         dropout=dropout)
        
        self.norm_layer = NormLayer(hidden_dim=hidden_dim)
        
    def forward(self, trg, src, trg_mask=None, src_mask=None):
        attn_trg, _ = self.masked_attention(trg, trg, trg, trg_mask)
        out1 = self.norm_layer(trg + attn_trg)

        size = src.size()   
        if len(size) == 4:     
            b, c, h, w = size
            src = src.view(b, c, h * w).transpose(1, 2)
            if src_mask is not None:
                src_mask = src_mask.view(b, 1, h * w)  
        else:
            b, s, c = size
            if src_mask is not None:
                src_mask = src_mask.view(b, 1, s)

        attn_out, _ = self.attention(out1, src, src, src_mask)
        out2 = self.norm_layer(out1+attn_out)

        ff_out = self.feedforward(out2)
        out3 = self.norm_layer(out2+ff_out)

        return out3