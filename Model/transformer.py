from torch import nn
import torch
from featureExtractor import FeatureExtractor
from PositionalEncoding import PositionalEncoding1D
from PositionalEncoding import Adaptive2DPositionEncoder
from PositionalEncoding import PositionalEncodingLearned
from PositionalEncoding import PositionalEncodingVaswani
from encoder import TransformerEncoder
from decoder import TransformerDecoder



class Transformer2D(nn.Module):
    def __init__(self, classes, in_features=1024, hidden_dim=256, n_heads=8, 
                 num_layers=4, cnn='resnet50', partial_cnn='layer3', height=250, 
                 width=250, dropout=.1, pretrain=False, onedim=False):
        super(Transformer2D, self).__init__()
        self.classes = classes
        self.onedim = onedim
        #Backbone feature extractor
        self.feature_extractor = FeatureExtractor(name=cnn,
                                                  partial_cnn=partial_cnn,
                                                  in_features=in_features,
                                                  hidden_dim=hidden_dim,
                                                  dropout=dropout,
                                                  pretrain=pretrain)

        #postional encoding
        self.pos_enc_1d = PositionalEncoding1D(hidden_dim)
        self.pos_enc_2d = Adaptive2DPositionEncoder(hidden_dim)

        #embedding
        self.emb = nn.Embedding(classes, hidden_dim)
        #linear
        self.linear = nn.Linear(hidden_dim, classes)
        #mask
        self.trg_mask = None
        self.trg_pad_mask = None
        self.src_mask = None

        #encoder   
        self.encoder = TransformerEncoder(hidden_dim=hidden_dim,
                                            n_heads=n_heads,
                                            num_layers=num_layers,
                                            dropout=dropout)

        #decoder
        self.decoder = TransformerDecoder(hidden_dim=hidden_dim,
                                            n_heads=n_heads,
                                            num_layers=num_layers,
                                            dropout=dropout)
    def pad_mask(self, trg):
        pad_mask = (trg == self.classes)
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)
        return pad_mask

    def generate_square_subsequent_mask(self, trg):
        t = trg.size(1)
        mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
        mask = mask.unsqueeze(0).to(trg.device)
        return mask

    def embedding(self, trg):
        trg = self.emb(trg)
        return trg


    def forward(self, src, trg):
        src = self.feature_extractor(src) 
        if self.onedim:
            src = src.flatten(2).permute(0, 2, 1)  
            src = self.pos_enc_1d(0.1*src)      
        else:
            src = self.pos_enc_2d(0.1*src)

        self.trg_pad_mask = self.pad_mask(trg)
        self.trg_mask = self.generate_square_subsequent_mask(trg)
        self.trg_mask = (self.trg_pad_mask | self.trg_mask)

        trg = self.embedding(trg)
        trg = self.pos_enc_1d(trg)

        enc_out = self.encoder(src, self.src_mask)
        dec_out = self.decoder(trg, enc_out, self.trg_mask, self.src_mask)

        return self.linear(dec_out)
