from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, name, partial_cnn, in_features, hidden_dim, dropout=.1, pretrain=False):
        super(FeatureExtractor, self).__init__()
        self.backbone = models.__getattribute__(name)(pretrained=pretrain) 
        self.stop = partial_cnn                                            
        self.backbone.fc = nn.Conv2d(in_features, hidden_dim, 1)
        self.dropout = nn.Dropout2d(p=dropout)

    def get_features(self, x):
        for name, mod in self.backbone._modules.items():
            x = mod(x)
            if name is self.stop:
                break
        return x

    def forward(self, img): 
        f = self.get_features(img)
        f = self.backbone.fc(f)
        return f