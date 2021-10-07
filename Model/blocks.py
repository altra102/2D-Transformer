from torch import nn

class NormLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(NormLayer, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(x)
    
    
class FeedForward2D(nn.Module):
    def __init__(self, hidden_dim, dropout=.1):
        super(FeedForward2D, self).__init__()

        #self.conv1 = nn.Conv2d(hidden_dim, hidden_dim*4, 3, padding=1)
        #self.conv2 = nn.Conv2d(hidden_dim*4, hidden_dim, 3, padding=1)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim*4, 1)
        self.conv2 = nn.Conv2d(hidden_dim*4, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.dropout(x)
    
class FeedForward1D(nn.Module):
    def __init__(self, hidden_dim, dropout=.1):
        super(FeedForward1D, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.linear2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)