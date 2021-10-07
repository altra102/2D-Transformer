from torch.nn.utils import clip_grad_norm_
from torch import no_grad
from tqdm.notebook import tqdm #use normal tqdm if in shell and not in notebook

def train(model, classes, criterion, optimizer, scheduler,dataloader, device):
    model.train()

    total_loss = 0
    total_samples = len(dataloader)

    for imgs, labels in tqdm(dataloader):
          imgs = imgs.to(device)
          labels = labels.to(device)
    
          optimizer.zero_grad()
          output = model(imgs.float(),labels.long()[:,:-1])

          norm = (labels != 0).sum()
          loss = criterion(output.log_softmax(-1).contiguous().view(-1, classes), labels[:,1:].contiguous().view(-1).long()) / norm


          loss.backward()
          clip_grad_norm_(model.parameters(), 0.2)
          optimizer.step()
          total_loss += loss.item() * norm

    return total_loss / total_samples

def evaluate(model, classes, criterion, dataloader, device):
    model.eval()

    epoch_loss = 0
    total_samples = len(dataloader)

    with no_grad():
      for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs.float(),labels.long()[:,:-1])
              
            norm = (labels != 0).sum()
            loss = criterion(output.log_softmax(-1).contiguous().view(-1, classes), labels[:,1:].contiguous().view(-1).long()) / norm

  
            epoch_loss += loss.item() * norm

    return epoch_loss / total_samples


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx          # ['PAD'] = 0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
