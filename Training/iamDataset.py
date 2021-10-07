from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torch import Tensor
from preprocessfunctions import normalization


class dict_dataset(Dataset):
    def __init__(self, data_dict, symbol_encoder, partition='train', transform=None):
        self.data = data_dict
        self.partition = partition
        self.s2i = symbol_encoder

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.data[self.partition]['label'])

    def __getitem__(self, i):
        img = self.data[self.partition]['image'][i]

        img = normalization(img)
        img = self.transform(img)

        label = encode_label(self.data[self.partition]['label'][i],self.s2i)
        label = np.pad(label, (0, 128 - len(label)))
        label = Tensor(label)

        return img, label

def create_alphabet(labels):
    alphabet = ['PAD', 'SOS']

    for label in labels:
        for char in label:
            if char not in alphabet:
                alphabet.append(char)

    alphabet += ['EOS']
    return alphabet

def create_converter(alphabet):
    sym_to_int = {i: sym for sym, i in enumerate(alphabet)}
    int_to_sym = {i: sym for i, sym in enumerate(alphabet)}
    return sym_to_int, int_to_sym 

def encode_label(label, s2i):
    enc_label = [s2i[char] for char in label]
    return [s2i['SOS']] + enc_label + [s2i['EOS']]

def decode_label(label,i2s):
    dec_label = "".join([i2s[char] for char in label])
    if dec_label.find('EOS') == -1:
        return dec_label
    else:
        return dec_label[:dec_label.find('EOS')]
