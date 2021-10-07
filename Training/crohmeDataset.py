from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torch import Tensor
import csv
from os import chdir
from preprocessfunctions.py import normalization

class dict_dataset(Dataset):
    def __init__(self, data_dict, symbol_encoder, pad, partition='train', transform=None):
        self.data = data_dict
        self.partition = partition
        self.s2i = symbol_encoder
        self.pad = pad

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

        label = self.data[self.partition]['label'][i].replace(' ','')
        label = encode_label(label, self.s2i)
        label = np.pad(label, (0, self.pad - len(label)))
        label = Tensor(label)

        return img, label
    
def create_alphabet(dir):          
    label_dir = dir + '/Labels'
    chdir(label_dir)
    f = open('latexSymbolWhitelistPart4.txt')
    lines = list(csv.reader(f, delimiter='\n'))

    alphabet = ['PAD', 'SOS',' ', '^', '_', '{', '}', '&', ';']  

    for line in lines:
        alphabet.append(line[0])

    alphabet += ['EOS']
    f.close()
    return alphabet


def get_latex(alphabet):
    latex = []

    for item in alphabet:
        if item[0] in '\\':
            latex.append(item)
    return latex


def remove_useless_latex(enc):
    enc_clear = enc
    if '\mbox' in enc_clear:
        enc_clear.remove('\mbox')
    if '\mathrm' in enc_clear:
        enc_clear.remove('\mathrm')
    if '\left' in enc_clear:
        enc_clear.remove('left')
    if '\right' in enc_clear:
        enc_clear.remove('right')

    return enc_clear

def encode_label(label, s2i, latex):               
    lbl = label.replace('$', '')            
    save = ''
    enc = []
    for char in lbl:
        if char is '\\':
            save += char
            continue
        if save is not '':
            save += char
            for lat in latex:
                if lat in save:
                    if save in s2i:
                        enc.append(s2i[save])
                    save = ''
                    break
            continue
        else:
            if char in s2i:
                enc.append(s2i[char])
    enc = remove_useless_latex(enc)
    return [s2i['SOS']] + enc + [s2i['EOS']]

def create_converter(alphabet):
    sym_to_int = {i: sym for sym, i in enumerate(alphabet)}
    int_to_sym = {i: sym for i, sym in enumerate(alphabet)}
    return sym_to_int, int_to_sym 


def decode_label(label, i2s):
    dec_label = "".join([i2s[char] for char in label])
    if dec_label.find('EOS') == -1:
        return '$' + dec_label
    else:
        return '$' + dec_label.replace('EOS', '$')