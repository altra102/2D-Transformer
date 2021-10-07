# install this before using these functions
#!pip install compress_pickle

from compress_pickle import dump, load

def save_dict(dictionary, save_path, save_name): 
    dump(dictionary, save_path + "/" + save_name +".gz")

def load_dict(pkl_path):
    output = load(pkl_path)
    return output
