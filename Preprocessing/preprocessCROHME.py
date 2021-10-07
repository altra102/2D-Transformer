import cv2
from os import chdir
from preprocessfunctions import illumination_compensation
import numpy as np
from natsort import natsorted
from glob import glob
import csv

from tqdm.notebook import tqdm #use normal tqdm , if using no notebook



def preprocess_crohme(dir, alternate_image_dir=''): 
    if alternate_image_dir is not '':
        image_dir = alternate_image_dir
    else:   
        image_dir = dir + '/Images'
    label_dir = dir + '/Labels'

    crohme_dict = create_crohme_dict()
    
    crohme_dict['train']['image'], crohme_dict['validation']['image'], crohme_dict['test']['image'] = preprocess_images(image_dir)
    crohme_dict['train']['label'], crohme_dict['validation']['label'], crohme_dict['test']['label'] = preprocess_labels(label_dir)
    

    return crohme_dict

def preprocess_images(dir):
    train_dir      = dir + '/train'
    validation_dir = dir + '/validation'
    test_dir       = dir + '/test'

    train_images      = get_images(train_dir)
    validation_images = get_images(validation_dir)
    test_images       = get_images(test_dir)

    return train_images, validation_images, test_images


def preprocess_labels(dir):
    train_dir      = dir + '/train'
    validation_dir = dir + '/validation'
    test_dir       = dir + '/test'

    train_labels      = get_labels(train_dir)
    validation_labels = get_labels(validation_dir)
    test_labels       = get_labels(test_dir)

    return train_labels, validation_labels, test_labels

def more_black(img):
    ret_img = img
    for i, pixelrow in enumerate(img):
        for j, pixel in enumerate(pixelrow):
            if pixel != 255:
                ret_img[i][j] = 35
    return ret_img

def find_smallest_img(img):
    h, w = img.shape
    found = False
    for i, pixel in enumerate(img):
        for j, pixel in enumerate(pixel):
            if pixel != 255:
                found = True
                break
        if found:
            new_h_u = i
            break

    found = False
    for i, pixel in reversed(list(enumerate(img))):
        for j, pixel in enumerate(pixel):
            if pixel != 255:
                found = True
                break
        if found:
            new_h_d = h - i
            break
    return new_h_u, new_h_d

def remove_white(img, up, down, pad):
    h, w = img.shape
    return img[up-pad:h-down+pad, 0:w]

def scale_crohme_img(img, input_size, pad):
    h, w = img.shape

    up, down = find_smallest_img(img)
    img = remove_white(img, up, down, pad)

    h, w = img.shape
    w_new, h_new, _ = input_size

    black_scale = h/w

    scale = max((w / w_new), (h / h_new)) 
    scaled_w = int(max(min(w_new, w // scale), 0))
    scaled_h = int(max(min(h_new, h // scale), 0))
    scaled_size = (scaled_w, scaled_h)

    img = illumination_compensation(img)  
    img = cv2.resize(img, scaled_size, interpolation = cv2.INTER_AREA) 

    pad = np.full((h_new, w_new), 255, dtype=np.uint8)
    pad[0:scaled_size[1], 0:scaled_size[0]] = img

    #if black_scale > 0.15:
        #pad = more_black(pad)

    img = cv2.transpose(pad)
    img = np.repeat(img[..., np.newaxis], 3, -1)
    img = cv2.transpose(img)
    return img

def get_images(dir):
    chdir(dir)
    files = natsorted(glob('*/*.png', recursive=True))

    images = []

    for file in tqdm(files):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = scale_crohme_img(img, (1024, 256, 3), 0)  #output size
        
        images.append(img)

    return images


def get_labels(dir):                                                
    chdir(dir)
    files = natsorted(glob('*/*.inkml', recursive=True))

    truth  = '<annotation type="truth">'
    truthx = '<annotation typx="truth">'
    label = ''
    labels = []
    for file in tqdm(files):
        f = open(file)
        lines = list(csv.reader(f, delimiter='\n'))
        for line in lines:
            if truth in line[0] or truthx in line[0]:
                label = line[0].replace(truth, '').replace(truthx, '').replace('</annotation>', '')
                if label[0] is not '$':
                    label = '$' + label
                if label[-1] is not '$':
                    label = label + '$'
                labels.append(label)
                break
        f.close()

    return labels

def create_crohme_dict():
    crohme_dict = dict()
    datasets = ['train', 'validation', 'test']
    for dataset in datasets:
        crohme_dict[dataset] = {'image': [], 'label': []}
    return crohme_dict


