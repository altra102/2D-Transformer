import cv2
from preprocessfunctions import illumination_compensation, remove_cursive_style
import numpy as np
from glob import glob
import csv
import os


#list of list into list
def flatten(t):
    return [item for sublist in t for item in sublist]

#returns dictionary with images and labels with train,validation,test partitions
def preprocess_iam(iam_image_filepaths, iam_label_filepath, trainset_path,validationset_path,testset_path, img_path="" , input_size=(1024,128,3)):

    def get_image_filenames(file, img_path=""):
        f = open(file)
        filenames = list(csv.reader(f))
        filenames = [''.join(name) for name in filenames]
        if img_path == "":
            pass
        else:
            file_paths=[glob(img_path + f"/**/{dir}*.png", recursive = True) for dir in filenames]
            path_list = flatten(file_paths)
            filenames = [os.path.basename(path).strip('.png') for path in path_list]
        f.close()
        return filenames


    #returns filepaths to corresponding filenames given by train,test,valid -set
    def get_image_filepaths(image_paths,image_filenames):
        filtered_paths = []

        for file in image_paths:
            if any(substring in file for substring in image_filenames):
                filtered_paths.append(file)
                continue
        
        return filtered_paths

#change to rwth aachen 
    def preprocess_images(dataset):
        images = []

        for path in dataset:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            h, w = img.shape    # ex: (89, 1661)
            w_new, h_new, _ = input_size  # (1024, 128, 1)

            scale = max((w / w_new), (h / h_new)) # scale either height or width to keep ratio
            scaled_w = int(max(min(w_new, w // scale), 0)) # ex: 1024
            scaled_h = int(max(min(h_new, h // scale), 0)) # ex: 89//1.62.. = 54
            scaled_size = (scaled_w, scaled_h) # ex : (1024, 54)

            img = illumination_compensation(img)  #standardizing imgs
            img = remove_cursive_style(img)
            img = cv2.resize(img, scaled_size) # ex: (54, 1024)

            pad = np.full((h_new, w_new), 255, dtype=np.uint8) # create white image with input size
            pad[0:scaled_size[1], 0:scaled_size[0]] = img # ex: (128, 1024), insert image into white image to creat padding # maybe get it centerd?
            #pad = np.roll(pad, h - scaled_h, axis=0) #center approach, h - scaled_h = center # wasnt so good bc not all pictures are at top?
            img = cv2.transpose(pad) # ex: (1024, 128)

            img = np.repeat(img[..., np.newaxis],3, -1) # ex: (1024, 128, 3) resnet compatibility

            images.append(img)
        return images


    def remove_whitespace(label):      # "***" problem , maybe with regex but is it nec.?           #standardizeing
        return label.replace( "|", " ").replace(" .", ".").replace(" ,", ",").replace("( ", "(") \
                    .replace(" :", ":").replace(" !", "!").replace(" ?", "?").replace(" )", ")") \
                    .replace(" '", "'")


    def preprocess_labels(label_file, image_filenames):
        f = open(label_file)
        lines = list(csv.reader(f, delimiter=" ", quoting=csv.QUOTE_NONE))[23:] #skip first 24 lines
        labels = []

        for line in lines:
            if any(substring in line[0] for substring in image_filenames):
                if len(line) > 10:    # errors in label file, problems with the delimiter
                    s = line[8] + line[9] + line[10]
                elif len(line) == 10:
                    s = line[8] + line[9]
                else:
                    s = line[8]
                clean_label = remove_whitespace(s)
                labels.append(clean_label)
                continue

        f.close()
        return labels

    def create_iam_dict(train_images,validation_images,test_images,train_labels,validation_labels,test_labels):
        iam_dict = dict()
        datasets = ['train', 'validation', 'test']

        for dataset in datasets:
            iam_dict[dataset] = {'image': [], 'label': []}

        iam_dict['train']['image'] = train_images
        iam_dict['train']['label'] = train_labels

        iam_dict['validation']['image'] = validation_images
        iam_dict['validation']['label'] = validation_labels

        iam_dict['test']['image'] = test_images
        iam_dict['test']['label'] = test_labels
        
        return iam_dict

    


    #get filepaths to corresponding images
    trainset = get_image_filepaths(iam_image_filepaths,get_image_filenames(trainset_path, img_path))
    validationset = get_image_filepaths(iam_image_filepaths,get_image_filenames(validationset_path, img_path))
    testset = get_image_filepaths(iam_image_filepaths,get_image_filenames(testset_path, img_path))
    #get labels from correct imagenames
    train_labels = preprocess_labels(iam_label_filepath, get_image_filenames(trainset_path, img_path))
    validation_labels = preprocess_labels(iam_label_filepath, get_image_filenames(validationset_path, img_path))
    test_labels = preprocess_labels(iam_label_filepath, get_image_filenames(testset_path, img_path))
    #get images
    train_images = preprocess_images(trainset)
    validation_images = preprocess_images(validationset)
    test_images = preprocess_images(testset)
    #fill dictionary
    iam_dict = create_iam_dict(train_images,validation_images,test_images,train_labels,validation_labels,test_labels)

    return iam_dict