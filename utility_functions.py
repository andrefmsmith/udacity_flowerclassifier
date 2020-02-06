import torch, torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np


def load_train(trainpath, re=251,rot=30,crop=224, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], b_size = 64):
    train_transforms = transforms.Compose([transforms.Resize(re),
                                           transforms.RandomRotation(rot),
                                           transforms.RandomResizedCrop(crop),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stds)])
    
    train_dataset = datasets.ImageFolder(trainpath, transform=train_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    
    return train_dataset, trainloader


def load_valid_or_test(datapath, re=251,crop=224, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], b_size = 64):
    vt_transforms = transforms.Compose([transforms.Resize(re),
                                           transforms.CenterCrop(crop),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stds)])
    
    vt_dataset = datasets.ImageFolder(datapath, transform=vt_transforms)
    
    vt_loader = torch.utils.data.DataLoader(vt_dataset, batch_size=b_size, shuffle=True)
    
    return vt_dataset, vt_loader


def get_labels(filepath):
    import json
    
    with open(filepath, 'r') as f:
        label_dic = json.load(f)
        
    return label_dic


def choose_device(gpu):
    if gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        print('Choose to train on GPU (type in cuda) or CPU (type in cpu).')
        device = input()
        device = device.lower()
        while device not in ['cuda', 'cpu']:
            print('Please choose cuda or cpu.')
            device = input()
            device = device.lower()
        if torch.cuda.is_available():
            pass
        else:
            device = 'cpu'
        
    return device


def process_image(image, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225],minsize=256,cropsize=244,bitdepth=256.0):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)

    shortest = im.size.index(min(im.size))
    if shortest == 0:
        longest = 1
    else:
        longest = 0

    sizes = [ [], []]
    sizes[shortest] = minsize
    sizes[longest] = im.size[longest]
    im.thumbnail(sizes)

    size_thumb = im.size
    size_thumb

    left = (size_thumb[0]-cropsize)//2
    up = (size_thumb[1]-cropsize)//2

    im_crop = im.crop((left, up, left + cropsize, up + cropsize))
    im_crop.size

    np_image = np.array(im_crop)/bitdepth
    np_image -= means
    np_image /= stds
    np_image = np_image.transpose((2,1,0))
    
    
    return torch.from_numpy(np_image).type(torch.FloatTensor)