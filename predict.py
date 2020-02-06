##### IMPORTS
import argparse

from model_functions import load_checkpoint, predict
from utility_functions import load_train, get_labels, choose_device, process_image

from PIL import Image
import torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
import numpy as np
##### Example usage for easy reference python predict.py flowers/test/1/image_06743.jpg checkpoints/0502_ams.pth

##### ARGPARSER
parser = argparse.ArgumentParser('Enter path to image file.')

parser.add_argument('image', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--cat_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store', type=bool, default=True)
parser.add_argument('--top_k', action='store', type=int, default=5)

args = parser.parse_args()


##### PREDICTION
### Load model checkpoint and dictionary
#train_dataset, trainloader = load_train('flowers/train')
model, epochs, learn, optim_state, optimizer, criterion, classes_d = load_checkpoint(args.checkpoint)
label_dict = get_labels(args.cat_names)
#model.class_to_idx = train_dataset.class_to_idx

### Configure device to use - gpu or cpu
device = choose_device(args.gpu)

### Load and process image, run forward propagation to obtain probabilities and classes for top_k predictions
image = process_image(args.image)

probabilities, classes = predict(image, model, device, classes_d, k=args.top_k)

### Print result
print('The top {} predictions of the model are the following:'.format(args.top_k))

for p, c, r in zip(probabilities, classes, range(1, args.top_k + 1)):
    print('{}. {}, p = {}'.format(r, label_dict[str(c)], round(p,3)))