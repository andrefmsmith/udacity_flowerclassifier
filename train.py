##### IMPORTS
import argparse

from model_functions import import_model, three_hl_classifier, save_checkpoint
from utility_functions import load_train, load_valid_or_test, choose_device

import torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
import os
##### Example usage for easy reference: python train.py flowers --save_dir checkpoints --arch vgg16 --gpu true

##### SET UP ARGPARSER: data directory (mandatory), save directory, architecture, learning rate, gpu, epochs & hidden units (all optional)
parser = argparse.ArgumentParser('Enter data directory & checkpoint save directory')
parser.add_argument('data_dir', action='store')
parser.add_argument('--save_dir', action='store', help='Where to store checkpoint')
parser.add_argument('--arch', action='store', help='Choose vgg16 or densenet161 architecture')
parser.add_argument('--learning_rate', action='store', type=float)
parser.add_argument('--gpu', action='store', type=bool, help='Choose true to use gpu, false to use cpu')
parser.add_argument('--epochs', action='store', type=int, default=1)
parser.add_argument('--hidden_units', nargs='+', type=int, help='Choose number of hidden units per layer for a 3-layer model')

args = parser.parse_args()



##### LOAD DATA
train_dataset, trainloader = load_train(args.data_dir+'/train')
valid_dataset, validloader = load_valid_or_test(args.data_dir+'/valid')



##### CONFIGURE MODEL

# Import either densenet161 or vgg16
model, L1_in, arch = import_model(args.arch)


# Set hyperparameters: learning rate, number of hidden units (3-layer FC plus output), training epochs.
if args.learning_rate == None:
    learn = 0.003
else:
    learn = args.learning_rate
    
output = 102

if args.hidden_units == None:
    h_units = [L1_in, 2208, 2208, 256, output]
else:
    h_units = [L1_in]+args.hidden_units+[output]
    print(h_units)

if args.epochs == None:
    epochs = 20
else:
    epochs = args.epochs


# Replace pretrained classifier with naive one for flowers
model = three_hl_classifier(model, h_units[0], h_units[1], h_units[2], h_units[3], h_units[4], dropout1=0.1, dropout2=0.1, dropout3=0.1)


# Pick optimizer and loss.
optimizer = optim.Adam(model.classifier.parameters(), lr = learn)
criterion = nn.NLLLoss()


# Choose to train the model on cuda or cpu & how often to display training/validation results.
device = choose_device(args.gpu)
print_every = 50



##### TRAIN MODEL
# Train the beast.
model.to(device)

steps = 0
running_loss = 0

with active_session():
    print('Training started...')
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            log_probs = model.forward(inputs)
            loss = criterion(log_probs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_probs = model.forward(inputs)
                        batch_loss = criterion(log_probs, labels)

                        validation_loss += batch_loss.item()

                        probs = torch.exp(log_probs)
                        top_prob, top_class = probs.topk(1, dim=1)
                        matches = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(matches.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {100*accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()
    print('Training complete.')


##### Save checkpoint
#if args.save_dir not in os.listdir():
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

save_checkpoint(model, train_dataset, h_units[0], h_units[1], h_units[2], h_units[3], h_units[4], epochs, learn, optimizer, criterion, filename=args.save_dir+'/0502_densenet161_ams.pth',arch=arch)