import torch, torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict


def import_model(arch):
    '''Import and freeze either densenet161 or vgg16 as the base architecture, specifying as a string in arch.'''
    while arch == None:
        print('Please type either densenet161 or vgg16 for the script to run.')
        arch = input()
        
        
    arch = arch.lower()
    while arch not in ['vgg16', 'densenet161']:
        print('Please type either densenet161 or vgg16 for the script to run.')
        arch = input()
        arch = arch.lower()

    if arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_layer = 2208
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_layer = 25088
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model, input_layer, arch


def three_hl_classifier(model, fc1_inp, fc1_out, fc2_out, fc3_out, output, dropout1, dropout2, dropout3):
    '''Instantiates a naive 3-hidden layer classifier on top of a pre-trained network.\\\
    Arguments\\\
    model: a pre-trained model imported from torchvision (network).\\\
    fc1_inp: number of input units (int).\\\
    fc1-3_out: number of units per layer(int).\\\
    dropout1-3: dropout probability per layer (float, 0 to 1).'''
    naive_classifier = nn.Sequential(nn.Linear(fc1_inp, fc1_out),nn.ReLU(),nn.Dropout(dropout1),#fc layer1
                             nn.Linear(fc1_out, fc2_out),nn.ReLU(),nn.Dropout(dropout2),#fc layer2
                             nn.Linear(fc2_out, fc3_out),nn.ReLU(),nn.Dropout(dropout3),#fc layer3
                             nn.Linear(fc3_out, output),nn.LogSoftmax(dim=1))#output layer
    
    model.classifier = naive_classifier
    
    return model


def save_checkpoint(model, dataset, fc1_inp, fc1_out, fc2_out, fc3_out, output, epochs, learn, optimizer, criterion, filename, arch):
    #model.class_to_idx = dataset.class_to_idx
    checkpoint = {'L1_in': fc1_inp,
                  'L1_out': fc1_out,
                  'L2_out': fc2_out,
                  'L3_out': fc3_out,
                  'output': output,
                  'epochs': epochs,
                  'lr': learn,
                  'optimizer_state': optimizer.state_dict(),
                  'state_dict': model.classifier.state_dict(),
                  'optimizer': optimizer,
                  'criterion': criterion,
                 'arch': arch,
                 'classes': dataset.class_to_idx}
    
    torch.save(checkpoint, filename)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    while checkpoint['arch'] not in ['vgg16', 'densenet161']:
        print('Please type either densenet161 or vgg16 for the script to run.')
        arch = input()
        arch = arch.lower()
    
    if checkpoint['arch'] == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    #model = models.densenet161(pretrained=True)
    model.classifier = nn.Sequential(  nn.Linear(checkpoint['L1_in'], checkpoint['L1_out']),nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(checkpoint['L1_out'], checkpoint['L2_out']),nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(checkpoint['L2_out'], checkpoint['L3_out']),nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(checkpoint['L3_out'], checkpoint['output']),nn.LogSoftmax(dim=1)
                                    )
    
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    epochs = checkpoint['epochs']
    alpha = checkpoint['lr']
    optim_state = checkpoint['optimizer_state']
    optimizer = checkpoint['optimizer']
    criterion = checkpoint['criterion']
    classes = checkpoint['classes']
    
    
    return model, epochs, alpha, optim_state, optimizer, criterion, classes


def predict(image, model, device, classes_d, k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = image.unsqueeze(0)
    image = image.to(device)
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        log_probs = model.forward(image)
    probs, indices = torch.exp(log_probs).topk(k)
    
    probabilities = [p.item() for p in probs[0]]
    
    classes = []
    
    for key in classes_d:#model.class_to_idx:
        for idx in indices[0]:
            if classes_d[key]==idx.item():#model.class_to_idx[key]==idx.item():
                classes.append(key)
                
    return probabilities, classes
    # TODO: Implement the code to predict the class from an image file