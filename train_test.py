# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 22:42:49 2021

@author: Min
"""
import torch
import torch.nn as nn
import numpy as np
from dataread import MyDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from backbone_Min import *
from Confusion_matrix import *


def get_default_device():
    """if the GPU is available, pick GPU else CPU"""
    if torch.cuda.is_available():
        print('GPU is using. Rich man!')
        return torch.device('cuda')
    else:
        print('CPU is using. Bro, I recommend you to get a GPU!')
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
      

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr,weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        train_acc =[]
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            acc = model.training_acc(batch)
            train_acc.append(acc)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_acc).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def train():
    batch_size = 10
    epochs = 50
    max_lr = 1e-3
    grad_clip = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    opt_func = torch.optim.Adam # when you used the Adagrad, pls off the sched function
    
    
    device = get_default_device()
    Dataset = MyDataset(data_path = r'F:\FYP\enlarge_dataset', annotation_path = r'F:\FYP\data\annotarion_1500_N.csv', transform = transforms.ToTensor())
    train_ds, val_ds = random_split(dataset=Dataset,
                                    lengths=[int(4500*.8), 4500-int(4500*.8)],
                                    generator = torch.Generator().manual_seed(0))
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size*2)
    
    
    train_dl = DeviceDataLoader(train_dl,device)
    val_dl = DeviceDataLoader(val_dl, device)
    
    model = to_device(ResNet18(1, 4), device) # you can change the model at here
                                                
    history = fit(epochs, max_lr, model, train_dl, val_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func = opt_func)

    torch.save(model.state_dict(), 'Mc_resnet18.pth')    


def test():
    batch_size = 10
    weight = r'F:\FYP\code_nes9\weight\Mc_VGG16.pth'
    device = get_default_device()
    Dataset = MyDataset(data_path = r'F:\FYP\data\renamed_file\txtdata', annotation_path = r'F:\FYP\data\test_ds.csv', transform = transforms.ToTensor())
    test_dl = DataLoader(Dataset, batch_size, shuffle=True)
    test_dl = DeviceDataLoader(test_dl, device)
    model = to_device(VGG16(1, 4), device)
    model.load_state_dict(torch.load(weight))
    model.eval()
    result = []
    label = []
    for i in test_dl:
        Ms_data, labels = i
        out = model(Ms_data)
        _, preds = torch.max(out, dim = 1)
        pred = list(preds.cpu().numpy())
        tag = list(labels.cpu().numpy())
        result.extend(pred)
        label.extend(tag)
    return result, label



if __name__=='__main__':

    
    
    
    