# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 22:54:32 2021

@author: Min
"""
import torch
import torch.nn as nn
from dataread import MyDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Ms_classificationbase(nn.Module):
    def training_step(self, batch):
        Ms_data, labels = batch
        out = self(Ms_data)                  # Generate predictions
        #loss = F.mse_loss(out, labels)
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    def training_acc(self, batch):
        Ms_data, labels = batch
        out = self(Ms_data)
        acc = accuracy(out, labels)
        return acc
    def validation_step(self, batch):
        Ms_data, labels = batch 
        out = self(Ms_data) 
        #loss = F.mse_loss(out, labels)                   # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.9f}, train_loss: {:.4f}, val_loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'],result['train_acc'], result['val_acc']))



def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d((1,2)))
    return nn.Sequential(*layers)

class ResNet18(Ms_classificationbase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        
        self.conv2 = conv_block(64, 128, pool=True)
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.res3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        self.conv4 = conv_block(256, 512, pool=True)
        self.res4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d((1,2)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(384000, num_classes))
                                        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out)+ out
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out)+out
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out)+ out
        out = self.res3(out) + out
        out = self.conv4(out)
        out = self.res4(out)+ out
        out = self.res4(out) + out
        out = self.classifier(out)
        
        return out
    
    
def vgg_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d((1,2)))
    return nn.Sequential(*layers)
    
class VGG16(Ms_classificationbase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.block1 = nn.Sequential(vgg_block(in_channels, 64), vgg_block(64, 64, pool = True))
        self.block2 = nn.Sequential(vgg_block(64, 128), vgg_block(128, 128, pool = True))
        self.block3 = nn.Sequential(vgg_block(128, 256), vgg_block(256, 256),vgg_block(256, 256, pool = True))
        self.block4 = nn.Sequential(vgg_block(256, 512), vgg_block(512, 512), vgg_block(512, 512, pool = True))
        self.block5 = nn.Sequential(vgg_block(512, 512), vgg_block(512, 512),vgg_block(512, 512,pool = True))
        self.fullyconnected = nn.Sequential(nn.Flatten(),
                                            nn.Linear(384000,1024),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(1024, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(1024, 400),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(400,num_classes))
                                            
        
    def forward(self, xb):
        out = self.block1(xb)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.fullyconnected(out)
        return out
         
def Alex_block(in_channels, out_channels, kernel_sizes = 3,pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=1),
              nn.BatchNorm2d(out_channels)]
    if pool: layers.append(nn.MaxPool2d((1,2)))
    return nn.Sequential(*layers)
    
class AlexNet(Ms_classificationbase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.block1 = nn.Sequential(Alex_block(in_channels, 96, pool = True), 
                                    Alex_block(96, 256, pool = True))
        
        self.block2 = nn.Sequential(nn.Conv2d(256, 384, kernel_size = 3, padding = 1),
                                    Alex_block(384,384),
                                    Alex_block(384,256, pool = True))

        self.fullyconnect = nn.Sequential(nn.Flatten(),
                                          nn.Linear(768000, 400),
                                          nn.Linear(400, num_classes))
                                          
        
    def forward(self, xb):
        out  = self.block1(xb)
        out = self.block2(out)
        out = self.fullyconnect(out)
        return out 

