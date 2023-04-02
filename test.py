import torch
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from data_generator import ImageDataset
from model import CNN
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn


class ImageClassification(nn.Module):

    def test_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)
        loss = criterion(out, labels)
        acc = accuracy(out, labels)
        return {'test_loss': loss.detach(), 'test_acc': acc}

    def test_epoch_end(self, outputs):
        batch_losses = [x['test_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['test_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item()}


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.test_step(batch) for batch in test_loader]
    return model.test_epoch_end(outputs)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'resnet101', 'alexnet', 'mobilenetv2', 'mobilenetv3', 'vgg19'], required=True, help="select from the pytorch model hub")
    parser.add_argument('--load', required=False ,help="load pretrained checkpoint")
    args = parser.parse_args()

    test = r"data/test"
    classes = os.listdir(test)
    test_data = ImageDataset(test)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CNN(len(classes), args.model)
    
    if(args.load):
        checkpoint = torch.load(args.load)
        if('state_dict' in checkpoint):
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=True,  num_workers=1, pin_memory=False)
    
    print('Starting testing.....')
    
    history = evaluate(model, test_loader)
    print(history)
