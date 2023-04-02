import torch
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from data_generator import ImageDataset
from model import CNN
from tqdm import tqdm
from sklearn.metrics import roc_curve, f1_score
import numpy as np
import torch
import torch.nn as nn


class ImageClassification(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)
        loss = criterion(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)
        loss = criterion(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func
    for epoch in tqdm(range(epochs)):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        if (epoch+1) % 100 == 0:
            print('Saving model.......')
            torch.save({'epoch': epoch+1, 
                        'state_dict': model.state_dict()}, os.path.join('save', f'model-epoch-{epoch+1}.pth'))
    return history


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    plt.savefig("Loss_vs_Epoch.png")
    plt.show()

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='scratch', choices=['scratch', 'finetune'], required=True, help="train from scratch or finetune")
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'resnet101', 'alexnet', 'mobilenetv2', 'mobilenetv3', 'vgg19'], required=True, help="select from the pytorch model hub")
    parser.add_argument('--load', help="load pretrained checkpoint")
    args = parser.parse_args()

    train = r"data/train"
    val = r"data/val"

    classes = os.listdir(train)

    train_data = ImageDataset(train)
    val_data = ImageDataset(val)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CNN(len(classes), args.model, args.mode)

    if(args.load):
        checkpoint = torch.load(args.load)
        if('state_dict' in checkpoint):
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)

    # Uncomment and free appropriate layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # for param in model.model.layer4.parameters():
    #     param.requires_grad = True

    # for param in model.model.fc.parameters():
    #     param.requires_grad = True
    
    epochs=100
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True,  num_workers=1, pin_memory=False)
    val_loader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle=True,  num_workers=1, pin_memory=False)
    
    
    print('Starting training.....')
    
    history = fit(epochs, model, train_loader, val_loader, optimizer)
    plot_losses(history)
