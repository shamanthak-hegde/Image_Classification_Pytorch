import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from data_generator import ImageDataset
from torchvision.models import resnet50
from tqdm import tqdm
from sklearn.metrics import roc_curve, f1_score
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Focal_Loss(nn.CrossEntropyLoss):
    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
       
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss)


class ImageClassification(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)
        loss = criterion(out, labels)
        return loss

    def validation_step(self, val_pred, val_target, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        bs = images.shape[0]
        out = self(images)
        for i in range(bs):
            val_pred.append(out[i])
            val_target.append(labels[i])
        loss = criterion(out, labels)
        acc = accuracy(out, labels)
        return val_pred, val_target, {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs, f1_score):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'val_f1': f1_score.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}".format(epoch+1, result['train_loss'], result['val_loss'], result['val_acc'], result['val_f1']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def f1(pred, target):
    sft = nn.Softmax(dim=0)
    out = []
    for i in range(len(pred)):
        out.append(torch.argmax(sft(pred[i])).item())
        target[i] = target[i].cpu().detach().numpy()
    out = f1_score(target, out)
    return out


class CNN(ImageClassification):
    def __init__(self, classes):
        super().__init__()
        self.model = resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, classes)
        
    def forward(self, image):
        features = self.model(image)
        return features

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = []
    val_pred = []
    val_target = []
    for batch in val_loader:
        val_pred, val_target, bo = model.validation_step(val_pred, val_target, batch)
        outputs.append(bo)
    f1_score = f1(val_pred, val_target)
    plot_roc_curve(val_target, val_pred)
    return model.validation_epoch_end(outputs, f1_score)


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
        if(epoch+1 % 100 == 0):
            train_acc = evaluate(model, train_loader)
            print("train_acc: {:.4f}, train_loss: {:.4f}, train_f1: {:.4f}".format(train_acc['val_acc'], result['train_loss'], train_acc['val_f1']))
    return history

def plot_roc_curve(target, pred):
    """
    plots the roc curve based of the probabilities
    """
    sft = nn.Softmax(dim=0)
    out = []
    for i in range(len(pred)):
        out.append(torch.argmax(sft(pred[i])).item())
    if(not type(target[0]) == np.ndarray):
        for i in range(len(pred)):
            target[i] = target[i].cpu().detach().numpy()
    fpr, tpr, _ = roc_curve(target, out)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

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
    train = r"data/train"
    val = r"data/val"

    classes = os.listdir(train)

    train_data = ImageDataset(train)
    val_data = ImageDataset(val)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CNN(len(classes))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.model.layer4.parameters():
        param.requires_grad = True

    for param in model.model.fc.parameters():
        param.requires_grad = True
    
    epochs=100
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    # criterion = Focal_Loss(gamma=5)
    batch_size = 128
    
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True,  num_workers=1, pin_memory=False)
    val_loader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle=True,  num_workers=1, pin_memory=False)
    
    
    print('Starting training.....')
    
    history = fit(epochs, model, train_loader, val_loader, optimizer)
    plot_losses(history)
