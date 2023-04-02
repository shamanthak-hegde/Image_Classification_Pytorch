import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, alexnet, mobilenet_v2, mobilenet_v3_large, vgg19
from train import ImageClassification


class CNN(ImageClassification):
    def __init__(self, classes, model, mode="finetune"):
        super().__init__()
        if(mode=="scratch"):
            pretrained=False
        elif(model=="finetune"):
            pretrained=True
        
        if(model=='resnet50'):
            self.model = resnet50(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif(model=='resnet101'):
            self.model = resnet101(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif(model=='alexnet'):
            self.model = alexnet(pretrained=pretrained)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifer[6] = nn.Linear(num_ftrs, classes)

        elif(model=='mobilenetv2'):
            self.model = mobilenet_v2(pretrained=pretrained)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif(model=='mobilenetv3'):
            self.model = mobilenet_v3_large(pretrained=pretrained)
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, classes)

        elif(self.model=='vgg19'):
            self.model = vgg19(pretrained=pretrained)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, classes)
        
    def forward(self, image):
        features = self.model(image)
        return features