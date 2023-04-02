import os
from glob import glob
import pandas as pd
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.classes = os.listdir(self.input_dir)
        self.input = glob(self.input_dir + '/*/*')
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Resize([224,224]),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])


    def __getitem__(self, index):
        images = self.input
        image_name = images[index]
        image = cv2.imread(image_name)
        image = self.transform(image)
        gt = self.classes.index(image_name.split('/')[-2])
        return (image, gt)


    def __len__(self):
        return len(self.input)