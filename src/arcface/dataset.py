# author: @NavinKumarMNK

from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from torchvision import transforms
from PIL import Image

import os
import pandas as pd
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
    

def get_transforms(config):
    return transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((config['input_size'], config['input_size']), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    ])

def make_dataframe(data_path: str, train_val_split: float=0.8):
    train_label = []
    val_label = []
    train_image_path = []
    val_image_path = []
    
    for label in os.listdir(data_path):
        images = []
        for image in os.listdir(os.path.join(data_path, label)):
            images.append(os.path.join(data_path, label, image))
        
        np.random.shuffle(images)
        split = int(len(images)*train_val_split)
        
        train_images = images[:split]
        val_images = images[split:]
        
        train_label.extend([label]*len(train_images))
        val_label.extend([label]*len(val_images))
        
        train_image_path.extend(train_images)
        val_image_path.extend(val_images)
    
    train_df = pd.DataFrame({'image': train_image_path, 'label': train_label})
    val_df = pd.DataFrame({'image': val_image_path, 'label': val_label})
    
    return {'train': train_df.sample(frac=1), 'val': val_df.sample(frac=1)}
        
class FaceDataSet(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data.iloc[idx]['image']
        label = self.data.iloc[idx]['label']
        
        image = Image.open(image)
   
        if self.transform:
            image = self.transform(image)
            
        image = image/255.0
            
        return image, label
    
class FaceDataLoader(LightningDataModule):
    def __init__(self, data, batch_size=32, num_workers=4, transform=None):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        print(self.batch_size)
        self.setup()
        
    def setup(self):
        self.train_dataset = FaceDataSet(self.data['train'], transform=self.transform)
        self.val_dataset = FaceDataSet(self.data['val'], transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    

if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(config['data']['seed'])
    np.random.seed(config['data']['seed'])
    
    data_path = config['data']['path']
    data = make_dataframe(data_path, train_val_split=config['data']['train_val_split'])
    transform = get_transforms(config['data'])
    
    data_loader = FaceDataLoader(data=data, **config['data']['dataloader'], transform=transform)

    for image, label in data_loader.train_dataloader():
        print(image.shape, label)
        print(label)
        plt.imshow(image[1].permute(1, 2, 0))
        plt.show()
        break
    