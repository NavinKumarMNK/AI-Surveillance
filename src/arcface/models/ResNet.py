# author : @NavinKumarMNK

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import wandb
import torch.nn as nn
import tensorrt as trt
import onnx
from torchvision import models

# Encoder
class ResNet(L.LightningModule):
    def __init__(self, file_path=None, input_size=112):
        super(ResNet, self).__init__()
        self.file_path = file_path
        self.example_input_array = torch.rand(1, 3, 112, 112)
        self.save_hyperparameters()
        
        if file_path:
            self.model = torch.load(file_path)
        else:        
            self.model = models.resnet50(weights=None, )
            # linear => batchnorm => relu => dropout           
            self.model.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.25),
            )
        
    def forward(self, x):
        return self.model(x)

    def save_model(self, file_path=None):
        print("Saving Model")
        self.file_path = file_path or self.file_path
        torch.save(self.model, self.file_path+'.pt')
  
if __name__ == '__main__':
    model = ResNet()
    
    input_tensor = torch.rand(2, 3, 112, 112)
    output = model(input_tensor)
    print(output.shape)
    print(model)
    
    # no of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    
    
    #model.save_model('model')
    
        