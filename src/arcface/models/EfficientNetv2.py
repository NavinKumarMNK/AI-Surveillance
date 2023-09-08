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
class EfficientNetv2(L.LightningModule):
    def __init__(self, file_path=None, input_size=112):
        super(EfficientNetv2, self).__init__()
        self.file_path = file_path
        self.example_input_array = torch.rand(1, 3, 112, 112)
        self.save_hyperparameters()
        
        if file_path:
            self.model = torch.load(file_path)
        else:        
            self.model = models.efficientnet_v2_s(weights=None)
            # linear => batchnorm => relu => dropout           
            self.model.classifier = nn.Sequential(
                nn.Linear(1280, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(2048, 512),
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
    model = EfficientNetv2()
    
    input_tensor = torch.rand(2, 3, 112, 112)
    output = model(input_tensor)
    print(output.shape)
    
    
    #model.save_model('model')
    
        