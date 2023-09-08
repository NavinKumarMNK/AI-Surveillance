# author : @NavinKumarMNK

import torch 
import numpy as np
import yaml
import torch

class ArcFaceLoss(torch.nn.Module):
    """
    ArcFaceLoss is a PyTorch implementation of the ArcFace loss function for face recognition.

    Args:
        s (float): Scale factor for the logits. Default is 30.0.
        m (float): Margin value for the loss function. Default is 0.5.
        easy_margin (bool): Whether to use the "easy margin" variant of the loss function. Default is False.
        interclass_filtering_threshold (float): Threshold value for interclass filtering. Default is 0.0.

    Inputs:
        logits (torch.Tensor): Input tensor of shape (batch_size, num_classes). The logits are the output of the neural network before the softmax activation function is applied.
        labels (torch.Tensor): Target tensor of shape (batch_size,). Each element in the tensor is an integer representing the class label for the corresponding input sample.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes). The tensor contains the modified logits after applying the ArcFace loss function.
    """
    def __init__(self, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

    def forward(self, logits, labels):
        theta = torch.acos(logits) # cos_theta = logits
        if self.easy_margin:
            phi = torch.where(theta < torch.pi - self.m, torch.cos(theta + self.m), logits)
        else:
            phi = torch.cos(theta + self.m)
        
        top_k = torch.zeros_like(logits)
        top_k.scatter_(1, labels.view(-1, 1), 1)

        logits = torch.where(top_k.bool(), phi, logits)
        logits *= self.s
        
        return logits

if __name__ == '__main__':
    torch.manual_seed(0)
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    loss = ArcFaceLoss(**config['loss']['arcface'])
    
    logits = torch.rand(10, 5)*2 -1 # 10 samples, 100 classes
    labels = torch.randint(0, 5, (10,))
    
    print(logits, labels)
    logits1 = loss(logits, labels)
    print(logits1)
