# author : @NavinKumarMNK

import torch 
import torch.nn.functional as F
import numpy as np
import yaml

class ArcFaceLoss(torch.nn.Module):
    def __init__(self, num_classes, emb_dim, s=30.0, m=0.5, easy_margin=False,
                 sample_rate=0.1):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.sample_rate = sample_rate
        
        self.weights = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, emb_dim)))
        
    def forward(self, features, labels):
        # Normalize feature vectors
        features = F.normalize(features)
        # Normalize weight vectors
        weights = F.normalize(self.weights)
        # Compute cosines of angles between features and weights
        cos_theta = torch.mm(features, weights.t())
        
        theta = torch.acos(cos_theta)
        if self.easy_margin:
            phi = torch.where(theta < torch.pi - self.m, torch.cos(theta + self.m), cos_theta)
        else:
            phi = torch.cos(theta + self.m)
        
        top_k = torch.zeros_like(cos_theta)
        top_k.scatter_(1, labels.view(-1, 1), 1)

        logits = torch.where(top_k.bool(), phi, cos_theta)
        logits *= self.s
        
        return logits

if __name__ == '__main__':
    torch.manual_seed(0)
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    loss = ArcFaceLoss(**config['loss']['arcface'],
                       num_classes=config['general']['num_classes'],
                       emb_dim=config['general']['emb_dim'])
    
    logits = torch.rand(32, 512) # 10 samples, 100 classes
    logits = F.normalize(logits)
    labels = torch.randint(0, 135, (32,))
    
    logits = loss(logits, labels)
    print(logits)
