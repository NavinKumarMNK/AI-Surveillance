# author : @NavinKumarMNK

import torch 
import torch.nn.functional as F
import yaml
import math
from utils import l2_norm


class ArcFaceLoss(torch.nn.Module):
    def __init__(self, emb_dim=512, num_classes=None,  s=64., m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.kernel = torch.nn.Parameter(torch.Tensor(emb_dim,num_classes))
    
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.s = s 
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m 
        self.threshold = math.cos(math.pi - m)
        
    def forward(self, embbedings, label):
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)
        cos_theta_m = cos_theta_m.to(dtype=keep_val.dtype)
        # print(cos_theta_m.dtype, keep_val.dtype)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s 
        return output


if __name__ == '__main__':
    torch.manual_seed(0)
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    loss = ArcFaceLoss(**config['loss']['arcface'])
    
    logits = torch.rand(32, 512) # 10 samples, 100 classes
    logits = F.normalize(logits)
    labels = torch.randint(0, 135, (32,))
    
    logits = loss(logits, labels)
    print(logits)
