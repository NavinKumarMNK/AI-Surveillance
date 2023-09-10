# author : @NavinKumarMNK

import torch.nn.functional as F
import torch
import yaml

class Train():
    def __init__(self, testing, optimization, post_training):
        self.optimization = optimization
        self.testing = testing
        self.post_training = post_training
    
    def train(self):
        # Training : Check the config.yaml file and run the training
        from train import run, test
        
        model, data = run()
        if self.testing:
            test()
        
        # optimization
        if self.optimization:
            pass
        
        # post training
        if self.post_training:
            from db import VectorDB as vecdb
            
            pass
        
class Inference():    
    def __init__(self, ckpt_path, device):
        from model import FaceModel

        self.device = device
        self.model = FaceModel.load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.model.freeze()
        self.model = self.model.to(self.device)
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)
        
    
    def inference(self, img_path):
        from PIL import Image
        from dataset import get_transforms
        
        transform = get_transforms(self.config['data']) 
        img = Image.open(img_path)
        img = transform(img)
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img)
            features = F.normalize(features)


def train_test():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
    train_obj = Train(**config['train_pipeline'])
    train_obj.train()

def inference_test():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    img_path = '/workspace/SurveillanceAI/temp/sek22fio.png'
    infer = Inference(**config['inference_pipeline'])
    

if __name__ == '__main__':
    
    
    
    #infer = Inference(ckpt_path)