# author : @NavinKumarMNK

import torch.nn.functional as F
import torch
import yaml
import asyncio
import pandas as pd

from db import VectorDB
from utils import wandb_push_model
from model import FaceModel
from PIL import Image
from dataset import get_val_transforms
from tqdm import tqdm

class Train():
    def __init__(self, optimization, post_training, 
                 push_registry, model_path, payload_map,
                 compile_bs, device):
        self.optimization = optimization
        self.post_training = post_training
        self.device = device
        self.push_registry = push_registry
        self.model_path = model_path
        self.payload_map = pd.read_csv(payload_map)
        self.compile_bs = compile_bs
        
        self.db = VectorDB()
    
    def _upload_dict(self):
        self.dataset.setup()
        self.model.eval()
        
        self.model = self.model.to(self.device)
        ret_lst = []
        
        with torch.no_grad():    
            for image, label in tqdm(self.dataset.val_dataloader()):
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model.backbone(image)
                output = F.normalize(output)
                output = output.cpu().numpy().to_list()
                label = label
                payload = self.payload_map[
                    self.payload_map['label_id'] == label
                ]['label'].values[0]
                ret_lst.append({
                    'payload': payload,
                    'vector': output
                })
        
        return ret_lst
    
    def update_db(self):
        res = self.db.verify_collection()
        if not res:
            self.db.create_collections()
            res = self.db.verify_collection()
            print(res)
        
        # upload vectors - batch addition
        res = self.db.insert_vectors(
            data=self._upload_dict()
        )
        print(res)
         
    
    def train(self):
        # Training : Check the config.yaml file and run the training
        from train import run, test
        
        # run & test
        run()
        self.model, self.dataset = test()
        path = self.model.save_model()
    
        # optimization - Optional
        if self.optimization:
            path= self.model.finalize(batch_size=self.compile_bs)

        # post training
        if self.post_training:
            # vectors to db
            self.update_db()
            

        # push to registry            
        if self.push_registry:
            wandb_push_model(
                model_path=path,
            )
        
class Inference():    
    def __init__(self, ckpt_path, device):
        
        self.device = device
        self.model = FaceModel.load_from_checkpoint(ckpt_path)
        self.model.eval()
        
        self.model = self.model.to(self.device)
        self.db = VectorDB()
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)
        
    
    async def inference(self, img_path):
        transform = get_val_transforms(self.config['data']) 
        img = Image.open(img_path)
        img = transform(img)
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img)
            features = F.normalize(features)
            res = self.db.search(features)
            return res

def train():    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
    train_obj = Train(**config['train_pipeline'])
    train_obj.train()

def inference():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    img_path = '/workspace/SurveillanceAI/temp/sek22fio.png'
    infer = Inference(**config['inference_pipeline'])
    pred = infer.inference(img_path)
    return pred

if __name__ == '__main__':
    train()
    #inference()
    