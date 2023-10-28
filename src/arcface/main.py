# author : @NavinKumarMNK

import torch.nn.functional as F
import torch
import yaml
import asyncio
import numpy as np
import os
import pandas as pd

from db import VectorDB
from utils import wandb_push_model
from model import FaceModel
from PIL import Image
from dataset import get_val_transforms
from tqdm import tqdm


class Train():
    def __init__(self, optimization, post_training, push_registry, 
                 model_path, compile_bs, device, method, data_path):
        self.optimization = optimization
        self.post_training = post_training
        self.device = device
        self.push_registry = push_registry
        self.model_path = model_path
        self.compile_bs = compile_bs
        self.method = method
        self.data_path = data_path
        self.db = VectorDB()
    
    @torch.no_grad() 
    def do_inference(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = Image.open(imgs[i])
            imgs[i] = self.transform(imgs[i])    
        
        imgs = torch.stack(imgs).to(self.device)
        output = self.model(imgs)
        # outputs {num_emb, emb_dim} normalize for every vector
        output = F.normalize(output, dim=-1)
        output = output.cpu().numpy().tolist()
        return output
    
    def _upload_dict(self):   
        for folder_label in tqdm(os.listdir(self.data_path)):
            imgs = []
            for faces in os.listdir(os.path.join(self.data_path, folder_label)):
                imgs.append(os.path.join(self.data_path, folder_label, faces))             
            
            embeddings = self.do_inference(imgs)
            
            if self.method == 'all':
                res = [{
                    'payload': {'label' : folder_label},
                    'vector': vector
                } for vector in embeddings]
                
                    
            if self.method == 'avg':
                embeddings = np.array(embeddings)
                embeddings = np.mean(embeddings, axis=0)
                
                
                res = [{
                    'payload': {'label' : folder_label},
                    'vector': embeddings.tolist()
                }]
                            
            yield res
            
    async def update_db(self):
        res = await self.db.verify_collection()
        if res:
            print('Collection Deleted')
            await self.db.delete_collection()
        await self.db.create_collections()
        await self.db.verify_collection()
    
        # upload vectors - batch addition
        for data in self._upload_dict():
            res = await self.db.insert_vectors(
                data=data
            )
        
        res = self.db.create_snapshot()
        return res
         
    
    async def run_pipeline(self, train: bool):
        # Training : Check the config.yaml file and run the training
        from train import run
        
        # run & test
        self.model = run(train=train)
        self.model.eval()
        self.transform = get_val_transforms(
            config={'input_size':112}
        )
        
        # optimization - Optional
        if self.optimization:
            path= self.model.finalize(batch_size=self.compile_bs)

        # post training - vectors to db    
        if self.post_training:
            response = await self.update_db()
            print(response)            

        # push to registry            
        if self.push_registry:
            wandb_push_model(
                model_path=path,
            )

        
class Inference():    
    def __init__(self, ckpt_path, device): 
        self.device = device
        self.model = torch.load(ckpt_path)
        self.model.eval()
        
        self.model = self.model.to(self.device)
        self.db = VectorDB()
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)
        
    
    async def inference(self, img_path):
        transform = get_val_transforms(self.config['data']) 
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img)
            features = F.normalize(features)
            features = features.cpu().numpy().tolist()[0]
            res = await self.db.search(features)
            return res

def run(train: bool):    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
    train_obj = Train(**config['train_pipeline'])
    asyncio.run(train_obj.run_pipeline(train))

async def inference():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    img_path = '/workspace/SurveillanceAI/src/arcface/temp/OIP.jpeg'
    infer = Inference(**config['inference_pipeline'])
    pred = await infer.inference(img_path)
    print(pred)
    
if __name__ == '__main__':
    run(train=False)
    asyncio.run(inference())
    