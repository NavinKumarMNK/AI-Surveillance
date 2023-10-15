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
                 compile_bs, device, upload_size):
        self.optimization = optimization
        self.post_training = post_training
        self.device = device
        self.push_registry = push_registry
        self.model_path = model_path
        self.payload_map = payload_map
        self.compile_bs = compile_bs
        self.upload_size= upload_size
        self.db = VectorDB()
    
    def _upload_dict(self):
        self.dataset.batch_size = 1
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
                output = output.cpu().numpy().tolist()[0]
                label = label.cpu().numpy().tolist()[0]
                payload = self.payload_map.loc[
                    self.payload_map['label_id'] == label, 'label'].values[0]
                ret_lst.append({
                    'payload': {'label' : payload},
                    'vector': output
                })
                
                if len(ret_lst) % self.upload_size == 0:
                    yield ret_lst
                    ret_lst = []   

        if len(ret_lst) % self.upload_size != 0:
            yield ret_lst
            
            
    async def update_db(self):
        res = await self.db.verify_collection()
        if res:
            await self.db.delete_collection()
            await self.db.create_collections()
            await self.db.verify_collection()
        else:
            await self.db.create_collections()
            await self.db.verify_collection()
        
        # upload vectors - batch addition
        for data in self._upload_dict():
            res = await self.db.insert_vectors(
                data=data
            )
        
        res = self.db.create_snapshot()
        return res
         
    
    async def train(self):
        # Training : Check the config.yaml file and run the training
        from train import run, test
        
        # run & test
        run()
        self.model, self.dataset = test()
        
        path = self.model.save_model()
        self.payload_map = pd.read_csv(self.payload_map)
        
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
        print(img)
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img)
            features = F.normalize(features)
            features = features.cpu().numpy().tolist()[0]
            res = await self.db.search(features)
            return res

def train():    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
    train_obj = Train(**config['train_pipeline'])
    asyncio.run(train_obj.train())

async def inference():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    img_path = '/workspace/SurveillanceAI/src/arcface/data/faces/shah_rukh_khan/0bbdb98f05.jpg'
    infer = Inference(**config['inference_pipeline'])
    pred = await infer.inference(img_path)
    print(pred)
    
if __name__ == '__main__':
    #train()
    asyncio.run(inference())
    