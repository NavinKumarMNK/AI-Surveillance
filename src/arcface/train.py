# author : @NavinKumarMNK

import lightning as L
import lightning.pytorch.loggers as logger
import lightning.pytorch.callbacks as callback
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os

from utils import find_most_recent_checkpoint
from tqdm import tqdm_gui as tqdm
from model import FaceModel
from dataset import (get_transforms, make_dataframe, FaceDataLoader)


def run():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])
    
    data_path = config['data']['path']
    data = make_dataframe(data_path, train_val_split=config['data']['train_val_split'])
    transform = get_transforms(config['data'])
    
    dataset = FaceDataLoader(data=data, **config['data']['dataloader'], transform=transform)
    
    model=config['model']['type']
    model = FaceModel(model=model, 
                      loss_config=config['loss']['arcface'],
                      num_classes=config['general']['num_classes'],
                      emb_dim=config['general']['emb_dim'],
                      **config['model'][model])
    
    # wandb
    wandb_logger = None
    if config['general']['wandb'] == True:
        wandb_logger = logger.WandbLogger(**config['wandb'])
 
    # callbacks
    callbacks = [
        callback.EarlyStopping(**config['train']['callbacks']['early_stopping']),
        callback.ModelCheckpoint(**config['train']['callbacks']['model_checkpoint']),
        callback.LearningRateMonitor(logging_interval='step'),
        callback.DeviceStatsMonitor(),  
        callback.ModelSummary(max_depth=5),
    ]
        
    # trainer
    trainer = L.Trainer(
        **config['train']['args'], 
        callbacks=callbacks, 
        logger=wandb_logger)
    try:
        trainer.fit(model, dataset)
    except KeyboardInterrupt:
        print("Keyboard Interrupted... Continuing Further")
    model.save_model()
    
    return model, data

def test():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    ckpt_path = find_most_recent_checkpoint(
        config['train']['args']['default_root_dir']
    ) + '/checkpoints/' 
    
    # take the only file in the directory
    ckpt_path = ckpt_path + os.listdir(ckpt_path)[0].split('.')[0]
    
    print(f"Loading checkpoint from {ckpt_path}")
    model = FaceModel.load_from_checkpoint(checkpoint_path=ckpt_path)
        
    model.eval()
    model.freeze()
    model = model.to('cuda')
    
    
    data_path = config['data']['path']
    data = make_dataframe(data_path, 0)
    transform = get_transforms(config['data'])
    
    dataset = FaceDataLoader(data=data, **config['data']['dataloader'], transform=transform)
    
    print("Testing on entire dataset")
    correct = 0
    total = 0
    for image, label in tqdm(dataset.val_dataloader()):
        image = image.to('cuda')
        label = label.to('cuda')
        output = model(image)
        output = F.normalize(output)
        output = model.arcface(output, label)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print('Accuracy of the network on the %d test images: %f %%' % (total, 100 * correct / total))
    
if __name__ == '__main__':
    run()
    test()
    
    
    