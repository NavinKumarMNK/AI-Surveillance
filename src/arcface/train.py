# author : @NavinKumarMNK

import lightning as L
import lightning.pytorch.loggers as logger
import lightning.pytorch.callbacks as callback
import torch
import torch.nn.functional as F
import numpy as np
import yaml

from tqdm import tqdm_gui as tqdm
from models.EfficientNetv2 import EfficientNetv2
from models.ResNet import ResNet
from loss import ArcFaceLoss
from dataset import (get_transforms, make_dataframe, FaceDataLoader)

__model__ = {
    'efficientnet' : EfficientNetv2,
    'resnet' : ResNet
}

class FaceModel(L.LightningModule):
    def __init__(self, model: str, backbone_path: str, input_size: int,
                 loss_config, pretrained: bool=False, lr: float=1e-3):
        super(FaceModel, self).__init__()
        self.backbone = __model__[model](
            file_path=backbone_path if pretrained else None,
            input_size=input_size,
        )
        self.backbone_path = backbone_path
        self.lr = lr
        self.arcface = ArcFaceLoss(**loss_config)
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()    

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = self.arcface(logits, y)
        loss = self.loss(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = self.arcface(logits, y)
        loss = self.loss(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Adam with Scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, }
    
    def save_model(self):
        self.backbone.save_model(self.backbone_path)


if __name__ == '__main__':
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
    
    # fit 
    trainer.fit(model, dataset)
    model.save_model()