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
                 num_classes: int, loss_config, pretrained: bool=False, 
                 lr: float=1e-3,
                 emb_dim: int=512):
        super(FaceModel, self).__init__()
        self.backbone = __model__[model](
            file_path=backbone_path if pretrained else None,
            input_size=input_size,
        )
        self.backbone_path = backbone_path
        self.lr = lr
        self.arcface = ArcFaceLoss(**loss_config,
                                   num_classes=num_classes,
                                   emb_dim=emb_dim)
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()    

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits= self.arcface(logits, y)
        loss = self.loss(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = self.arcface(logits, y)
        loss = self.loss(logits,y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Adam with Scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, }
    
    def save_model(self):
        self.backbone.save_model(self.backbone_path)


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
        print("Keyboard Interrupted")
    model.save_model()
    
    return model, data

def test():

if __name__ == '__main__':
    
    
    model = FaceModel.load_from_checkpoint(
        '/workspace/SurveillanceAI/models/efficientnet/lightning_logs/version_22/checkpoints/epoch=01-val_loss=0.45.ckpt')
    model.eval()
    model.freeze()
    model = model.to('cuda')
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
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
    
    