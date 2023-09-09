import torch
import lightning as L
from models.EfficientNetv2 import EfficientNetv2
from models.ResNet import ResNet
from loss import ArcFaceLoss

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
