# author : @NavinKumarMNK

import torch
import lightning as L
import tensorrt as trt
import openvino.runtime as ov
import os 

from models.EfficientNetv2 import EfficientNetv2
from models.IResNet import IResNet
from loss import ArcFaceLoss
from typing import Union
from openvino.tools.mo import convert_model
from typing import List

__model__ = {
    'efficientnet' : EfficientNetv2,
    'iresnet' : IResNet,
}

class FaceModel(L.LightningModule):
    def __init__(self, model: str, backbone_path: str, input_size: int,
                 num_classes: int, loss_config, pretrained: bool, 
                 lr: float=1e-3,
                 emb_dim: int=512):
        super(FaceModel, self).__init__()
        self.backbone: Union[EfficientNetv2, IResNet] = __model__[model](
            file_path=backbone_path if pretrained else None,
            input_size=input_size,
        )
        self.input_size = input_size
        self.backbone_path = backbone_path
        self.learning_rate = lr
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def save_model(self) -> str:
        self.backbone.save_model(self.backbone_path)
        return self.backbone_path

    def load_model(self):
        self.backbone = torch.load(self.backbone_path+'.pt')


    def finalize(self, batch_size) -> str:
        self.backbone.to_onnx(
            file_path=self.backbone_path+'.onnx',
            input_sample=torch.randn(batch_size, 3, self.input_size, self.input_size).to(self.device),
            export_params=True,
        )
        
        self.backbone.to_torchscript(
            file_path=self.backbone_path+'_ts.pt',
            method='script',
            example_inputs=torch.randn(batch_size, 3, self.input_size, self.input_size).to(self.device),
        )
        
        self.to_tensorrt(
            example_inputs=torch.randn(batch_size, 3, self.input_size, self.input_size).to(self.device),
        )
    
        self.to_openvino()
        
        return os.path.dirname(self.backbone_path)
        
    
    def to_openvino(self):
        core = ov.Core()
        ov_model = convert_model(self.backbone_path+'.onnx')
        ov.serialize(ov_model, self.backbone_path+'.xml')
    
    
    def to_tensorrt(self, example_inputs):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            builder.max_batch_size=example_inputs.shape[0]
            with open(self.backbone_path+'.onnx', 'rb') as f:
                parser.parse(f.read())
            
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.max_workspace_size = 1 << 30
            
            network.get_input(0).shape = example_inputs.shape
            engine = builder.build_serialized_network(network, config)
            engine = builder.build_engine(network, config)
            
            with open(self.backbone_path + '.trt', 'wb') as f:
                f.write(engine.serialize())
                
            