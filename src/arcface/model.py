# author : @NavinKumarMNK

import torch
import lightning as L
import tensorrt as trt
import openvino.runtime as ov
import os 

from models.EfficientNetv2 import EfficientNetv2
from models.IRSeNet import IRSeNet
from models.MobileNet import MobileNet
from models.IResNet import IResNet
from loss import ArcFaceLoss
from typing import Union
from openvino.tools.mo import convert_model
from typing import List

__model__ = {
    'efficientnet' : EfficientNetv2,
    'irsenet' : IRSeNet,
    'mobilenet': MobileNet,
    'iresnet': IResNet,
}

class FaceModel(L.LightningModule):
    def __init__(self, model: str, backbone_path: str, input_size: int,
                 num_classes: int, loss_config, pretrained: bool, 
                 lr: float=1e-3, momentum: float=0.9,
                 emb_dim: int=512):
        super(FaceModel, self).__init__()
        self.backbone: Union[EfficientNetv2, IRSeNet, MobileNet] = __model__[model](
            file_path=backbone_path if pretrained else None,
            input_size=input_size,
        )
        self.input_size = input_size
        self.backbone_path = backbone_path
        self.learning_rate = lr
        self.momentum = momentum
        self.arcface = ArcFaceLoss(**loss_config,
                                   num_classes=num_classes,
                                   emb_dim=emb_dim)
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()    
        self.backbone.train()

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
        if not isinstance(self.backbone, list):
            backbone = [*self.backbone.modules()]
        paras_only_bn = []
        paras_wo_bn = []
        for layer in backbone:
            if 'model' in str(layer.__class__):
                continue
            if 'container' in str(layer.__class__):
                continue
            else:
                if 'batchnorm' in str(layer.__class__):
                    paras_only_bn.extend([*layer.parameters()])
                else:
                    paras_wo_bn.extend([*layer.parameters()])
        
        if isinstance(self.backbone, MobileNet):
            self.optimizer = torch.optim.SGD([
                                {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                {'params': [paras_wo_bn[-1]] + [self.arcface.kernel], 'weight_decay': 4e-4},
                                {'params': paras_only_bn}
                            ], lr = self.learning_rate, momentum = self.momentum)
        else:
            self.optimizer = torch.optim.SGD([
                                {'params': paras_wo_bn + [self.arcface.kernel], 'weight_decay': 5e-4},
                                {'params': paras_only_bn}
                            ], lr = self.learning_rate, momentum = self.momentum)
        
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.1)
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}
    
    def save_model(self) -> str:
        self.backbone.save_model(self.backbone_path+'.pt')
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
                
