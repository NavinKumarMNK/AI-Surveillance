# author : @NavinKumarMNK

import torch
import yaml

class ModelOptimizer():
    def __init__(self):
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        self.config = config['optimization']
                
    def optimizer(self):
        pass