# author : @NavinKumarMNK

import os
import uuid 
import time
import wandb
import dotenv
import os
import yaml

def find_most_recent_subfolder(folder_path):
    subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    if not subdirectories:
        return None  

    subdirectories.sort(
        key=lambda x: os.path.getctime(os.path.join(folder_path, x)), 
        reverse=True)
    return os.path.join(folder_path, subdirectories[0])

def generate_id():
    # add time to make it unique
    return str(uuid.uuid4())[:8] + '-' + str(int(time.time()))

def wandb_push_model(model_path):
    dotenv.load_dotenv('wandb.env')
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    run = wandb.init(project=config['wandb']['project'],
                     dir=config['wandb']['save_dir'],)
    
    artifact = wandb.Artifact(
        name=config['wandb']['project'],
        type='model',
        metadata={
            "version" : generate_id(),
        },
    )
    artifact.add_dir(model_path)
    run.log_artifact(artifact)
        
