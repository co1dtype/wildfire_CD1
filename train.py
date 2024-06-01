import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from modules.utils import load_yaml
from modules.trainer import LightningModel

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything



import torch
from pytorch_lightning import seed_everything

from datetime import datetime, timezone, timedelta


import warnings
warnings.filterwarnings('ignore')


# Root Directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yaml')
config = load_yaml(config_path)

# Train Serial
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# Seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed_everything(config['TRAINER']['seed'], workers=True)
# torch.use_deterministic_algorithms(False)


# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['TRAINER']['gpu'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # torch.use_deterministic_algorithms(False)
    model = LightningModel(
        model_name = config['TRAINER']['model'],
        model_args= config['MODEL'],
        criterion=config['TRAINER']['criterion'],
        optimizer=config['TRAINER']['optimizer'],
        learning_rate=config['TRAINER']['learning_rate'],
        weight_decay=config['TRAINER']['weight_decay'],
        batch_size=config['TRAINER']['batch_size'],
        num_workers=config['TRAINER']['num_workers'],
        threshold=config['TRAINER']['threshold'],
        seed=config['TRAINER']['seed'],
    )

    early_stop_callback = EarlyStopping(
        monitor='val_miou', 
        min_delta=0.001,     
        patience=5,          
        verbose=True,        
        mode='min'           
    )

    # "MODEL_PATH": "/kaggle/working/my_logs/",
    # "filename" : 'best_model_{epoch:02d}-{val_acc:.4f}',

    checkpoint_callback = ModelCheckpoint(
        monitor='val_miou',  
        dirpath= config["DIR"]['model_path'],
        filename= config["DIR"]['file_name'],
        save_top_k=1,       
        mode='max'         
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config['TRAINER']['n_epochs'],
        deterministic=True,
        benchmark=False,
        
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    model = model.to(device)
    trainer.fit(model)

