import torch
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from modules.datasets import CustomDataset
from modules.optimizers import get_optimizer
from modules.losses import get_loss
from models.utils import get_model
from pytorch_lightning import LightningModule

from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

class LightningModel(LightningModule):
    def __init__(self, model_name, model_args, criterion, optimizer, learning_rate, weight_decay, batch_size, num_workers, threshold, seed):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(model_name = model_name, model_args = model_args)
        self.criterion = get_loss(loss_name=criterion)
        self.optimizer = get_optimizer(optimizer_name=optimizer)
        self.optimizer = self.optimizer(params=self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.threshold = threshold
        self.seed = seed

        self.miou = BinaryJaccardIndex()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        self.f1scroe = BinaryF1Score()


    def forward(self, inputs):
        x = self.model(inputs)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output, labels)

        output = output > self.threshold
        
        self.miou(output, labels)
        self.recall(output, labels)
        self.precision(output, labels)
        self.f1scroe(output, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_miou", self.miou.compute(), prog_bar=True)
        self.log("train_recall", self.recall.compute(), prog_bar=True)
        self.log("train_precision", self.precision.compute(), prog_bar=True)
        self.log("train_f1scroe", self.f1scroe.compute(), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output, labels)

        output = output > self.threshold

        self.miou(output, labels)
        self.recall(output, labels)
        self.precision(output, labels)
        self.f1scroe(output, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_miou", self.miou.compute(), prog_bar=True)
        self.log("val_recall", self.recall.compute(), prog_bar=True)
        self.log("val_precision", self.precision.compute(), prog_bar=True)
        self.log("val_f1scroe", self.f1scroe.compute(), prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        return self(x)

    def on_train_epoch_end(self):
        self.miou.reset()
        self.recall.reset()
        self.precision.reset()
        self.f1scroe.reset()

    def on_validation_epoch_end(self):
        self.miou.reset()
        self.recall.reset()
        self.precision.reset()
        self.f1scroe.reset()
    

    
    def configure_optimizers(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  
                'interval': 'epoch',
                'frequency': 2
            }
        }
    
    ####################
    # DATA RELATED HOOKS
    ####################
    def prepare_data(self) -> None:
        path_images = '/home/hyj/ChanHyung/Image_segementation/Forest_Fire_Segmentation/dataset/train_img/'
        path_masks = '/home/hyj/ChanHyung/Image_segementation/Forest_Fire_Segmentation/dataset/train_mask/'

        images_paths = glob(path_images + '*.tif')
        masks_paths = glob(path_masks + '*.tif')

        images_paths = sorted([str(p) for p in images_paths])
        masks_paths = sorted([str(p) for p in masks_paths])

        self.df_train = pd.DataFrame({'images': images_paths, 'masks': masks_paths})

        path_images = '/home/hyj/ChanHyung/Image_segementation/Forest_Fire_Segmentation/dataset/test_img/'
        path_masks = '/home/hyj/ChanHyung/Image_segementation/Forest_Fire_Segmentation/dataset/test_mask/'

        images_paths = glob(path_images + '*.tif')
        masks_paths = glob(path_masks + '*.tif')

        images_paths = sorted([str(p) for p in images_paths])
        masks_paths = sorted([str(p) for p in masks_paths])

        self.df_test = pd.DataFrame({'images': images_paths, 'masks': masks_paths})


    def setup(self, stage=None):
        self.wildfire_train = CustomDataset(self.df_train)
        self.wildfire_val = CustomDataset(self.df_test, train_mode=False)
        self.wildfirer_test = CustomDataset(self.df_test, train_mode=False)
    
    def train_dataloader(self):
        return DataLoader(self.wildfire_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.wildfire_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.wildfirer_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.wildfirer_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
