#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 17:55:12 2023
@author: Carlos Gómez-Huélamo
"""

"""
e.g. 

# How to specify the devices:

https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html

# In summary
# e.g. --devices 0 to specifically use GPU 0 # Single-GPU
# e.g. --devices 6,7 to specifically use GPU 6 and 7 # Multi-GPU
           
python model/trainers/train_TGR.py --use_preprocessed True \
                    --use_map True \
                    --devices 6,7 \
                    --final_latent_info "fuse" \
                    --freeze_decoder True \
                    --decoder "decoder_residual" \
                    --feature_dir "test_wandb" \
 
"""

# General purpose imports

import sys
import os
import pdb
import git
import argparse
import logging

# DL & Math imports

import numpy as np
import pytorch_lightning as pl
import wandb

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# Plot imports

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)
DATASET_DIR = "dataset/argoverse2"
MODEL_DIR = "model/models"

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.models.TFMF_TGR import TMFModel

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

parser = argparse.ArgumentParser()
parser = TMFModel.init_args(parser,BASE_DIR,DATASET_DIR)
parser.add_argument("--ckpt_path", type=str, default="no_model_ckpt")
parser.add_argument("--decoder", type=str, default="decoder_residual", required=True)
parser.add_argument("--feature_dir", type=str, default="non_specified", required=True)
parser.add_argument("--devices", type=list, nargs='+', default=[0,1], required=True)

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

def main():
    args = parser.parse_args()
    args.MODEL_DIR = MODEL_DIR
    
    # Initialize train and val dataloaders
    
    dataset = ArgoCSVDataset(args.train_split, args.train_split_pre, args, args.train_split_pre_map)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=False, # n multi-process loading, the drop_last argument drops the last non-full batch of each worker’s iterable-style dataset replica.
        shuffle=True
    )
    
    dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args, args.val_split_pre_map)
    val_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers, # PyTorch provides an easy switch to perform multi-process data loading
        collate_fn=collate_fn_dict, # A custom collate_fn can be used to customize collation, convert the list of dictionaries to the dictionary of lists 
        pin_memory=True # For data loading, passing pin_memory=True to a DataLoader will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
    )
    
    # Callbacks
    
    ## Model checkpoint
    ## Save the model periodically by monitoring a quantity. 
    ## Every metric logged with log() or log_dict() in LightningModule is a candidate for the monitor key.
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint( 
        filename="{epoch}-{loss_train:.2f}-{loss_val:.2f}-{ade1_val:.2f}-{fde1_val:.2f}-{ade_val:.2f}-{fde_val:.2f}",
        monitor="loss_val",
        save_top_k=4
        # save_top_k=-1,
    )
    
    ## TODO EarlyStopping (in this case, after reaching a minimum Learning Rate)
    
    # Logger
    
    LOG_DIR = os.path.join(args.BASE_DIR,
                           "save_models",
                           args.feature_dir+"/")
    wandb.init(project="CGHFormer_trainings",
               dir=LOG_DIR,
               name=args.
               save_code=True)
    wandb_logger = WandbLogger(save_dir=LOG_DIR)
    args.LOG_DIR = LOG_DIR
    
    model = TMFModel(args)

    # distributed training ways : 1 - lightning, 
    #                             2 - nn.DataParallel
    #                             3 - nn.DistributedDataParallel 
    #                             4 - Horovod 
    
    # [['0'],['1'],etc.] -> [0,1,etc.]
    pdb.set_trace()
    args.devices = [int(device[0]) for device in args.devices]
    bb = [int(device[0]) for device in args.devices]
    pdb.set_trace()
    trainer = pl.Trainer(
        default_root_dir=LOG_DIR,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        devices=args.devices,
        accelerator="cuda",
        strategy="ddp", # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html
        # gradient_clip_val=1.1, 
        # gradient_clip_algorithm="value",
        max_epochs=args.num_epochs, # 
        # check_val_every_n_epoch=args.check_val_every_n_epoch
        # detect_anomaly=True
    )
    
    # # Run learning rate finder (does not support ddp)
    # trainer_simple = pl.Trainer()
    # lr_finder = trainer_simple.tuner.lr_find(model,
    #                                          train_loader,
    #                                          val_loader)
    # lr = lr_finder.suggestion()

    # # Results can be found in
    # print(lr_finder.results)

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()

    # # update hparams of the model
    # model.hparams.lr = new_lr
    # pdb.set_trace()
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
