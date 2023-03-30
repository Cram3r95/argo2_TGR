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

Flags:

--use_map (True,False)
--devices (GPUs available)
TODO: Complete

python model/trainers/train_TGR.py --use_preprocessed True \
                    --devices 5 \
                    --final_latent_info "concat" \
                    --decoder "decoder_residual" \
                    --exp_name "exp_1_multi_agent_actornet_all_feats"
"""
# --freeze_decoder True \
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
# from model.models.TFMF_TGR import TMFModel
from model.models.TFMF_TGR_multi_agent import TMFModel
# from model.models.TFMF_TGR_ganet import TMFModel

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

parser = argparse.ArgumentParser()
parser = TMFModel.init_args(parser,BASE_DIR,DATASET_DIR)
parser.add_argument("--ckpt_path", type=str, default="no_model_ckpt")
parser.add_argument("--decoder", type=str, default="decoder_residual", required=True)
parser.add_argument("--exp_name", type=str, default="non_specified", required=True)
parser.add_argument("--devices", type=list, default=[0], required=True)

# logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
  
def main():
    args = parser.parse_args()
    args.MODEL_DIR = MODEL_DIR
    args.devices = [int(device) for device in args.devices if device.isnumeric()] # ['0',[','],['1'],etc.] -> [0,1,etc.]
    
    # Initialize train and val dataloaders
    
    print("Initialize train split ...")
    dataset = ArgoCSVDataset(args.train_split, args.train_split_pre_social, args, 
                             input_preprocessed_map=args.train_split_pre_map, input_preprocessed_full=args.train_split_pre)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=False, # n multi-process loading, the drop_last argument drops the last non-full batch of each worker’s iterable-style dataset replica.
        shuffle=True
    )

    print("Initialize val split ...")
    dataset = ArgoCSVDataset(args.val_split, args.val_split_pre_social, args, 
                             input_preprocessed_map=args.val_split_pre_map, input_preprocessed_full=args.val_split_pre)
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
        # If it is not specified, dirpath (the folder where we are going to save the checkpoints) matches
        # the default_root_dir variable of the pl.Trainer (see below)
        filename="{epoch}-{loss_train:.2f}-{loss_val:.2f}-{ade1_val:.2f}-{fde1_val:.2f}-{ade_val:.2f}-{fde_val:.2f}",
        monitor="loss_val",
        save_top_k=-1
    )
    
    ## EarlyStopping (in this case, after reaching a minimum Learning Rate)
    
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="lr", 
                                                     patience=5, 
                                                     divergence_threshold=args.min_lr,
                                                     verbose=False, 
                                                     mode="min")
    
    # Logger

    LOG_DIR = os.path.join(args.BASE_DIR,
                           DATASET_DIR,
                           "save_models/wandb",
                           args.exp_name+"/")
    
    if not os.path.exists(LOG_DIR):
        print("Create exp folder: ", LOG_DIR)
        os.makedirs(LOG_DIR) # makedirs creates intermediate folders
   
    wandb.init(project="CGHFormer_lightning_trainings",
               dir=LOG_DIR,
               name=args.exp_name,
               # group=args.exp_name, # all runs for the experiment in one group
               save_code=True)
    wandb_logger = WandbLogger(save_dir=LOG_DIR)
    args.LOG_DIR = LOG_DIR
    
    model = TMFModel(args)

    # distributed training ways : 1 - lightning, 
    #                             2 - nn.DataParallel
    #                             3 - nn.DistributedDataParallel 
    #                             4 - Horovod 
    
    trainer = pl.Trainer(
        default_root_dir=LOG_DIR,
        logger=wandb_logger,
        # callbacks=[checkpoint_callback,early_stop_callback],
        callbacks=[checkpoint_callback],
        devices=args.devices,
        accelerator="cuda",
        strategy="ddp" if len(args.devices)>1 else None, # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html
        gradient_clip_val=1.1, 
        gradient_clip_algorithm="value",
        max_epochs=args.num_epochs, 
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
