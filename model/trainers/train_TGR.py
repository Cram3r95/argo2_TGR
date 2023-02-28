#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 17:55:12 2023
@author: Carlos Gómez-Huélamo
"""

"""
e.g. 
python train_TGR.py --use_preprocessed=True \
                    --devices 4 \
                    --feature_dir "decoder_temporal" \
                    --ckpt_path="/home/robesafe/Argoverse2_Motion_Forecasting/lightning_logs/version_9/checkpoints/epoch=9-loss_train=127.62-loss_val=173.21-ade1_val=2.93-fde1_val=7.12-ade_val=2.94-fde_val=6.79.ckpt"
"""

import os
import argparse
import sys
import logging
import pdb
import torch
import git

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch import nn

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.TFMF_TGR import TMFModel

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser = TMFModel.init_args(parser)
parser.add_argument("--ckpt_path", type=str, default="no_model_ckpt")
parser.add_argument("--feature_dir", type=str, default="non_specified")
parser.add_argument("--devices", type=list, nargs='+', default=[0,1], required=True)

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

def main():
    args = parser.parse_args()

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
    
    # Save the model periodically by monitoring a quantity. Every metric logged with log() or log_dict() in LightningModule is a candidate for the monitor key.
    checkpoint_callback = pl.callbacks.ModelCheckpoint( 
        filename="{epoch}-{loss_train:.2f}-{loss_val:.2f}-{ade1_val:.2f}-{fde1_val:.2f}-{ade_val:.2f}-{fde_val:.2f}",
        monitor="loss_val",
        save_top_k=-1,
    )

    model = TMFModel(args)

    if args.ckpt_path != "no_model_ckpt":
        model = TMFModel.load_from_checkpoint(checkpoint_path=args.ckpt_path)
        # TODO: We also have to load the state of the optimizer

    if args.feature_dir:
        log_dir = os.path.join(BASE_DIR,
                               "lightning_logs",
                               args.feature_dir)
    else:
        log_dir = os.path.join(BASE_DIR,
                               "lightning_logs")
    
    # distributed training ways : 1 - lightning, 
    #                             2 - nn.DataParallel
    #                             3 - nn.DistributedDataParallel 
    #                             4 - Horovod 
    
    # [['0'],['1'],etc.] -> [0,1,etc.]
    args.devices = [int(device[0]) for device in args.devices]
    
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
        devices = args.devices,
        accelerator = "cuda",
        #strategy="ddp",
        max_epochs=args.num_epochs
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
