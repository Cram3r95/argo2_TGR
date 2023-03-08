#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 17:55:12 2023
@author: Carlos Gómez-Huélamo
"""

"""
e.g. 
python train_TGR.py --use_preprocessed=True --devices 0 --decoder "decoder_temporal" \
                    --ckpt_path="/home/robesafe/Argoverse2_Motion_Forecasting/lightning_logs/version_9/checkpoints/epoch=9-loss_train=127.62-loss_val=173.21-ade1_val=2.93-fde1_val=7.12-ade_val=2.94-fde_val=6.79.ckpt"

python train_TGR.py --use_preprocessed=True --devices 1 --decoder "decoder_residual" --final_latent_info "fuse" --feature_dir "map_and_social"

python train_TGR.py --use_preprocessed True --devices 0 \
                    --final_latent_info "fuse" \
                    --decoder "decoder_residual" \
                    --feature_dir "map_social_fuse_v2"
                    he
python train_TGR.py --use_preprocessed True \
                    --use_map True \
                    --devices 7 \
                    --final_latent_info "fuse" \
                    --freeze_decoder True \
                    --decoder "decoder_residual" \
                    --feature_dir "test1" \
 
"""
#                     --use_map True \
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
parser.add_argument("--decoder", type=str, default="decoder_residual", required=True)
parser.add_argument("--feature_dir", type=str, default="non_specified", required=True)
parser.add_argument("--devices", type=list, nargs='+', default=[0,1])#, required=True)

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

    log_dir = os.path.join(BASE_DIR,
                            "save_models",
                            args.feature_dir+"/")

    args.log_dir = log_dir
    
    model = TMFModel(args)

    # distributed training ways : 1 - lightning, 
    #                             2 - nn.DataParallel
    #                             3 - nn.DistributedDataParallel 
    #                             4 - Horovod 
    
    # [['0'],['1'],etc.] -> [0,1,etc.]

    args.devices = [int(device[0]) for device in args.devices]

    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
        devices=args.devices,
        accelerator="cuda",
        strategy="ddp", # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html
        #gradient_clip_val=1.1, 
        #gradient_clip_algorithm="value",
        max_epochs=args.num_epochs, # 
        check_val_every_n_epoch=args.check_val_every_n_epoch
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
