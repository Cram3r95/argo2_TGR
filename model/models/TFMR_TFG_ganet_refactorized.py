#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 17:55:12 2023
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import sys
import os
import pdb
import git

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

if str(sys.version_info[0])+"."+str(sys.version_info[1]) >= "3.9": # Python >= 3.9
    from math import gcd
else:
    from fractions import gcd
    
# DL & Math imports

import math
import numpy as np
import torch
import pytorch_lightning as pl

from numpy import float64, ndarray
from scipy import sparse

from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix

# Plot imports

# Custom imports

from model.models.layers import Conv1d, Res1d, Linear, LinearRes, Null, no_pad_Res1d
from model.models.utils import gpu, to_long,  Optimizer, StepLR

# Global variables 

# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium") # highest, high, medium

#######################################

class TMFModel(pl.LightningModule):
    def __init__(self, args):
        super(TMFModel, self).__init__() # allows us to avoid using the base class name explicitly
        self.args = args
        
        # Save model in log_dir as backup

        self.save_hyperparameters() # It will enable Lightning to store all the provided arguments under the self.hparams attribute. 
                                    # These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.
        
        self.actor_net1 = ActorNet(self.args)
        # self.actor_net2 = ActorNet(self.args)
        self.a2a = A2A(self.args)
        
        self.map_sub_net = MapSubNet(self.args)
        self.A2L_1 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
        self.L2A_1 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)

        self.A2L_2 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
        self.L2A_2 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
        
        self.pred_net = PredNet(self.args)
        
        self.loss_lane = LossLane(self.args)
        
        self.initial_lr = self.args.initial_lr_conf
        self.min_lr = self.args.min_lr_conf
        
        self.is_frozen = False
        self.save_model_script = True

    @staticmethod
    def init_args(parent_parser, BASE_DIR, DATASET_DIR):
        parser_dataset = parent_parser.add_argument_group("dataset")
        parser_dataset.add_argument(
            "--BASE_DIR", type=str, default=BASE_DIR)
        parser_dataset.add_argument(
            "--LOG_DIR", type=str, default="non_specified")
        parser_dataset.add_argument(
            "--train_split", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "train"))
        parser_dataset.add_argument(
            "--val_split", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "val"))
        parser_dataset.add_argument(
            "--test_split", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "test"))
        
        # Social preprocess
        
        parser_dataset.add_argument(
            "--train_split_pre", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_social", "train_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_social", "val_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_social", "test_pre_clean.pkl"))
        
        # Map preprocess
        
        parser_dataset.add_argument(
            "--train_split_pre_map", type=str, default=os.path.join(
            BASE_DIR, DATASET_DIR, "processed_map", "train_map_data.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre_map", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_map", "val_map_data.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre_map", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_map", "test_map_data.pkl"))
        
        parser_dataset.add_argument("--reduce_dataset_size", type=int, default=0)
        parser_dataset.add_argument("--use_preprocessed", type=bool, default=False)
        parser_dataset.add_argument("--use_map", type=bool, default=False)
        parser_dataset.add_argument("--align_image_with_target_x", type=bool, default=True)

        parser_training = parent_parser.add_argument_group("training")
        parser_training.add_argument("--num_epochs", type=int, default=200)
        parser_training.add_argument("--check_val_every_n_epoch", type=int, default=10)
        parser_training.add_argument("--lr_values", type=list, default=[1e-3, 1e-4, 1e-3 , 1e-4])
        parser_training.add_argument("--lr_step_epochs", type=list, default=[10, 20, 30])
        parser_training.add_argument("--initial_lr_conf", type=float, default=1e-3)
        parser_training.add_argument("--min_lr_conf", type=float, default=5e-6)
        parser_training.add_argument("--wd", type=float, default=0.01)
        parser_training.add_argument("--batch_size", type=int, default=32)
        parser_training.add_argument("--val_batch_size", type=int, default=32)
        parser_training.add_argument("--workers", type=int, default=0) # TODO: Not working with >= 0
        parser_training.add_argument("--val_workers", type=int, default=0)
        parser_training.add_argument("--gpus", type=int, default=1)

        parser_model = parent_parser.add_argument_group("model")
        parser_dataset.add_argument("--MODEL_DIR", type=str, default="non_specified")
        parser_model.add_argument("--data_dim", type=int, default=2)
        parser_model.add_argument("--obs_len", type=int, default=50)
        parser_model.add_argument("--pred_len", type=int, default=60)
        parser_model.add_argument("--centerline_length", type=int, default=40)
        parser_model.add_argument("--num_centerlines", type=int, default=6)
        parser_model.add_argument("--actor2actor_dist", type=float, default=100.0)
        parser_model.add_argument("--num_attention_heads", type=int, default=8)
        parser_model.add_argument("--apply_dropout", type=float, default=0.2)
        parser_model.add_argument("--data_aug_gaussian_noise", type=float, default=0.05)
        parser_model.add_argument("--social_latent_size", type=int, default=64)
        parser_model.add_argument("--map_latent_size", type=int, default=64)
        parser_model.add_argument("--final_latent_info", type=str, default="non_specified")
        parser_model.add_argument("--decoder_latent_size", type=int, default=-1)
        parser_model.add_argument("--decoder_temporal_window_size", type=int, default=30) # 49 
        parser_model.add_argument("--num_modes", type=int, default=6)
        parser_model.add_argument("--freeze_decoder", type=bool, default=False)
        parser_model.add_argument("--mod_steps", type=list, default=[1, 5]) # First unimodal -> Freeze -> Multimodal
        parser_model.add_argument("--mod_freeze_epoch", type=int, default=20)
        parser_model.add_argument("--mod_full_unfreeze_epoch", type=int, default=40)
        parser_model.add_argument("--reg_loss_weight", type=float, default=3.0) # xy predictions
        parser_model.add_argument("--cls_loss_weight", type=float, default=1.0) # classification = confidences
        parser_model.add_argument("--epsilon", type=float, default=0.0001)
        parser_model.add_argument("--mgn", type=float, default=0.2)
        parser_model.add_argument("--cls_th", type=float, default=2.0)
        parser_model.add_argument("--cls_ignore", type=float, default=0.2)

        return parent_parser

    def add_noise(self, input, factor=1):
        """_summary_
        Args:
            input (_type_): _description_
            factor (int, optional): _description_. Defaults to 1.
        Returns:
            _type_: _description_
        """

        noise = factor * torch.randn(input.shape).to(input)
        noisy_input = input + noise
        return noisy_input
    
    def forward(self, batch):  
        agents_per_sample = [x.shape[0] for x in batch["displ"]]
        batch_size = len(agents_per_sample)
        focal_agent_id = np.cumsum(agents_per_sample)
        focal_agent_id = np.roll(focal_agent_id,1)
        focal_agent_id[0] = 0
        
        actors, actor_idcs = actor_gather(gpu(batch["displ"]))
        actor_ctrs = gpu(batch["centers"])
        
        if self.training:
            actors[:,:self.args.data_dim,:] = self.add_noise(actors[:,:self.args.data_dim,:], self.args.data_aug_gaussian_noise)
            # actor_ctrs = self.add_noise(actor_ctrs, self.args.data_aug_gaussian_noise)
            
        actors_1 = self.actor_net1(actors, actor_ctrs)
        # actors_2 = self.actor_net2(actors, actor_ctrs)
        # actors = actors_1 + actors_2
        actors = actors_1
        
        actors = self.a2a(actors, actor_idcs, actor_ctrs)
        
        if self.args.use_map:
            
            ### Get relevant centerlines (non-padded) per scenario
            
            rel_candidate_centerlines = batch["rel_candidate_centerlines"]
            rel_candidate_centerlines = torch.stack(rel_candidate_centerlines,dim=0)
            
            # Data augmentation (TODO: It should be in collate_fn_dict, in the DataLoader)

            # if self.training:
            #     rel_candidate_centerlines = self.add_noise(rel_candidate_centerlines, self.args.data_aug_gaussian_noise)
                
            ### Get the map latent vector associated 

            _, num_centerlines, points_centerline, data_dim = rel_candidate_centerlines.shape
            rel_candidate_centerlines = rel_candidate_centerlines.contiguous().view(-1, points_centerline, data_dim)

            non_empty_mask = rel_candidate_centerlines.abs().sum(dim=1).sum(dim=1) # A padded-centerline must sum 0.0
            # in each dimension, and after that both dimensions together
            rows_mask = torch.where(non_empty_mask == 0.0)[0]
            non_masked_centerlines = rel_candidate_centerlines.shape[0] - len(rows_mask)

            rel_candidate_centerlines_mask = torch.zeros([rel_candidate_centerlines.shape[0]], device=rel_candidate_centerlines.device).type(torch.bool) # False
            rel_candidate_centerlines_mask[rows_mask] = True # Padded-centerlines
            rel_candidate_centerlines_mask_inverted = ~rel_candidate_centerlines_mask # Non-padded centerlines (so, relevant) to True
            
            centerlines_per_sample = [] # Relevant centerlines (non-padded) per sequence
            num_current_centerlines = 0
            
            for i in range(rel_candidate_centerlines_mask.shape[0]+1):
                if i % self.args.num_centerlines == 0 and i > 0: # Next traffic scenario
                    centerlines_per_sample.append(num_current_centerlines)
                    num_current_centerlines = 0
                    
                    if i == rel_candidate_centerlines_mask.shape[0]:
                        break
                if rel_candidate_centerlines_mask_inverted[i]: # Non-masked
                    num_current_centerlines += 1
            
            assert non_masked_centerlines == sum(centerlines_per_sample), \
                "The number of relevant centerlines do not match"
        
            centerlines_per_sample = np.array(centerlines_per_sample)
            rel_candidate_centerlines_ = rel_candidate_centerlines[rel_candidate_centerlines_mask_inverted,:,:]
            rel_candidate_centerlines_mask_ = rel_candidate_centerlines_mask.reshape(-1,1).repeat_interleave(points_centerline,dim=1)

            physical_info = self.map_sub_net(rel_candidate_centerlines, rel_candidate_centerlines_mask_) 
        
        social_info = actors[focal_agent_id,:]

        physical_info = physical_info + self.A2L_1(physical_info, social_info)
        social_info = social_info + self.L2A_1(social_info, physical_info)
        
        physical_info = physical_info + self.A2L_2(physical_info, social_info)
        social_info = social_info + self.L2A_2(social_info, physical_info)

        merged_info = social_info
        
        # actor_idcs__ = [actor_idcs_[0] for actor_idcs_ in actor_idcs]
        actor_idcs__ = torch.arange(batch_size,dtype=torch.int32).tolist()
        
        actor_ctrs__ = [actor_ctrs_[0] for actor_ctrs_ in actor_ctrs]
            
        # prediction
        
        out, feats = self.pred_net(merged_info, actor_idcs__, actor_ctrs__)

        # Iterate over each batch and transform predictions into the global coordinate frame
        
        rot, orig = gpu(batch["rotation"]), gpu(batch["origin"])
        
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        out["focal_agent_id"] = focal_agent_id
        
        return out

    # Overwrite Pytorch-Lightning functions
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                        weight_decay=self.args.wd,
                                        lr=self.initial_lr)
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='min',
                                                                factor=0.5,
                                                                patience=5,
                                                                min_lr=1e-6,
                                                                verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "ade_val"}

    def on_train_epoch_start(self):
        # Get learning rate according to current epoch
        for single_param in self.optimizers().param_groups:
            self.log("lr", single_param["lr"], prog_bar=True, sync_dist=True)
        
    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch)
        loss = self.loss_lane(out, train_batch)

        self.log("loss_train", loss["loss"], sync_dist=True)

        return loss["loss"]

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch)
        loss = self.loss_lane(out, val_batch)
        self.log("loss_val", loss["loss"], sync_dist=True)

        # Extract target agent only
        # pdb.set_trace()
        # pred = [x[0].detach().cpu().numpy() for x in out["reg"]]
        pred = torch.cat(out["reg"],0)
        pred = pred.detach().cpu().numpy()
        # conf = [x[0].detach().cpu().numpy() for x in out["cls"]]
        conf = torch.stack(out["cls"],0)
        conf = conf.detach().cpu().numpy()
        gt = [x[0].detach().cpu().numpy() for x in val_batch["gt"]]
        has_preds = [x[0] for x in val_batch["has_preds"]]

        # if self.save_model_script:
        #     model_filename = os.path.join(self.args.BASE_DIR,
        #                                   self.args.MODEL_DIR,
        #                                   "TFMF_TGR.py")
        #     os.system(f"cp {model_filename} {self.args.LOG_DIR}")
        #     self.save_model_script = False
            
        return {"predictions": pred, 
                "confidences": conf,
                "groundtruth": gt,
                "has_preds": has_preds} # = validation_outputs
        
    def validation_epoch_end(self, validation_outputs):
        # Extract predictions
        pred = [out["predictions"] for out in validation_outputs]
        pred = np.concatenate(pred, 0) 
        conf = [out["confidences"] for out in validation_outputs]
        conf = np.concatenate(conf, 0)
        
        has_preds = [out["has_preds"] for out in validation_outputs]
        has_preds = sum(has_preds,[])
        # has_preds = np.concatenate(has_preds, 0)
        gt = [out["groundtruth"] for out in validation_outputs]
        gt = np.concatenate(gt, 0) 

        ade1, fde1, ade, fde, brier_fde, min_idcs = pred_metrics(pred, gt, has_preds, conf)

        print("ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f, brier_fde %2.4f" % (ade1, fde1, ade, fde, brier_fde))
        print()
        
        self.log("ade1_val", ade1, prog_bar=True, sync_dist=True)
        self.log("fde1_val", fde1, prog_bar=True, sync_dist=True)
        self.log("ade_val", ade, prog_bar=True, sync_dist=True)
        self.log("fde_val", fde, prog_bar=True, sync_dist=True)
        self.log("brier_fde_val", brier_fde, prog_bar=True, sync_dist=True)

def pred_metrics(preds, gt_preds, has_preds_, preds_cls):
    #assert has_preds.all()

    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    cls = np.asarray(preds_cls, np.float32)
    m, num_mods, num_preds, _ = preds.shape
    # has_preds = torch.cat([x for x in has_preds_], 0)
    has_preds = torch.stack(has_preds_).detach().cpu()

    last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(has_preds.device) / float(num_preds)
    max_last, last_idcs = last.max(1)
    
    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))
    
    row_idcs_last = np.arange(len(last_idcs)).astype(np.int64) 
    ade1 =  np.asarray([err[i, 0, :last_idcs[i]].mean() for i in range(m)]).mean()
    fde1 = err[row_idcs_last, 0, last_idcs].mean()
    #cls = softmax(cls, axis=1)
    min_idcs = err[row_idcs_last, :, last_idcs].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    cls = cls[row_idcs, min_idcs]
    ade = np.asarray([err[i, :last_idcs[i]].mean() for i in range(m)]).mean()
    fde = err[row_idcs_last, last_idcs].mean()
    one_arr = np.ones(m)
    brier_fde = (err[row_idcs_last, last_idcs] + (one_arr-cls)**2).mean()

    return ade1, fde1, ade, fde, brier_fde, min_idcs

# Aux functions and layers

def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs

class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, args):
        super(ActorNet, self).__init__()
        self.args = args
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 32, 64, 128]
        blocks = [no_pad_Res1d, Res1d, Res1d, Res1d]
        num_blocks = [1, 2, 2, 2]
        # num_blocks = [1, 1, 1, 1]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        self.latent_size = self.args.social_latent_size
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], self.latent_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)
        
        ctrs_in = 2
        self.lstm_h0_init_function = nn.Linear(ctrs_in, self.latent_size, bias=False)
        self.lstm_encoder = nn.LSTM(self.latent_size, self.latent_size, batch_first=True)
        
        self.output = Res1d(self.latent_size, self.latent_size, norm=norm, ng=ng)
        
    def forward(self, actors: Tensor, actor_ctrs) -> Tensor:
        actor_ctrs = torch.cat(actor_ctrs, 0)
        
        actors_aux = torch.zeros(actors.shape[0],actors.shape[1],50).to(actors)
        actors_aux[:,:,1:] = actors
        actors_aux[:,2,0] = actors[:,2,0]
        
        out = actors_aux
        M,d,L = actors.shape

        outputs = []
        for i in range(len(self.groups)):   
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])    
        out = self.output(out)
        out_init = out[:, :, -1]
    
        #1. TODO fuse map data as init hidden and cell state
        h0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, self.latent_size)
        c0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, self.latent_size)

        out = out.transpose(1, 2).contiguous()
        output, (hn, cn) = self.lstm_encoder(out, (h0, c0))
        out_lstm = hn.contiguous().view(M, self.latent_size)
        out = out_lstm + out_init

        return out

class map_smooth_decoder(nn.Module):
    def __init__(self, args):
        super(map_smooth_decoder, self).__init__()

        self.args = args
        self.latent_size = self.args.map_latent_size
        
        self.norm0 = nn.BatchNorm1d(self.latent_size)
        
        self.conv1 = nn.Conv1d(self.latent_size, self.latent_size // 4, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(self.latent_size // 4)
        
        self.conv2 = nn.Conv1d(self.latent_size // 4, self.latent_size // 8, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(self.latent_size // 8)
        
        self.linear3 = nn.Linear(self.args.centerline_length * (self.latent_size // 8), self.latent_size // 8)
        self.norm3 = nn.BatchNorm1d(self.latent_size // 8)
        
        self.linear4 = nn.Linear(self.args.num_centerlines * (self.latent_size // 8), self.latent_size)

    def forward(self, x):
        total_centerlines = x.shape[0]
        batch_size = x.shape[0] // self.args.num_centerlines
        
        x = x.permute(0, 2, 1)
        x = self.norm0(x)
        
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.norm2(F.relu(self.conv2(x)))
        
        x = self.norm3(F.relu(self.linear3(x.contiguous().view(total_centerlines,-1))))
        x = self.linear4(x.contiguous().view(batch_size,-1))

        return x

class MLP(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size // 2)
        self.norm = nn.LayerNorm(output_size // 2)
        self.GELU = nn.GELU()
        self.linear2 = nn.Linear(output_size // 2, output_size)
        
        # self.linear1 = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.GELU(x)
        x = self.linear2(x)
        return x
    
class MapSubNet(nn.Module):

    def __init__(self, args, depth=None):
        super(MapSubNet, self).__init__()
        
        self.args = args
        
        if depth is None:
            depth = 2

        self.hidden_size = self.args.map_latent_size
        self.input_dim = self.args.data_dim
        self.dropout = self.args.apply_dropout

        self.MLPs = nn.ModuleList([MLP(self.input_dim, self.hidden_size // 8), MLP(self.hidden_size // 4, self.hidden_size // 2)])
        self.Attn = nn.ModuleList([nn.MultiheadAttention(self.hidden_size // 8, self.args.num_attention_heads, dropout=self.dropout), 
                                   nn.MultiheadAttention(self.hidden_size // 2, self.args.num_attention_heads, dropout=self.dropout)])
        self.Norms = nn.ModuleList([nn.LayerNorm(self.hidden_size // 4), nn.LayerNorm(self.hidden_size)])

        self.final_layer = map_smooth_decoder(self.args)

    def forward(self, inputs, inputs_mask):

        hidden_states_batch = inputs
        hidden_states_mask = inputs_mask

        for layer_index, layer in enumerate(self.Attn):
            hidden_states_batch = self.MLPs[layer_index](hidden_states_batch)
            if torch.any(torch.isnan(hidden_states_batch)):
                pdb.set_trace()
            temp = hidden_states_batch 
            query = key = value = hidden_states_batch.permute(1,0,2)

            # hidden_states_batch = layer(query, key, value=value, attn_mask=None, key_padding_mask=hidden_states_mask)[0].permute(1,0,2)
            hidden_states_batch = layer(query, key, value=value)[0].permute(1,0,2)
            if torch.any(torch.isnan(hidden_states_batch)):
                pdb.set_trace()
            hidden_states_batch = torch.cat([hidden_states_batch, temp], dim=2)
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)
            if torch.any(torch.isnan(hidden_states_batch)):
                pdb.set_trace()
        if torch.any(torch.isnan(hidden_states_batch)):
            pdb.set_trace()
            
        hidden_states_batch = self.final_layer(hidden_states_batch)
        return hidden_states_batch
    
class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    For the student (without map)
    """
    def __init__(self, args):
        super(PredNet, self).__init__()
        self.args = args
        norm = "GN"
        ng = 1

        self.latent_size = self.args.social_latent_size
        self.num_modes = self.args.num_modes

        pred = []
        for i in range(self.args.num_modes):
            pred.append(
                nn.Sequential(
                    LinearRes(self.latent_size, self.latent_size, norm=norm, ng=ng),
                    nn.Linear(self.latent_size, 2 * self.args.pred_len),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = myAttDestNoMap(self.latent_size, self.num_modes)
        self.cls = nn.Sequential(
            LinearRes(self.latent_size, self.latent_size, norm=norm, ng=ng), nn.Linear(self.latent_size, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        # feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs, actor_idcs) 
        feats = self.att_dest(actors, torch.stack(actor_ctrs), dest_ctrs, actor_idcs) 
        cls = self.cls(feats).view(-1, self.num_modes)
        cls = self.softmax(cls)
        
        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out, feats
    
class myAttDestNoMap(nn.Module):
    def __init__(self, n_agt: int, K):
        super(myAttDestNoMap, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)
        self.K = K
        #self.K_agt = Linear(self.K * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor, actor_idcs) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)
        ctrs = []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs.append(dest_ctrs[idcs])

        for k in range(num_mods):
            dest_ctr = dest_ctrs[:,k,:]
            dist = (agt_ctrs - dest_ctr).view(-1, 2)
            dist = self.dist(dist)
            actors = torch.cat((dist, agts), 1)
            actors = self.agt(actors)
            if k == 0:
                k_actors = actors
            else:
                k_actors = torch.cat((k_actors, actors), 1)
        k_actors = k_actors.view(-1, n_agt)
        # agts = self.K_agt(k_actors)
        return k_actors

class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self, args):
        super(A2A, self).__init__()
        self.args = args
        norm = "GN"
        ng = 1

        n_actor = self.args.social_latent_size

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.args.actor2actor_dist,
            )
        return actors
    
class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """
    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float) -> Tensor:
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts
    
class PostProcess(nn.Module):
    def __init__(self, args):
        super(PostProcess, self).__init__()
        self.args = args

    def forward(self, out,data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["cls"] = [x[0:1].detach().cpu().numpy() for x in out["cls"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1] for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        # preds = np.concatenate(metrics["preds"], 0)
        # gt_preds = np.concatenate(metrics["gt_preds"], 0)
        # has_preds = np.concatenate(metrics["has_preds"], 0)
        # ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        # print(
        #     "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
        #     % (loss, cls, reg, ade1, fde1, ade, fde)
        # )
        # print()
        
        preds = np.concatenate(metrics["preds"], 0)
        preds_cls = np.concatenate(metrics["cls"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = metrics["has_preds"]
        #has_preds = np.concatenate(metrics["has_preds"], 0)
        pdb.set_trace()
        ade1, fde1, ade, fde, brier_fde, min_idcs = pred_metrics(preds, gt_preds, has_preds, preds_cls)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f, brier_fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde, brier_fde)
        )
        print()

def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data

class PredLoss(nn.Module):
    def __init__(self, args):
        super(PredLoss, self).__init__()
        self.args = args
        
        self.num_modes = self.args.num_modes
        self.pred_len = self.args.pred_len
        
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds_: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls_, reg_ = out["cls"], out["reg"]
        # cls = torch.cat([x for x in cls], 0)
        # reg = torch.cat([x for x in reg], 0)
        # gt_preds = torch.cat([x for x in gt_preds], 0)
        # has_preds = torch.cat([x for x in has_preds], 0)

        cls = torch.stack([x for x in cls_], 0)
        reg = torch.cat([x for x in reg_], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        gt_preds = gt_preds[out["focal_agent_id"],:,:]
        has_preds = torch.cat([x for x in has_preds_], 0)
        has_preds = has_preds[out["focal_agent_id"],:]

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.num_modes, self.pred_len
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.args.cls_th).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.args.cls_ignore
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.args.mgn
        coef = self.args.cls_loss_weight
        loss_out["cls_loss"] += coef * (
            self.args.mgn * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.args.reg_loss_weight
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out

class LossLane(nn.Module):
    def __init__(self, args):
        super(LossLane, self).__init__()
        self.args = args
        self.pred_loss = PredLoss(args)

    def forward(self, out: Dict, data: Dict) -> Dict:

        # Add here 'has_preds' -> A mask to know if the object was observed in that frame
        
        future_trajs = gpu(data["fut_trajs"])
        has_preds = [future_trajs_[:,:,-1].bool() for future_trajs_ in future_trajs]
        data["has_preds"] = has_preds
        # pdb.set_trace()
        loss_out = self.pred_loss(out, gpu(data["gt"]), gpu(data["has_preds"]))

        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) \
                         + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out 
    
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, head_num=8, dropout=0.1) -> None:
        super(TransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, head_num, dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_size, head_num, dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.linear1 = nn.Linear(hidden_size, 256)
        self.linear2 = nn.Linear(256, hidden_size)

    def forward(self, x_padding, y_padding):

        self_attn_output = self.self_attn(query=x_padding,
                                     key=x_padding, 
                                     value=x_padding)[0]
        x_padding = x_padding + self.dropout1(self_attn_output)
        x_padding = self.norm1(x_padding)

        cross_attn_output = self.cross_attn(query=x_padding,
                                            key=y_padding,
                                            value=y_padding)[0]

        x_padding = x_padding + self.dropout2(cross_attn_output)
        x_padding = self.norm2(x_padding)

        output = self.linear1(x_padding)
        output = F.relu(output)
        output = self.dropout3(output)
        output = self.linear2(output)

        x_padding = x_padding + self.dropout4(output)
        x_padding = self.norm3(x_padding)

        return x_padding
    

    
