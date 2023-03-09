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

from scipy import sparse

from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix

# Plot imports

# Custom imports

from model.models.layers import Conv1d, Res1d, Linear, LinearRes, Null, no_pad_Res1d

# Global variables 

# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high") # highest, high, medium

#######################################

### config ###
config = dict()
"""Train"""
config["display_iters"] = 200000
config["val_iters"] = 200000 
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = False
config["opt"] = "adam"
config["num_epochs"] = 50
config["start_val_epoch"] = 30
config["lr"] = [1e-3, 1e-4, 5e-5, 1e-5]
config["lr_epochs"] = [36,42,46]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "test_results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "test_results", config["save_dir"])
    
# if "save_dir" not in config:
#     config["save_dir"] = os.path.join(
#         root_path, "results_student", model_name
#     )

# if not os.path.isabs(config["save_dir"]):
#     config["save_dir"] = os.path.join(root_path, "results_student", config["save_dir"])

config["batch_size"] = 128
config["val_batch_size"] = 128
config["workers"] = 0
config["val_workers"] = config["workers"]

"""Dataset"""
# data_path = "/usr/src/app/datasets/motion_forecasting/argoverse2"
data_path = "/home/robesafe/shared_home/benchmarks/argoverse2/motion-forecasting"

# Raw Dataset
config["train_split"] = os.path.join(data_path, "train")
config["val_split"] = os.path.join(data_path, "val")
config["test_split"] = os.path.join(data_path, "test")

# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(
    data_path, "preprocess_c", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    data_path, "preprocess_c", "val_crs_dist6_angle90.p"
)
config['preprocess_test'] = os.path.join(data_path, 'preprocess_c', 'test_test.p')

"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["actor2map_dist"] = 7.0
config["map2actor_dist"] = 6.0
config["actor2actor_dist"] = 100.0
config["pred_size"] = 60
config["pred_step"] = 1
config["num_test_mod"] = 3
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["cls_coef"] = 3.0
config["reg_coef"] = 1.0
config["end_coef"] = 1.0
config["test_cls_coef"] = 1.0
config["test_coef"] = 0.2
config["test_half_coef"] = 0.1
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2
config["use_map"] = False
### end of config ###

class TMFModel(pl.LightningModule):
    def __init__(self, args):
        super(TMFModel, self).__init__() # allows us to avoid using the base class name explicitly
        self.args = args
        self.config = config
        
        # Save model in log_dir as backup

        self.save_hyperparameters() # It will enable Lightning to store all the provided arguments under the self.hparams attribute. 
                                    # These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.
        
        # # Encoder
        
        # ## Social
        
        # self.linear_embedding = LinearEmbedding(3,self.args)
        # self.pos_encoder= PositionalEncoding1D(self.args.social_latent_size)
        # self.encoder_transformer = EncoderTransformer(self.args)
        # self.agent_gnn = AgentGNN(self.args)
        
        # ## Physical 
        
        # if self.args.use_map:
            
        #     self.map_sub_net = MapSubNet(self.args)
            
        #     assert self.args.social_latent_size == self.args.map_latent_size
            
        #     if self.args.final_latent_info == "concat":
        #         self.args.decoder_latent_size = self.args.social_latent_size + self.args.map_latent_size
        #     elif self.args.final_latent_info == "fuse":
        #         self.A2L_1 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
        #         self.L2A_1 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
    
        #         self.A2L_2 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
        #         self.L2A_2 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
                
        #         self.args.decoder_latent_size = self.args.social_latent_size
        #     else:
        #         raise AssertionError
        # else:
        #     self.args.decoder_latent_size = self.args.social_latent_size
            
        # if self.args.decoder == "decoder_residual": self.decoder = DecoderResidual(self.args)
        # elif self.args.decoder == "decoder_temporal": self.decoder = Temporal_Multimodal_Decoder(self.args)

        # # Metrics
        
        # self.reg_loss = nn.SmoothL1Loss(reduction="none")
        
        self.config = config
        self.use_map = config["use_map"]
        
        self.actor_net1 = ActorNet(config)
        self.actor_net2 = ActorNet(config)
          
        else:
            self.a2a = A2A(config)
            self.test_half_net = TestNet_Half(config)
            self.pred_net = PredNetNoMap(config)
            
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
        parser_training.add_argument("--initial_lr_conf", type=float, default=0.25e-4)
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
        parser_model.add_argument("--reg_loss_weight", type=float, default=1) # xy predictions
        parser_model.add_argument("--cls_loss_weight", type=float, default=1) # classification = confidences
        parser_model.add_argument("--epsilon", type=float, default=0.0001)

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
        # Set batch norm to eval mode in order to prevent updates on the running means,
        # if the weights are frozen
        if self.args.freeze_decoder:
            if self.is_frozen:
                for module in self.modules():
                    if isinstance(module, torch.nn.modules.BatchNorm1d):
                        module.eval()

        # # Encoder

        # ## Social
        
        # ### Extract the social features in each sample of the current batch
        
        # displ, centers = batch["displ"], batch["centers"]
        # rotation, origin = batch["rotation"], batch["origin"]
        # agents_per_sample = [x.shape[0] for x in displ]
        # batch_size = len(agents_per_sample)
        
        # ### OBS: For each sequence, we always set the focal (target) agent as the first agent
        # ###      of the scene, then our ego-vehicle (AV) and finally the remanining agents
        # ###      (See extractor_proc.py preprocessing)
        
        # focal_agent_id = np.cumsum(agents_per_sample)
        # focal_agent_id = np.roll(focal_agent_id,1)
        # focal_agent_id[0] = 0
        
        # ### Convert the list of tensors to tensors
        
        # displ_cat = torch.cat(displ, dim=0)
        # centers_cat = torch.cat(centers, dim=0)
        
        # ### Data augmentation (TODO: It should be in collate_fn_dict, in the DataLoader)

        # if self.training:
        #     displ_cat[:,:,:2] = self.add_noise(displ_cat[:,:,:2], self.args.data_aug_gaussian_noise)
        #     centers_cat = self.add_noise(centers_cat, self.args.data_aug_gaussian_noise)
        
        # linear_output = self.linear_embedding(displ_cat)
        # pos_encoding = self.pos_encoder(linear_output)
        # pos_encoding = pos_encoding + linear_output

        # out_transformer = self.encoder_transformer(pos_encoding, agents_per_sample)
        # out_agent_gnn = self.agent_gnn(out_transformer, centers_cat, agents_per_sample)

        # social_info = torch.stack([x[0] for x in out_agent_gnn])
        
        # if torch.any(torch.isnan(social_info)):
        #     pdb.set_trace()
        
        # ## Physical
        
        # if self.args.use_map:
            
        #     ### Get relevant centerlines (non-padded) per scenario
            
        #     rel_candidate_centerlines = batch["rel_candidate_centerlines"]
        #     rel_candidate_centerlines = torch.stack(rel_candidate_centerlines,dim=0)
            
        #     # Data augmentation (TODO: It should be in collate_fn_dict, in the DataLoader)

        #     # if self.training:
        #     #     rel_candidate_centerlines = self.add_noise(rel_candidate_centerlines, self.args.data_aug_gaussian_noise)
                
        #     ### Get the map latent vector associated 

        #     _, num_centerlines, points_centerline, data_dim = rel_candidate_centerlines.shape
        #     rel_candidate_centerlines = rel_candidate_centerlines.contiguous().view(-1, points_centerline, data_dim)

        #     non_empty_mask = rel_candidate_centerlines.abs().sum(dim=1).sum(dim=1) # A padded-centerline must sum 0.0
        #     # in each dimension, and after that both dimensions together
        #     rows_mask = torch.where(non_empty_mask == 0.0)[0]
        #     non_masked_centerlines = rel_candidate_centerlines.shape[0] - len(rows_mask)

        #     rel_candidate_centerlines_mask = torch.zeros([rel_candidate_centerlines.shape[0]], device=rel_candidate_centerlines.device).type(torch.bool) # False
        #     rel_candidate_centerlines_mask[rows_mask] = True # Padded-centerlines
        #     rel_candidate_centerlines_mask_inverted = ~rel_candidate_centerlines_mask # Non-padded centerlines (so, relevant) to True
            
        #     centerlines_per_sample = [] # Relevant centerlines (non-padded) per sequence
        #     num_current_centerlines = 0
            
        #     for i in range(rel_candidate_centerlines_mask.shape[0]+1):
        #         if i % self.args.num_centerlines == 0 and i > 0: # Next traffic scenario
        #             centerlines_per_sample.append(num_current_centerlines)
        #             num_current_centerlines = 0
                    
        #             if i == rel_candidate_centerlines_mask.shape[0]:
        #                 break
        #         if rel_candidate_centerlines_mask_inverted[i]: # Non-masked
        #             num_current_centerlines += 1
            
        #     assert non_masked_centerlines == sum(centerlines_per_sample), \
        #         "The number of relevant centerlines do not match"
        
        #     centerlines_per_sample = np.array(centerlines_per_sample)
        #     rel_candidate_centerlines_ = rel_candidate_centerlines[rel_candidate_centerlines_mask_inverted,:,:]
        #     rel_candidate_centerlines_mask_ = rel_candidate_centerlines_mask.reshape(-1,1).repeat_interleave(points_centerline,dim=1)

        #     physical_info = self.map_sub_net(rel_candidate_centerlines, rel_candidate_centerlines_mask_) 
        
        # # Decoder
        
        # if self.args.use_map:
        #     if self.args.final_latent_info == "concat": # Concat info
        #         merged_info = torch.cat([social_info, 
        #                                 physical_info], 
        #                                 dim=1)
                    
        #     if self.args.final_latent_info == "fuse": # Fuse info
        #         physical_info = physical_info + self.A2L_1(physical_info, social_info)
        #         social_info = social_info + self.L2A_1(social_info, physical_info)
                
        #         physical_info = physical_info + self.A2L_2(physical_info, social_info)
        #         social_info = social_info + self.L2A_2(social_info, physical_info)
                
        #         merged_info = social_info
        # else:
        #     merged_info = social_info

        # if torch.any(torch.isnan(merged_info)):
        #     pdb.set_trace()
                    
        # # If self.args.freeze_decoder is set to True, conf are useless
        
        # if self.args.decoder == "decoder_residual":
        #     pred_traj, conf = self.decoder(merged_info, self.is_frozen, self.current_epoch)
        # elif self.args.decoder == "decoder_temporal":
        #     traj_agent_abs_rel = displ_cat[focal_agent_id,:self.args.decoder_temporal_window_size,:self.args.data_dim]
        #     last_obs_agent = centers_cat[focal_agent_id,:]
            
        #     decoder_h = merged_info.unsqueeze(0)
        #     decoder_c = torch.zeros(tuple(decoder_h.shape)).to(decoder_h)
        #     state_tuple = (decoder_h, decoder_c)
            
        #     pred_traj_rel, conf = self.decoder(traj_agent_abs_rel, state_tuple)
            
        #     # Convert relative displacements to absolute coordinates (around origin)

        #     pred_traj = relative_to_abs_multimodal(pred_traj_rel, last_obs_agent)

        # ### In this model we are only predicting
        # ### the focal agent. We would actually 
        # ### have batch_size x num_agents x num_modes x pred_len x data_dim
        
        # num_agents = 1
        # out = pred_traj.contiguous().view(batch_size, num_agents, -1, self.args.pred_len, self.args.data_dim) 
        # if not self.args.freeze_decoder: conf = conf.view(batch_size, num_agents, -1)

        
        # Iterate over each batch and transform predictions into the global coordinate frame
        
        for i in range(len(out)):
            out[i] = torch.matmul(out[i], rotation[i]) + origin[i].view(
                1, 1, 1, -1
            )
        return out, conf

    # Aux class functions
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.decoder.unfreeze_layers()

        self.is_frozen = True
        
    def full_unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
            
        self.is_frozen = False

    def prediction_loss(self, preds, gts, conf=None):
        """_summary_

        Args:
            preds (torch.tensor): batch_size x num_agents x num_modes x pred_len x data_dim
                                  OBS: At this moment, num_agents = 1 since we are only predicting the focal agent 
            gts (list): list of gt of each scenario (num_agents x pred_len x 2)
            conf (torch.tensor): batch_size x num_agents x 1

        Returns:
            _type_: _description_
        """
        
        if self.args.freeze_decoder:
            # # Stack all the predicted trajectories of the target agent
            # num_mods = preds.shape[2]
            # # [0] is required to remove the unneeded dimensions
            # preds = torch.cat([x[0] for x in preds], 0)

            # # Stack all the true trajectories of the target agent
            # # Keep in mind, that there are multiple trajectories in each sample, 
            # # but only the first one ([0]) corresponds to the target agent
            
            # gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0)
            # gt_target = torch.repeat_interleave(gt_target, num_mods, dim=0) # repeate the gt for all ks 

            # loss_single = self.reg_loss(preds, gt_target)
            # loss_single = torch.sum(torch.sum(loss_single, dim=2), dim=1)
            # loss_single = torch.split(loss_single, num_mods)

            # # Tuple to tensor
            
            # loss_single = torch.stack(list(loss_single), dim=0)

            # min_loss_index = torch.argmin(loss_single, dim=1) # Get best mode
            # min_loss_combined = [x[min_loss_index[i]] for i, x in enumerate(loss_single)]
 
            # loss_out = torch.sum(torch.stack(min_loss_combined))
            # # loss_out = torch.mean(torch.stack(min_loss_combined))
            
            # return loss_out
            
            
            
            # Stack all the predicted trajectories of the target agent
            
            preds = preds.squeeze(1)

            batch_size, num_modes, pred_len, data_dim = preds.shape
            
            # Stack all the true trajectories of the target agent
            # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
            # to the target agent
            
            gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0) # batch_size x pred_len x data_dim
            gt_target_repeated = gt_target.unsqueeze(1).repeat(1,preds.shape[1],1,1) # repeate the gt for all ks 
                                                                                     # batch_size x num_modes x pred_len x data_dim

            fde_k = torch.sqrt((preds[:, :, -1, 0] - gt_target_repeated[:, :, -1, 0]) ** 2 + # x
                               (preds[:, :, -1, 1] - gt_target_repeated[:, :, -1, 1]) ** 2 + # y
                               self.args.epsilon) # to avoid division by zero
            k_hat = torch.argmin(fde_k, dim=1) 

            index = torch.tensor(range(preds.shape[0]), dtype=torch.long)
            pred_fut_traj = preds[index, k_hat] # Best trajectory in terms of FDE per sequence  

            batch_size, pred_len, _ = pred_fut_traj.shape
            num_modes = preds.shape[1]
            
            # Regression loss
            
            # reg_loss = torch.zeros(1, dtype=torch.float32).to(preds)

            mse_loss = F.mse_loss(pred_fut_traj, gt_target, reduction='none')
            mse_loss = mse_loss.sum(dim=2) + self.args.epsilon # sum epsilon to avoid division by zero
            mse_loss = torch.sqrt(mse_loss)
            mse_loss = mse_loss.mean(dim=1)

            fde_loss = fde_k[index, k_hat]

            reg_loss = mse_loss * 0.5 + fde_loss * 0.5
            reg_loss = reg_loss.mean()
            
            return reg_loss
        else:
            # Stack all the predicted trajectories of the target agent
            
            preds = preds.squeeze(1)
            conf = conf.squeeze(1)
            batch_size, num_modes, pred_len, data_dim = preds.shape
            
            # Stack all the true trajectories of the target agent
            # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
            # to the target agent
            
            gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0) # batch_size x pred_len x data_dim
            gt_target_repeated = gt_target.unsqueeze(1).repeat(1,preds.shape[1],1,1) # repeate the gt for all ks 
                                                                                     # batch_size x num_modes x pred_len x data_dim

            fde_k = torch.sqrt((preds[:, :, -1, 0] - gt_target_repeated[:, :, -1, 0]) ** 2 + # x
                               (preds[:, :, -1, 1] - gt_target_repeated[:, :, -1, 1]) ** 2 + # y
                               self.args.epsilon) # to avoid division by zero
            k_hat = torch.argmin(fde_k, dim=1) 

            index = torch.tensor(range(preds.shape[0]), dtype=torch.long)
            pred_fut_traj = preds[index, k_hat] # Best trajectory in terms of FDE per sequence  

            batch_size, pred_len, _ = pred_fut_traj.shape
            num_modes = preds.shape[1]
            
            # Regression loss
            
            # reg_loss = torch.zeros(1, dtype=torch.float32).to(preds)

            mse_loss = F.mse_loss(pred_fut_traj, gt_target, reduction='none')
            mse_loss = mse_loss.sum(dim=2) + self.args.epsilon # sum epsilon to avoid division by zero
            mse_loss = torch.sqrt(mse_loss)
            mse_loss = mse_loss.mean(dim=1)

            fde_loss = fde_k[index, k_hat]

            reg_loss = mse_loss * 0.5 + fde_loss * 0.5
     
            reg_loss = reg_loss.mean()
            
            # Classification loss (max-margin)

            score_hat = conf[index, k_hat].unsqueeze(-1)
            score_hat = score_hat.repeat(1, num_modes)
            cls_loss = conf + 0.2 - score_hat
            cls_loss[cls_loss < 0] = 0
            cls_loss = cls_loss.sum(dim=-1).sum(dim=-1)
            cls_loss = cls_loss /((num_modes-1) * batch_size)

            # Final loss
            
            loss = reg_loss * self.args.reg_loss_weight + \
                   cls_loss * self.args.cls_loss_weight

            return loss

    def get_lr(self, epoch):
        lr_index = 0
        for lr_epoch in self.args.lr_step_epochs:
            if epoch < lr_epoch:
                break
            lr_index += 1
        return self.args.lr_values[lr_index]
    
    def get_best_predictions(self, pred, best_pred_indeces):
        """
        pred: batch_size x num_modes x pred_len x data_dim
        best_pred_indeces: batch_size x 1
        Take the best prediction (best mode) according to the best confidence for each sequence
        """
        return pred[torch.arange(pred.shape[0]), best_pred_indeces, :, :].squeeze()
    
    def calc_prediction_metrics(self, preds, gts, conf=None):
        if self.args.freeze_decoder:
            # Calculate prediction error for each mode
            # Output has shape (batch_size, n_modes, n_timesteps)
            error_per_t = np.linalg.norm(preds - np.expand_dims(gts, axis=1), axis=-1)

            # Calculate the error for the first mode (at index 0)
            fde_1 = np.average(error_per_t[:, 0, -1])
            ade_1 = np.average(error_per_t[:, 0, :])

            # Calculate the error for all modes
            # Best mode is always the one with the lowest final displacement
            lowest_final_error_indices = np.argmin(error_per_t[:, :, -1], axis=1)
            error_per_t = error_per_t[np.arange(
                preds.shape[0]), lowest_final_error_indices]
            fde = np.average(error_per_t[:, -1])
            ade = np.average(error_per_t[:, :])
        else:
            # Calculate prediction error for each mode

            # K = 1
            # Calculate the error for the theoretically best mode (that with the highest confidence)

            best_pred_traj_indeces = conf.argmax(1)
            k1_predictions = self.get_best_predictions(preds,best_pred_traj_indeces)
            error_per_t_k1 = np.linalg.norm(k1_predictions - gts, axis=-1)
            
            fde_1 = np.average(error_per_t_k1[:, -1])
            ade_1 = np.average(error_per_t_k1[:, :])

            # K = 6
            # Calculate the error for all modes
            # Best mode is always the one with the lowest final displacement
            error_per_t = np.linalg.norm(preds - np.expand_dims(gts, axis=1), axis=-1)
            lowest_final_error_indices = np.argmin(error_per_t[:, :, -1], axis=1)
            error_per_t = error_per_t[np.arange(
                preds.shape[0]), lowest_final_error_indices]
            fde = np.average(error_per_t[:, -1])
            ade = np.average(error_per_t[:, :])

        return ade_1, fde_1, ade, fde
    
    # Overwrite Pytorch-Lightning functions
    
    def configure_optimizers(self):
        if self.args.freeze_decoder:
            if self.current_epoch == self.args.mod_freeze_epoch:
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.parameters()), weight_decay=self.args.wd) # Apply optimizer just to those parameters 
                                                                                                     # that require to be trained
            else:
                optimizer = torch.optim.AdamW(
                    self.parameters(), weight_decay=self.args.wd)
            return optimizer
        else:
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
        if self.args.freeze_decoder:
            # Trigger weight freeze and optimizer reinit on mod_freeze_epoch
            if self.current_epoch == self.args.mod_freeze_epoch:
                self.freeze()
                self.trainer.strategy.setup_optimizers(self.trainer)

            if self.current_epoch == self.args.mod_full_unfreeze_epoch:
                self.args.freeze_decoder = False
                self.full_unfreeze()
                self.trainer.strategy.setup_optimizers(self.trainer)
            
            # Set learning rate according to current epoch
            for single_param in self.optimizers().param_groups:
                single_param["lr"] = self.get_lr(self.current_epoch)
                self.log("lr", single_param["lr"], prog_bar=True, sync_dist=True)
        else:
            # Get learning rate according to current epoch
            for single_param in self.optimizers().param_groups:
                self.log("lr", single_param["lr"], prog_bar=True, sync_dist=True)
        
    def training_step(self, train_batch, batch_idx):
        out, conf = self.forward(train_batch)
        loss = self.prediction_loss(out, train_batch["gt"], conf)
        self.log("loss_train", loss, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        out, conf = self.forward(val_batch)
        loss = self.prediction_loss(out, val_batch["gt"], conf)
        self.log("loss_val", loss, sync_dist=True)

        # Extract target agent only

        pred = [x[0].detach().cpu().numpy() for x in out]
        gt = [x[0].detach().cpu().numpy() for x in val_batch["gt"]]
        if not self.args.freeze_decoder: conf = [x[0].detach().cpu().numpy() for x in conf]

        # if self.save_model_script:
        #     model_filename = os.path.join(self.args.BASE_DIR,
        #                                   self.args.MODEL_DIR,
        #                                   "TFMF_TGR.py")
        #     os.system(f"cp {model_filename} {self.args.LOG_DIR}")
        #     self.save_model_script = False
            
        return {"predictions": pred, 
                "groundtruth": gt, 
                "confidences": conf} # = validation_outputs

    def validation_epoch_end(self, validation_outputs):
        # Extract predictions
        pred = [out["predictions"] for out in validation_outputs]
        pred = np.concatenate(pred, 0) # get predictions along all validation steps
        gt = [out["groundtruth"] for out in validation_outputs]
        gt = np.concatenate(gt, 0) # get ground-truth along all validation steps
        
        if self.args.freeze_decoder:
            conf = None
        else:
            conf = [out["confidences"] for out in validation_outputs]
            conf = np.concatenate(conf, 0) # get confidences along all validation steps
            
        ade1, fde1, ade, fde = self.calc_prediction_metrics(pred, gt, conf)
        self.log("ade1_val", ade1, prog_bar=True, sync_dist=True)
        self.log("fde1_val", fde1, prog_bar=True, sync_dist=True)
        self.log("ade_val", ade, prog_bar=True, sync_dist=True)
        self.log("fde_val", fde, prog_bar=True, sync_dist=True)

# Layers

class LinearEmbedding(nn.Module):
    def __init__(self,input_size,args):
        super(LinearEmbedding, self).__init__()
        self.args = args
        self.input_size = input_size
        self.output_size = args.social_latent_size

        self.encoder_input_layer = nn.Linear(
                in_features=self.input_size, 
                out_features=self.output_size 
                    )
    def forward(self,linear_input):

        linear_out = F.relu(self.encoder_input_layer(linear_input))

        return linear_out 
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)

        return self.cached_penc

class EncoderTransformer(nn.Module):
    def __init__(self, args):
        super(EncoderTransformer, self).__init__()
        self.args = args

        self.d_model = self.args.social_latent_size # embedding dimension
        self.nhead =self.args.num_attention_heads # TODO: Is this correct?
        self.d_hid = 1 ## dimension of the feedforward network model in nn.TransformerEncoder
        self.num_layers = 1
        self.dropout = self.args.apply_dropout

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid , self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, transformer_in, agents_per_sample):

        transformer_out = F.relu(self.transformer_encoder(transformer_in))
        return transformer_out[:,-1,:]
      
class AgentGNN(nn.Module):
    def __init__(self, args):
        super(AgentGNN, self).__init__()
        self.args = args
        self.latent_size = args.social_latent_size

        self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
        self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)

    def forward(self, gnn_in, centers, agents_per_sample):
        # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

        x, edge_index = gnn_in, self.build_fully_connected_edge_idx(
            agents_per_sample).to(gnn_in.device)
        edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr))
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

        edge_index_out1 = []
        for i in agents_per_sample:
            edge_index_out1.append(gnn_out[0:i,:])
            gnn_out = gnn_out[i:,:]

        return edge_index_out1

    def build_fully_connected_edge_idx(self, agents_per_sample):
        edge_index = []

        # In the for loop one subgraph is built (no self edges!)
        # The subgraph gets offsetted and the full graph over all samples in the batch
        # gets appended with the offsetted subgrah
        offset = 0
        for i in range(len(agents_per_sample)):

            num_nodes = agents_per_sample[i]

            adj_matrix = torch.ones((num_nodes, num_nodes))
            adj_matrix = adj_matrix.fill_diagonal_(0)

            sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())
            edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)

            # Offset the list
            edge_index_subgraph = torch.Tensor(
                np.asarray(edge_index_subgraph) + offset)
            offset += agents_per_sample[i]

            edge_index.append(edge_index_subgraph)

        # Concat the single subgraphs into one
        edge_index = torch.LongTensor(np.column_stack(edge_index))
        
        return edge_index
    def build_edge_attr(self, edge_index, data):
        edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)

        rows, cols = edge_index
        # goal - origin
        edge_attr = data[cols] - data[rows]

        return edge_attr

class DecoderResidual(nn.Module):
    def __init__(self, args):
        super(DecoderResidual, self).__init__()

        self.args = args
        self.latent_size = self.args.decoder_latent_size
        self.num_modes = self.args.num_modes
        
        output = []
        for i in range(sum(args.mod_steps)):
            output.append(PredictionNet(args))

        self.output = nn.ModuleList(output) # is just like a Python list. It was designed to store any desired number of nn.Module’s

        if not self.args.freeze_decoder or self.args.mod_full_unfreeze_epoch != -1:
            # Classification
            
            norm = "BN"
            ng = 1

            self.latent_predictions = nn.Linear(self.args.num_modes * self.args.pred_len * self.args.data_dim,
                                                self.latent_size)
            self.confidences = nn.Sequential(LinearRes(self.latent_size*2, self.latent_size*2, norm=norm, ng=ng), 
                                             nn.Linear(self.latent_size*2, self.num_modes))
        
    def forward(self, decoder_in, is_frozen, current_epoch):
        batch_size = decoder_in.shape[0]
        
        if self.args.freeze_decoder:
            sample_wise_out = []

            if self.training is False: # If you are validating or test, use all decoders
                for out_subnet in self.output:
                    sample_wise_out.append(out_subnet(decoder_in))
            elif is_frozen: # If the first decoder has been frozen, decode and train the remaining ones
                for i in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
                    sample_wise_out.append(self.output[i](decoder_in))
            else: # If you are training and is_frozen = False, use only the first decoder
                sample_wise_out.append(self.output[0](decoder_in))
            
            decoder_out = torch.stack(sample_wise_out)
            decoder_out = torch.swapaxes(decoder_out, 0, 1)
          
            return decoder_out, [] 
        else:
            sample_wise_out = []

            for out_subnet in self.output:
                sample_wise_out.append(out_subnet(decoder_in))
               
            decoder_out = torch.stack(sample_wise_out)
            decoder_out = torch.swapaxes(decoder_out, 0, 1)
            pdb.set_trace()
            latent_predictions = self.latent_predictions(decoder_out.contiguous().view(batch_size,-1))
            conf_latent = torch.cat([decoder_in,
                                     latent_predictions],
                                     dim=1)

            conf = self.confidences(conf_latent)
            conf = torch.softmax(conf.view(batch_size,-1), dim=1) # batch_size, num_modes
            if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
                pdb.set_trace()
                
            return decoder_out, conf

    def unfreeze_layers(self):
        for layer in range(self.args.mod_steps[0], sum(self.args.mod_steps)): # Unfreeze all decoders except the first one
            for param in self.output[layer].parameters():
                param.requires_grad = True

class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_out, n_out)
        self.linear3 = nn.Linear(n_out, n_out)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
            self.norm3 = nn.BatchNorm1d(n_out)
        else:   
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.norm3(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out
    
class PredictionNet(nn.Module):
    def __init__(self, args):
        super(PredictionNet, self).__init__()

        self.args = args

        self.latent_size = args.decoder_latent_size

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.GroupNorm(1, self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.GroupNorm(1, self.latent_size) # Batch normalization solves a major problem called internal covariate shift. 

        self.output_fc = nn.Linear(self.latent_size, args.pred_len * 2)
        
    def forward(self, prednet_in):
        # Residual layer
        x = self.weight1(prednet_in)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.weight2(x)
        x = self.norm2(x)

        x += prednet_in

        x = F.relu(x)

        # Last layer has no activation function
        prednet_out = self.output_fc(x)

        return prednet_out
    
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
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(output_size // 2, output_size)
        
        # self.linear1 = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.ReLU(x)
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
    
class Temporal_Multimodal_Decoder(nn.Module):
    def __init__(self, args):
        super(Temporal_Multimodal_Decoder, self).__init__()

        self.args = args
        self.data_dim = self.args.data_dim
        self.obs_len = self.args.obs_len
        self.pred_len = self.args.pred_len
        self.window_size = self.args.decoder_temporal_window_size

        self.decoder_h_dim = self.args.decoder_latent_size
        self.num_modes = self.args.num_modes

        self.spatial_embedding = nn.Linear(self.window_size*2, self.window_size*4)
        
        self.decoder = nn.LSTM(self.window_size*4, 
                               self.decoder_h_dim, 
                               num_layers=1)
  
        pred = []
        for _ in range(self.num_modes):
            pred.append(nn.Linear(self.decoder_h_dim,self.data_dim))
        self.hidden2pos = nn.ModuleList(pred) 

        norm = "BN"
        ng = 1
        
        # Confidences
        
        self.latent_predictions = nn.Linear(self.args.num_modes*self.args.pred_len*self.args.data_dim,
                                            self.decoder_h_dim)
        self.confidences = nn.Sequential(LinearRes(self.decoder_h_dim*2, self.decoder_h_dim*2, norm=norm, ng=ng), 
                                         nn.Linear(self.decoder_h_dim*2, self.num_modes))
        
    def forward(self, traj_rel, state_tuple, num_mode=None, current_centerlines=None):
        """_summary_

        Args:
            traj_rel (_type_): _description_
            state_tuple (_type_): _description_
            num_mode (_type_, optional): _description_. Defaults to None.
            current_centerlines (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        traj_rel = traj_rel.permute(1,0,2)
        num_displacements, batch_size, data_dim = traj_rel.shape
        state_tuple_h, state_tuple_c = state_tuple
        
        pred_traj_fake_rel = []
    
        for num_mode in range(self.num_modes):
            traj_rel_ = torch.clone(traj_rel)
            decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel_.permute(1,0,2).contiguous().view(batch_size,-1))) # bs x window_size·2
        
            decoder_input = decoder_input.unsqueeze(0)
            decoder_input = F.dropout(decoder_input, p=self.args.apply_dropout, training=self.training)
        
            state_tuple_h_ = torch.clone(state_tuple_h)
            state_tuple_c_ = torch.zeros(tuple(state_tuple_h_.shape)).to(state_tuple_h_)
            
            curr_pred_traj_fake_rel = []
            for _ in range(self.pred_len):
                output, (state_tuple_h_, state_tuple_c_) = self.decoder(decoder_input, (state_tuple_h_, state_tuple_c_)) 
                rel_pos = self.hidden2pos[num_mode](output.contiguous().view(-1, self.decoder_h_dim))

                traj_rel_ = torch.roll(traj_rel_, -1, dims=(0))
                traj_rel_[-1] = rel_pos

                curr_pred_traj_fake_rel.append(rel_pos)

                decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel_.permute(1,0,2).contiguous().view(batch_size,-1))) # bs x window_size·2
                decoder_input = decoder_input.unsqueeze(0)
                decoder_input = F.dropout(decoder_input, p=self.args.apply_dropout, training=self.training)
        
            curr_pred_traj_fake_rel = torch.stack(curr_pred_traj_fake_rel,dim=0)
            curr_pred_traj_fake_rel = curr_pred_traj_fake_rel.permute(1,0,2)
            pred_traj_fake_rel.append(curr_pred_traj_fake_rel)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0) # num_modes, batch_size, pred_len, data_dim
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,0,2,3) # batch_size, num_modes, pred_len, data_dim

        # Obtain confidences based on the initial latent state and the predictions

        predictions_latent = self.latent_predictions(pred_traj_fake_rel.contiguous().view(batch_size, -1))
        state_tuple_h = state_tuple_h.squeeze(0)
        conf_latent = torch.cat([state_tuple_h,
                                 predictions_latent],
                                 dim=1)

        conf = self.confidences(conf_latent)
        conf = torch.softmax(conf.view(batch_size,-1), dim=1) # batch_size, num_modes
        if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
            pdb.set_trace()
        
        return pred_traj_fake_rel, conf

# Aux functions
    
def relative_to_abs_multimodal(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (batch_size, num_modes, seq_len, 2)
    - start_pos: pytorch tensor of shape (batch_size, 2)
      N.B. If you only have the predictions, this must be the last observation.
           If you have the whole trajectory (obs+pred), this must be the first observation,
           since you must reconstruct the relative displacements from this position 
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2) (around 0,0, not map coordinates)
    """

    displacement = torch.cumsum(rel_traj, dim=2) # Sum along the seq_len dimension!
    start_pos = torch.unsqueeze(torch.unsqueeze(start_pos, dim=1), dim=1) # batch, 1 (only one position) x 1 (same for all modes) x 2
    abs_traj = displacement + start_pos

    return abs_traj

############################ GANET

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
    def __init__(self, config):
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 32, 64, 128]
        blocks = [no_pad_Res1d, Res1d, Res1d, Res1d]
        num_blocks = [1, 2, 2, 2]

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

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)
        
        ctrs_in = 2
        self.lstm_h0_init_function = nn.Linear(ctrs_in, n, bias=False)
        self.lstm_encoder = nn.LSTM(n, n, batch_first=True)
        
        self.output = Res1d(n, n, norm=norm, ng=ng)
        
    def forward(self, actors: Tensor, actor_ctrs) -> Tensor:
        actor_ctrs = torch.cat(actor_ctrs, 0)
        out = actors
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
        h0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, config["n_actor"])
        c0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, config["n_actor"])
        #h0 = torch.zeros(1, M, config["n_actor"]).cuda()
        #c0 = torch.zeros(1, M, config["n_actor"]).cuda()
        out = out.transpose(1, 2).contiguous()
        output, (hn, cn) = self.lstm_encoder(out, (h0, c0))
        out_lstm = hn.contiguous().view(M, config["n_actor"])
        out = out_lstm + out_init
        return out

class TestNet_Half(nn.Module):
    """
    Final position prediction with Linear Residual block
    """
    def __init__(self, config):
        super(TestNet_Half, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1
        n_actor = config["n_actor"]
        self.test = nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2))

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        reg = self.test(actors)
        reg = reg.view(-1, 2)
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 2)
            reg[idcs] = reg[idcs] + ctrs
        test_ctrs = []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            test_ctrs.append(reg[idcs])
        return test_ctrs

class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    For the student (without map)
    """
    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = myAttDestNoMap(n_actor, config["num_mods"])
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
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
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs, actor_idcs) 
        cls = self.cls(feats).view(-1, self.config["num_mods"])
        cls = self.softmax(cls)
        
        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"], out["test_ctrs"], out["test_cls"] = [], [], [], []
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
        #agts = self.K_agt(k_actors)
        return k_actors

class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
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
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out

class LossLaneGCN(nn.Module):
    def __init__(self, config):
        super(LossLaneGCN, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) \
                         + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out 
    
class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

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
                self.config["actor2actor_dist"],
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

############################ GANET