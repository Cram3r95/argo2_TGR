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

from typing import List, Dict, Union, Tuple

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

from torch import nn, Tensor, optim
from torch.nn import functional as F
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix

# Plot imports

# Custom imports

from model.models.layers import Conv1d, Res1d, Linear, LinearRes, Null, no_pad_Res1d, CosineAnnealingWarmupRestarts

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
        
        # Encoder
        
        ## Social
        
        self.linear_embedding = LinearEmbedding(self.args)
        self.pos_encoder = PositionalEncoding1D(self.args.social_latent_size)
        self.encoder_transformer = EncoderTransformer(self.args)
        
        ## Physical 
        
        if self.args.use_map:
            
            self.map_sub_net = BoundariesNet(self.args)
            
            assert self.args.social_latent_size == self.args.map_latent_size
            
            if self.args.final_latent_info == "concat":
                self.args.decoder_latent_size = self.args.social_latent_size + self.args.map_latent_size
            elif self.args.final_latent_info == "fuse":
                self.A2L_1 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
                self.L2A_1 = TransformerDecoder(self.args.social_latent_size, head_num=self.args.num_attention_heads)
                
                self.args.decoder_latent_size = self.args.social_latent_size
            else:
                raise AssertionError
        else:
            self.args.decoder_latent_size = self.args.social_latent_size

        ## Local interaction
        
        self.a2a = A2A(self.args)
        
        ## Global interaction
        
        self.agent_gnn = AgentGNN(self.args)
        
        # Decoder
   
        self.decoder = PredNet(self.args)

        # Metrics

        self.validation_type = "single-agent" # single-agent = only focal track (category 3)
                                              # multi-agent = both focal track and scored tracks (category 2 & 3)
        self.loss_lane = LossLane(self.args)
        self.save_model_script = True

    @staticmethod
    def init_args(parent_parser, BASE_DIR, DATASET_DIR):
        parser_dataset = parent_parser.add_argument_group("dataset")
        parser_dataset.add_argument(
            "--BASE_DIR", type=str, default=BASE_DIR)
        parser_dataset.add_argument(
            "--DATASET_DIR", type=str, default=DATASET_DIR)
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
            "--train_split_pre_social", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_social", "train_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre_social", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_social", "val_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre_social", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_social", "test_pre_clean.pkl"))
        
        # Map preprocess
        
        parser_dataset.add_argument(
            "--train_split_pre_map", type=str, default=os.path.join(
            BASE_DIR, DATASET_DIR, "processed_map", "train_map_data_rot_right_x_multi_agent.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre_map", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_map", "val_map_data_rot_right_x_multi_agent.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre_map", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_map", "test_map_data_rot_right_x_multi_agent.pkl"))
        
        # Whole preprocess
        
        parser_dataset.add_argument(
            "--train_split_pre", type=str, default=os.path.join(
            BASE_DIR, DATASET_DIR, "processed_full", "train_full_data_rot_right_x_multi_agent.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_full", "val_full_data_rot_right_x_multi_agent.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre", type=str, default=os.path.join(
                BASE_DIR, DATASET_DIR, "processed_full", "test_full_data_rot_right_x_multi_agent.pkl"))
        
        parser_dataset.add_argument("--reduce_dataset_size", type=int, default=0)
        parser_dataset.add_argument("--use_preprocessed", type=bool, default=False)
        parser_dataset.add_argument("--use_map", type=bool, default=False)
        parser_dataset.add_argument("--align_image_with_target_x", type=bool, default=True)

        parser_training = parent_parser.add_argument_group("training")
        parser_training.add_argument("--num_epochs", type=int, default=100)
        parser_training.add_argument("--check_val_every_n_epoch", type=int, default=10)
        parser_training.add_argument("--lr_values", type=list, default=[1e-3, 1e-4, 1e-3 , 1e-4])
        parser_training.add_argument("--lr_step_epochs", type=list, default=[10, 20, 45])
        parser_training.add_argument("--initial_lr", type=float, default=1e-3)
        parser_training.add_argument("--scheduler_reduce_factor", type=float, default=0.1)
        parser_training.add_argument("--scheduler_patience", type=float, default=5)
        parser_training.add_argument("--min_lr", type=float, default=5e-5) 
        parser_training.add_argument("--wd", type=float, default=0.001)
        parser_training.add_argument("--batch_size", type=int, default=64)
        parser_training.add_argument("--val_batch_size", type=int, default=64)
        parser_training.add_argument("--workers", type=int, default=0) # TODO: Not working with >= 0
        parser_training.add_argument("--val_workers", type=int, default=0)
        parser_training.add_argument("--gpus", type=int, default=1)

        parser_model = parent_parser.add_argument_group("model")
        parser_dataset.add_argument("--MODEL_DIR", type=str, default="non_specified")
        parser_model.add_argument("--data_dim", type=int, default=2)
        # dispX (1), dispY (1), heading (1), object_type (3), object_category (2), mask (1) -> 9
        parser_model.add_argument("--num_social_features", type=int, default=9)
        parser_model.add_argument("--obs_len", type=int, default=50)
        parser_model.add_argument("--pred_len", type=int, default=60)
        parser_model.add_argument("--centerline_length", type=int, default=40)
        parser_model.add_argument("--num_centerlines", type=int, default=6)
        parser_model.add_argument("--num_attention_heads", type=int, default=8)
        parser_model.add_argument("--apply_dropout", type=float, default=0.25)
        parser_model.add_argument("--data_aug_gaussian_noise", type=float, default=0.01)
        parser_model.add_argument("--social_latent_size", type=int, default=128)
        parser_model.add_argument("--map_latent_size", type=int, default=128)
        parser_model.add_argument("--final_latent_info", type=str, default="non_specified")
        parser_model.add_argument("--decoder_latent_size", type=int, default=-1)
        parser_model.add_argument("--decoder_temporal_window_size", type=int, default=30) # 49 
        parser_model.add_argument("--num_modes", type=int, default=6)
        parser_model.add_argument("--freeze_decoder", type=bool, default=False)
        parser_model.add_argument("--mod_steps", type=list, default=[1, 5]) # First unimodal -> Freeze -> Multimodal
        parser_model.add_argument("--mod_freeze_epoch", type=int, default=20)
        parser_model.add_argument("--mod_full_unfreeze_epoch", type=int, default=60)
        parser_model.add_argument("--reg_loss_weight", type=float, default=1) # xy predictions
        parser_model.add_argument("--cls_loss_weight", type=float, default=1) # classification = confidences
        parser_model.add_argument("--epsilon", type=float, default=0.0000000001)
        parser_model.add_argument("--mgn", type=float, default=0.2) # ?
        parser_model.add_argument("--cls_th", type=float, default=2.0) # ?
        parser_model.add_argument("--cls_ignore", type=float, default=0.2) # ?
        parser_model.add_argument("--cls_coef", type=float, default=3.0) # ?
        parser_model.add_argument("--reg_coef", type=float, default=1.0) # ?
  
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
        # batch
        #   |
        #   v
        # 'argo_id', 'city', 'track_id', 'type', 'category', 'past_trajs', 'fut_trajs', 'gt', 'displ', 'centers', 'headings', 'origin', 
        # 'rotation', 'rel_centerline', 'centerline_type', 'is_intersection', 'rel_left_bound', 'left_type', 'rel_right_bound', 'right_type'
        
        # Set batch norm to eval mode in order to prevent updates on the running means,
        # if the weights are frozen

        # Encoder

        ## Social
        
        ### Extract the social features in each sample of the current batch

        idcs = get_actor_ids(gpu(batch["displ"]))
        displ_and_mask, centers, headings = batch["displ"], gpu(batch["centers"]), batch["headings"]
        object_types, object_categories = batch["type"], batch["category"]
        rotation, origin = batch["rotation"], batch["origin"]
        
        agents_per_sample = [x.shape[0] for x in displ_and_mask]
        batch_size = len(agents_per_sample)
        
        ### OBS: For each sequence, we always set the focal (target) agent as the first agent
        ###      of the scene, then our ego-vehicle (AV) and finally the remanining agents
        ###      (See extractor_proc.py preprocessing)
        
        focal_agent_id = np.cumsum(agents_per_sample)
        focal_agent_id = np.roll(focal_agent_id,1)
        focal_agent_id[0] = 0
        
        ### Convert the list of tensors to tensors
        
        displ_and_mask_cat = torch.cat(displ_and_mask, dim=0)
        centers_cat = torch.cat(centers, dim=0)
        headings = torch.cat(headings, dim=0)
        object_types = torch.cat(object_types, dim=0)
        object_categories = torch.cat(object_categories, dim=0)

        #### Include object type, category and orientation in the tensor

        actor_raw_features = torch.zeros((displ_and_mask_cat.shape[0], # number of agents
                                          displ_and_mask_cat.shape[1], # relative displacements
                                          self.args.num_social_features)).to(displ_and_mask_cat)

        actor_raw_features[:,:,:2] = displ_and_mask_cat[:,:,:2] # dispX and dispY
        actor_raw_features[:,:,2] = headings[:,1:self.args.obs_len,:].squeeze(2) # heading (start from 1 and end in obs_len frame to have obs_len - 1 values)
        actor_raw_features[:,:,3:6] = torch.tile(object_types[:,:].unsqueeze(1),(self.args.obs_len-1,1)) # object types
        actor_raw_features[:,:,6:8] = torch.tile(object_categories[:,:].unsqueeze(1),(self.args.obs_len-1,1)) # object categories
        actor_raw_features[:,:,8] = displ_and_mask_cat[:,:,2] # Mask
        
        ### Data augmentation (TODO: It should be in the collate_fn_dict function, in the DataLoader)

        # if self.training:
        #     actor_raw_features[:,:,:2] = self.add_noise(actor_raw_features[:,:,:2], self.args.data_aug_gaussian_noise)
        #     centers_cat = self.add_noise(centers_cat, self.args.data_aug_gaussian_noise)
        
        linear_output = self.linear_embedding(actor_raw_features)
        pos_encoding = self.pos_encoder(linear_output)
        pos_encoding = pos_encoding + linear_output

        social_info = self.encoder_transformer(pos_encoding, agents_per_sample) # Deep social features
        
        # actor_raw_features = displ_and_mask_cat.transpose(1,2)
        # actor_raw_features = actor_raw_features.transpose(1,2)
        # actors_1 = self.actor_net1(actor_raw_features, centers)
        # actors_2 = self.actor_net2(actor_raw_features, centers)
        # actors = actors_1 + actors_2
        # decoder_features = self.a2a(actors, idcs, centers)
        
        # ## Physical (map)
        
        ## Features fusion 
        
        if self.args.use_map:
            ### Extract map features

            lane_features = torch.cat([torch.cat(batch["centerline_type"],0),
                                    torch.cat(batch["is_intersection"],0)],-1)
            
            left_boundary_features = torch.cat([torch.cat(batch["rel_left_bound"],0),
                                                torch.cat(batch["left_type"],0)],-1)
            
            right_boundary_features = torch.cat([torch.cat(batch["rel_left_bound"],0),
                                                torch.cat(batch["left_type"],0)],-1)

            ### Encode map features
            
            physical_info = self.map_sub_net(lane_features, left_boundary_features, right_boundary_features)
            
        if self.args.use_map:
            if self.args.final_latent_info == "concat": # Concat info
                agents_features = torch.cat([social_info, 
                                             physical_info], 
                                             dim=1)
                    
            if self.args.final_latent_info == "fuse": # Fuse info
                physical_info = physical_info + self.A2L_1(physical_info, social_info)
                social_info = social_info + self.L2A_1(social_info, physical_info)
                
                # physical_info = physical_info + self.A2L_2(physical_info, social_info)
                # social_info = social_info + self.L2A_2(social_info, physical_info)
                
                agents_features = social_info
        else:
            agents_features = social_info

        agents_features = self.a2a(agents_features, idcs, centers) # Local interaction
        decoder_features = self.agent_gnn(agents_features, centers_cat, agents_per_sample) # Global interaction
        # pdb.set_trace()
        # decoder_features = torch.cat(decoder_features,0) # Concatenate all relevant agents
        
        # Decoder

        out = self.decoder(decoder_features, idcs, centers)
        
        ## Iterate over each batch and transform predictions into the global coordinate frame
        
        rot, orig = gpu(batch["rotation"]), gpu(batch["origin"])
        
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        
        out["focal_agent_id"] = focal_agent_id # Store focal agent id (only for single-agent validation)

        return out

    # Overwrite Pytorch-Lightning functions
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                        weight_decay=self.args.wd,
                                        lr=self.args.initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                               T_0=self.args.iterations_per_epoch * 2, 
                                                               eta_min=self.args.min_lr, 
                                                               last_epoch=-1)

        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "metric_to_track",
                                 "interval": "step",
                                 "frequency": 1}}

    # def on_train_epoch_start(self):
    #     """
    #     Get learning rate according to current epoch
    #     """

    #     for single_param in self.optimizers().param_groups:
    #         self.log("lr", single_param["lr"], prog_bar=True, sync_dist=True)
        
    def training_step(self, train_batch, batch_idx):
        """_summary_

        Args:
            train_batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """

        for single_param in self.optimizers().param_groups:
            self.log("lr", single_param["lr"], prog_bar=True, sync_dist=True)
            
        out = self.forward(train_batch)
        loss = self.loss_lane(out, train_batch)

        self.log("loss_train", loss["loss"], sync_dist=True)

        return loss["loss"]

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch)
        loss = self.loss_lane(out, val_batch, validation_type=self.validation_type)
            
        self.log("loss_val", loss["loss"], sync_dist=True)

        # Transform to numpy 
        
        pred = torch.cat(out["reg"],0)
        pred = pred.detach().cpu().numpy()

        conf = torch.cat(out["cls"],0)
        conf = conf.detach().cpu().numpy()
        
        gt = torch.cat([x for x in val_batch["gt"]])
        gt = gt.detach().cpu().numpy()
        has_preds = torch.cat([x for x in val_batch["has_preds"]]) # Do not transform to np here

        if self.save_model_script:
            model_filename = os.path.join(self.args.BASE_DIR,
                                          self.args.MODEL_DIR,
                                          os.path.basename(__file__))
            os.system(f"cp {model_filename} {self.args.LOG_DIR}")
            self.save_model_script = False
            
        return {"predictions": pred, 
                "confidences": conf,
                "groundtruth": gt,
                "has_preds": has_preds} # = validation_outputs
                                        #          |
                                        #          v
    def validation_epoch_end(self, validation_outputs):
        # Extract predictions

        if self.validation_type == "single-agent":
            pred = [out["predictions"][0] for out in validation_outputs] # Only take the first agent (0) of each scenario
            pred = np.stack(pred, 0) 
            conf = [out["confidences"][0] for out in validation_outputs]
            conf = np.stack(conf, 0)
            
            gt = [out["groundtruth"][0] for out in validation_outputs]
            gt = np.stack(gt, 0) 
            has_preds = [out["has_preds"][0] for out in validation_outputs]
            has_preds = torch.stack(has_preds, 0) 
        else:
            pred = [out["predictions"] for out in validation_outputs]
            pred = np.concatenate(pred, 0) 
            conf = [out["confidences"] for out in validation_outputs]
            conf = np.concatenate(conf, 0)
            
            gt = [out["groundtruth"] for out in validation_outputs]
            gt = np.concatenate(gt, 0) 
            has_preds = [out["has_preds"] for out in validation_outputs]
            has_preds = torch.concatenate(has_preds, 0)     

        ade1, fde1, ade, fde, brier_fde, min_idcs = pred_metrics(pred, conf, gt, has_preds)

        print("ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f, brier_fde %2.4f" % (ade1, fde1, ade, fde, brier_fde))
        print()
        
        self.log("ade1_val", ade1, prog_bar=True, sync_dist=True)
        self.log("fde1_val", fde1, prog_bar=True, sync_dist=True)
        self.log("ade_val", ade, prog_bar=True, sync_dist=True)
        self.log("fde_val", fde, prog_bar=True, sync_dist=True)
        self.log("brier_fde_val", brier_fde, prog_bar=True, sync_dist=True)
        
# Layers

class LinearEmbedding(nn.Module):
    def __init__(self,args):
        super(LinearEmbedding, self).__init__()
        self.args = args
        self.input_size = args.num_social_features
        self.output_size = args.social_latent_size
        self.dropout = self.args.apply_dropout

        self.encoder_input_layer = nn.Linear(
                in_features=self.input_size, 
                out_features=self.output_size 
                    )
        ng = 1
        self.norm = nn.GroupNorm(gcd(ng, self.output_size), self.output_size)
        self.dropout = nn.Dropout(self.dropout)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        
    def forward(self,linear_input):

        # linear_out = F.relu(self.encoder_input_layer(linear_input))
        x = self.encoder_input_layer(linear_input)
        linear_out = self.dropout(self.LeakyReLU(self.norm(x.transpose(1,2))))
        linear_out = linear_out.transpose(1,2)
        
        return linear_out 
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, latent_dim):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        channels = latent_dim
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
        # self.nhead = self.args.num_attention_heads # TODO: Is this correct?
        self.nhead = self.args.social_latent_size
        self.d_hid = 1 ## dimension of the feedforward network model in nn.TransformerEncoder
        self.num_layers = 2
        self.dropout = self.args.apply_dropout

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid , self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        
    def forward(self, transformer_in, agents_per_sample):

        transformer_out = self.LeakyReLU(self.transformer_encoder(transformer_in))
        return transformer_out[:,-1,:]

class AgentGNN(nn.Module):
    def __init__(self, args):
        super(AgentGNN, self).__init__()
        self.args = args
        # self.latent_size = args.social_latent_size # Message-Passing only included social info
        self.latent_size = args.decoder_latent_size # Message-Passing including social and map info
 
        self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
        self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

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
    
    # def forward(self, gnn_in, centers, agents_per_sample):
    #     # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

    #     x, edge_index = gnn_in, self.build_fully_connected_edge_idx(
    #         agents_per_sample).to(gnn_in.device)
    #     edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

    #     x = F.relu(self.gcn1(x, edge_index, edge_attr))
    #     gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

    #     edge_index_out1 = []
    #     for i in agents_per_sample:
    #         edge_index_out1.append(gnn_out[0:i,:])
    #         gnn_out = gnn_out[i:,:]

    #     return edge_index_out1
    
    def forward(self, gnn_in, centers, agents_per_sample):
        """_summary_
        Args:
            gnn_in (_type_): _description_
            centers (_type_): _description_
            agents_per_sample (_type_): _description_
        Returns:
            _type_: _description_
        """

        x, edge_index = gnn_in, self.build_fully_connected_edge_idx(agents_per_sample).to(gnn_in.device)
        edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

        x = self.LeakyReLU(self.gcn1(x, edge_index, edge_attr)) 
        gnn_out = self.LeakyReLU(self.gcn2(x, edge_index, edge_attr)) 

        return gnn_out
    
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

        self.latent_size = self.args.decoder_latent_size
        self.num_modes = self.args.num_modes
        self.pred_len = self.args.pred_len
        
        pred = []
        for i in range(self.num_modes):
            pred.append(
                nn.Sequential(
                    LinearRes(self.latent_size, self.latent_size, norm=norm, ng=ng),
                    nn.Linear(self.latent_size, 2 * self.pred_len),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = myAttDest(self.latent_size, self.num_modes)
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
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs, actor_idcs) 
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
        return out
    
class myAttDest(nn.Module):
    def __init__(self, n_agt: int, K):
        super(myAttDest, self).__init__()
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

# Loss and metrics functions

class LossLane(nn.Module):
    def __init__(self, args):
        super(LossLane, self).__init__()
        self.args = args
        self.pred_loss = PredLoss(self.args)

    def forward(self, out: Dict, data: Dict, validation_type: str = None) -> Dict:

        # Add here 'has_preds' -> A mask to know if the object was observed in that frame

        future_trajs = gpu(data["fut_trajs"])
        has_preds = [future_trajs_[:,:,-1].bool() for future_trajs_ in future_trajs]
        data["has_preds"] = has_preds

        loss_out = self.pred_loss(out, gpu(data["gt"]), gpu(data["has_preds"]), 
                                  validation_type=validation_type)

        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + self.args.epsilon) \
                         + loss_out["reg_loss"] / (loss_out["num_reg"] + self.args.epsilon)
        return loss_out 

class PredLoss(nn.Module):
    def __init__(self, args):
        super(PredLoss, self).__init__()
        self.args = args
        
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")
   
    def forward(self, out: Dict[str, List[Tensor]], 
                      gt_preds_: List[Tensor], 
                      has_preds_: List[Tensor],
                      validation_type: str = None)\
    -> Dict[str, Union[Tensor, int]]:
        """_summary_

        Returns:
            _type_: _description_
        """
        # TODO: Check this function
        
        if validation_type == "single-agent": # Only single-agent validation (focal track)
            reg = torch.stack([x[0] for x in out["reg"]], 0)
            cls = torch.stack([x[0] for x in out["cls"]], 0)
            
            gt_preds = torch.cat([x for x in gt_preds_], 0)
            gt_preds = gt_preds[out["focal_agent_id"],:,:]
            has_preds = torch.cat([x for x in has_preds_], 0)
            has_preds = has_preds[out["focal_agent_id"],:]
        elif validation_type == "multi-agent": # Multi-agent validation (focal-track + scored tracks) 
            # TODO: Not used at this moment, you should obtain only those objects with category 2 & 3
            pdb.set_trace()
        else: # train (prediction of all agents -> category 1, 2 and 3)
            reg = torch.cat([x for x in out["reg"]], 0)
            cls = torch.cat([x for x in out["cls"]], 0)
            
            gt_preds = torch.cat([x for x in gt_preds_], 0)
            has_preds = torch.cat([x for x in has_preds_], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.args.num_modes, self.args.pred_len

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
        coef = self.args.cls_coef
        loss_out["cls_loss"] += coef * (
            self.args.mgn * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.args.reg_coef
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()

        return loss_out
     
def pred_metrics(preds, preds_cls, gt_preds, has_preds):
    """
    """
 
    preds = np.asarray(preds, np.float32)
    cls = np.asarray(preds_cls, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    has_preds = has_preds.detach().cpu()

    num_agents, num_mods, num_preds, data_dim = preds.shape

    last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(has_preds.device) / float(num_preds)
    max_last, last_idcs = last.max(1)
    
    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))
    
    row_idcs_last = np.arange(len(last_idcs)).astype(np.int64) 
    ade1 =  np.asarray([err[i, 0, :last_idcs[i]].mean() for i in range(num_agents)]).mean()
    fde1 = err[row_idcs_last, 0, last_idcs].mean()
    
    #cls = softmax(cls, axis=1)
    min_idcs = err[row_idcs_last, :, last_idcs].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    cls = cls[row_idcs, min_idcs]
    
    ade = np.asarray([err[i, :last_idcs[i]].mean() for i in range(num_agents)]).mean()
    fde = err[row_idcs_last, last_idcs].mean()
    one_arr = np.ones(num_agents)
    brier_fde = (err[row_idcs_last, last_idcs] + (one_arr-cls)**2).mean()

    return ade1, fde1, ade, fde, brier_fde, min_idcs

# Aux functions

def gpu(data):
    """
    Transfer tensor in "data" to gpu recursively
    "data" can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data

def get_actor_ids(actors: List[Tensor]) -> List[Tensor]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors[0].device)
        actor_idcs.append(idcs)
        count += num_actors[i]

    return actor_idcs

class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, args):
        super(ActorNet, self).__init__()
        self.args = args
        norm = "GN"
        ng = 1

        n_in = self.args.num_social_features
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

        n = self.args.social_latent_size
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
        h0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, self.args.social_latent_size)
        c0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, self.args.social_latent_size)
        #h0 = torch.zeros(1, M, self.args.social_latent_size).cuda()
        #c0 = torch.zeros(1, M, self.args.social_latent_size).cuda()
        out = out.transpose(1, 2).contiguous()
        output, (hn, cn) = self.lstm_encoder(out, (h0, c0))
        out_lstm = hn.contiguous().view(M, self.args.social_latent_size)
        out = out_lstm + out_init

        return out
    
class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self, args):
        super(A2A, self).__init__()
        self.args = args
        self.args.actor2actor_dist = 100.0

        n_actor = self.args.decoder_latent_size

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
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float) -> Tensor:
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.LeakyReLU(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.LeakyReLU(agts)
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
        agts = self.LeakyReLU(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.LeakyReLU(agts)
        return agts
    
class BoundariesNet(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, args):
        super(BoundariesNet, self).__init__()
        self.args = args
        
        latent_dim = 32
        ng = 1
        
        self.linear_lane = nn.Linear(3,latent_dim,bias=False)
        self.norm_lane = nn.GroupNorm(gcd(ng, latent_dim), latent_dim)
        self.dropout_lane = nn.Dropout(self.args.apply_dropout)
        
        self.linear_left_boundary = nn.Linear(6,latent_dim,bias=False)
        self.norm_left_boundary = nn.GroupNorm(gcd(ng, latent_dim), latent_dim)
        self.dropout_left_boundary = nn.Dropout(self.args.apply_dropout)
        
        self.linear_right_boundary = nn.Linear(6,latent_dim,bias=False)
        self.norm_right_boundary = nn.GroupNorm(gcd(ng, latent_dim), latent_dim)
        self.dropout_right_boundary = nn.Dropout(self.args.apply_dropout)
        
        self.pos_encoder = PositionalEncoding1D(latent_dim)
        
        self.cross_attn_left = nn.MultiheadAttention(latent_dim, 8, self.args.apply_dropout)
        self.cross_attn_right = nn.MultiheadAttention(latent_dim, 8, self.args.apply_dropout)
        
        dim = 3*40*2*latent_dim
        self.linear1 = nn.Linear(dim,self.args.map_latent_size)
        self.norm1 = nn.GroupNorm(gcd(ng, self.args.map_latent_size), self.args.map_latent_size)
        self.dropout1 = nn.Dropout(self.args.apply_dropout)
        
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        
    def forward(self, lane_features, left_boundary_features, right_boundary_features):
        """_summary_

        Args:
            lanes (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        num_agents,num_lanes,num_points,lane_features_dim = lane_features.shape
        _,_,_,lane_boundaries_features_dim = left_boundary_features.shape

        lane_features = self.linear_lane(lane_features.view(-1,num_points,lane_features_dim))
        lane_features = self.norm_lane(lane_features.transpose(1,2))
        lane_features = self.LeakyReLU(lane_features.transpose(1,2))
        lane_features = self.dropout_lane(lane_features)
        lane_features = self.pos_encoder(lane_features)

        left_boundary_features = self.linear_left_boundary(left_boundary_features.view(-1,num_points,lane_boundaries_features_dim))
        left_boundary_features = self.norm_left_boundary(left_boundary_features.transpose(1,2))
        left_boundary_features = self.LeakyReLU(left_boundary_features.transpose(1,2))
        left_boundary_features = self.dropout_left_boundary(left_boundary_features)
        left_boundary_features = self.pos_encoder(left_boundary_features)
        
        right_boundary_features = self.linear_right_boundary(right_boundary_features.view(-1,num_points,lane_boundaries_features_dim))
        right_boundary_features = self.norm_right_boundary(right_boundary_features.transpose(1,2))
        right_boundary_features = self.LeakyReLU(right_boundary_features.transpose(1,2))
        right_boundary_features = self.dropout_right_boundary(right_boundary_features)
        right_boundary_features = self.pos_encoder(right_boundary_features)                      

        lane_features_left = self.cross_attn_left(query=lane_features,
                                            key=left_boundary_features,
                                            value=left_boundary_features)[0]
        
        lane_features_right = self.cross_attn_left(query=lane_features,
                                            key=left_boundary_features,
                                            value=left_boundary_features)[0]
        map_feats = torch.cat([lane_features_left,
                               lane_features_right],-1)

        map_feats = map_feats.contiguous().view(num_agents,-1)
        map_feats = self.dropout1(self.LeakyReLU(self.norm1(self.linear1(map_feats))))
        # map_feats = self.dropout2(self.relu(self.norm2(self.linear2(map_feats))))

        return map_feats
    
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
        
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

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
        output = self.LeakyReLU(output)
        output = self.dropout3(output)
        output = self.linear2(output)

        x_padding = x_padding + self.dropout4(output)
        x_padding = self.norm3(x_padding)

        return x_padding

