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

from typing import List, Dict, Union

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
        self.pos_encoder= PositionalEncoding1D(self.args)
        self.encoder_transformer = EncoderTransformer(self.args)
        
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
        
        self.args.decoder_latent_size = self.args.social_latent_size
        
        ## Global interaction
        
        self.agent_gnn = AgentGNN(self.args)
        
        # Decoder
        
        self.decoder = PredNet(self.args)

        # Metrics

        self.validation_type = "single-agent" # single-agent = only focal track (category 3)
                                              # multi-agent = both focal track and scored tracks (category 2 & 3)
        self.loss_lane = LossLane(self.args)
        self.initial_lr_conf = self.args.initial_lr_conf
        self.min_lr_conf = self.args.min_lr_conf
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
        parser_training.add_argument("--num_epochs", type=int, default=200)
        parser_training.add_argument("--check_val_every_n_epoch", type=int, default=10)
        parser_training.add_argument("--lr_values", type=list, default=[1e-3, 1e-4, 1e-3 , 1e-4])
        parser_training.add_argument("--lr_step_epochs", type=list, default=[10, 20, 45])
        parser_training.add_argument("--initial_lr_conf", type=float, default=5e-5)
        parser_training.add_argument("--min_lr_conf", type=float, default=1e-6)
        parser_training.add_argument("--wd", type=float, default=0.001)
        parser_training.add_argument("--batch_size", type=int, default=128)
        parser_training.add_argument("--val_batch_size", type=int, default=128)
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
        parser_model.add_argument("--apply_dropout", type=float, default=0.2)
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
        parser_model.add_argument("--epsilon", type=float, default=0.00000001)

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

        actor_idcs = get_actor_ids(gpu(batch["displ"]))
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

        agents_features = self.encoder_transformer(pos_encoding, agents_per_sample) # Deep social features
        
        ## Physical (map)
      
        dedoder_features = self.agent_gnn(agents_features, centers_cat, agents_per_sample)
        
        # Decoder
        pdb.set_trace()
        out = self.decoder(dedoder_features, actor_idcs, centers)
        
        ## Iterate over each batch and transform predictions into the global coordinate frame
        
        rot, orig = gpu(batch["rotation"]), gpu(batch["origin"])
        
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
            
        pdb.set_trace()

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
        """
        Get learning rate according to current epoch
        """

        for single_param in self.optimizers().param_groups:
            self.log("lr", single_param["lr"], prog_bar=True, sync_dist=True)
        
    def training_step(self, train_batch, batch_idx):
        """_summary_

        Args:
            train_batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        out = self.forward(train_batch)
        pdb.set_trace()
        loss = self.loss_lane(out, train_batch)

        self.log("loss_train", loss["loss"], sync_dist=True)

        return loss["loss"]

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch)
        pdb.set_trace()
        if self.validation_type == "single-agent":
            loss = self.loss_lane(out, val_batch)
        elif self.validation_type == "multi-agent":
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

        # TODO: Get current script name to save
        # if self.save_model_script:
        #     
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
        
# Layers

class LinearEmbedding(nn.Module):
    def __init__(self,args):
        super(LinearEmbedding, self).__init__()
        self.input_size = args.num_social_features
        self.output_size = args.social_latent_size

        self.encoder_input_layer = nn.Linear(
                in_features=self.input_size, 
                out_features=self.output_size 
                    )
    def forward(self,linear_input):

        linear_out = F.relu(self.encoder_input_layer(linear_input))

        return linear_out 
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, args):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        channels = args.social_latent_size
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
        # self.latent_size = args.social_latent_size # Message-Passing only included social info
        self.latent_size = args.decoder_latent_size # Message-Passing including social and map info
 
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
    
class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)
        
        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')
        
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out
    
class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
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

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out
    
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
        # feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs, actor_idcs) 
        feats = self.att_dest(actors, torch.stack(actor_ctrs), dest_ctrs, actor_idcs) 
        cls = self.cls(feats).view(-1, self.config["num_mods"])
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
        # agts = self.K_agt(k_actors)
        return k_actors

# Loss and metrics functions

class LossLane(nn.Module):
    def __init__(self, args):
        super(LossLane, self).__init__()
        self.args = args
        self.pred_loss = PredLoss(self.args)

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

class PredLoss(nn.Module):
    def __init__(self, args):
        super(PredLoss, self).__init__()
        self.args = args
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
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]

    return actor_idcs

