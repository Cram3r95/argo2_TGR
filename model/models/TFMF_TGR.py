#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 17:55:12 2023
@author: Carlos Gómez-Huélamo
"""

import numpy as np
import os
import pdb
import sys

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix

from scipy import sparse
import math

if str(sys.version_info[0])+"."+str(sys.version_info[1]) >= "3.9": # Python >= 3.9
    from math import gcd
else:
    from fractions import gcd

# Get the paths of the repository
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(file_path))

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

#######################################

class TMFModel(pl.LightningModule):
    def __init__(self, args):
        super(TMFModel, self).__init__() # allows us to avoid using the base class name explicitly.
        self.args = args

        self.save_hyperparameters() # It will enable Lightning to store all the provided arguments under the self.hparams attribute. These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.
        
        # Encoder
        
        ## Social
        
        self.linear_embedding = LinearEmbedding(3,self.args)
        self.pos_encoder= PositionalEncoding1D(self.args.social_latent_size)
        self.encoder_transformer = EncoderTransformer(self.args)
        self.agent_gnn = GNN(self.args)
        
        ## Physical
        
        self.physical_encoder = MapSubNet(args)
        mid_dim = (self.args.map_latent_size * self.args.num_centerlines + self.args.map_latent_size) // 2
        # self.mlp_latentmap = make_mlp([self.args.map_latent_size * self.args.num_centerlines, mid_dim, self.args.map_latent_size], 
        self.mlp_latentmap = make_mlp([self.args.map_latent_size * self.args.num_centerlines, self.args.map_latent_size], 
                                       batch_norm=True, 
                                       dropout=self.args.apply_dropout[-1])
        
        # Decoder
        
        self.args.decoder_latent_size = self.args.social_latent_size + self.args.map_latent_size
        # self.decoder_residual = DecoderResidual(self.args)
        self.decoder = Temporal_Multimodal_Decoder(self.args)

        # Metrics
        
        self.automatic_optimization = True
        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.min_ade = 50000
        self.lr = 0.001
        self.min_lr = 5e-5
        self.is_frozen = False

    @staticmethod
    def init_args(parent_parser):
        parser_dataset = parent_parser.add_argument_group("dataset")
        parser_dataset.add_argument(
            "--train_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "train"))
        parser_dataset.add_argument(
            "--val_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "val"))
        parser_dataset.add_argument(
            "--test_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "test"))
        
        # Social preprocess
        
        parser_dataset.add_argument(
            "--train_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "processed_social", "train_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "processed_social", "val_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "processed_social", "test_pre_clean.pkl"))
        
        # Map preprocess
        
        parser_dataset.add_argument(
            "--train_split_pre_map", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "processed_map", "train_map_data.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre_map", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "processed_map", "val_map_data.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre_map", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "processed_map", "test_map_data.pkl"))
        
        parser_dataset.add_argument(
            "--reduce_dataset_size", type=int, default=0)
        parser_dataset.add_argument(
            "--use_preprocessed", type=bool, default=False)
        parser_dataset.add_argument(
            "--align_image_with_target_x", type=bool, default=True)

        parser_training = parent_parser.add_argument_group("training")
        parser_training.add_argument("--num_epochs", type=int, default=100)
        parser_training.add_argument(
            "--lr_values", type=list, default=[1e-3, 6e-4, 3.6e-4 , 2e-4, 1e-4, 5e-5])
        parser_training.add_argument(
            "--lr_step_epochs", type=list, default=[36, 42, 48, 54, 60, 66])
        # parser_training.add_argument(
        #     "--lr_step_epochs", type=list, default=[18, 36, 54])
        parser_training.add_argument("--wd", type=float, default=0.001)
        parser_training.add_argument("--batch_size", type=int, default=1024)
        parser_training.add_argument("--val_batch_size", type=int, default=1024)
        parser_training.add_argument("--workers", type=int, default=0)
        parser_training.add_argument("--val_workers", type=int, default=0)

        parser_model = parent_parser.add_argument_group("model")
        parser_model.add_argument("--data_dim", type=int, default=2)
        parser_model.add_argument("--obs_len", type=int, default=50)
        parser_model.add_argument("--pred_len", type=int, default=60)
        parser_model.add_argument("--centerline_length", type=int, default=40)
        parser_model.add_argument("--num_centerlines", type=int, default=6)
        parser_model.add_argument("--num_attention_heads", type=int, default=4)
        parser_model.add_argument("--apply_dropout", type=list, default=[True,0.15])
        parser_model.add_argument("--social_latent_size", type=int, default=64)
        parser_model.add_argument("--map_latent_size", type=int, default=64)
        parser_model.add_argument("--decoder_latent_size", type=int, default=-1)
        parser_model.add_argument("--decoder_temporal_window_size", type=int, default=49) # Matches the number of displacements
        parser_model.add_argument("--num_modes", type=int, default=6)
        parser_model.add_argument("--mod_steps", type=list, default=[1, 5]) # First unimodal -> Freeze -> Multimodal
        parser_model.add_argument("--mod_freeze_epoch", type=int, default=42)
        parser_model.add_argument("--reg_loss_weight", type=float, default=1) # xy predictions
        parser_model.add_argument("--cls_loss_weight", type=float, default=1) # classification = confidences
        # parser_model.add_argument("--mod_freeze_epoch", type=int, default=36)

        return parent_parser

    def forward(self, batch):
        # Set batch norm to eval mode in order to prevent updates on the running means,
        # if the weights are frozen
        
        if self.is_frozen:
            for module in self.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
        
        displ, centers = batch["displ"], batch["centers"]
        rotation, origin = batch["rotation"], batch["origin"]
        
        # Encoder
        
        ## Social
        
        ### Extract the number of agents in each sample of the current batch
        
        agents_per_sample = [x.shape[0] for x in displ]
        batch_size = len(agents_per_sample)
        
        ### OBS: For each sequence, we always set the focal (target) agent as the first agent
        ###      of the scene, then our ego-vehicle (AV) and finally the remanining agents
        ###      (See extractor_proc.py preprocessing)
        
        focal_agent_id = np.cumsum(agents_per_sample)
        focal_agent_id = np.roll(focal_agent_id,1)
        focal_agent_id[0] = 0
        
        ### Convert the list of tensors to tensors
        
        displ_cat = torch.cat(displ, dim=0)
        centers_cat = torch.cat(centers, dim=0)

        # Data augmentation (TODO: It should be in collate_fn_dict, in the DataLoader)

        if self.training:
            displ_cat[:,:,:2] = self.add_noise(displ_cat[:,:,:2], 0.1)
            centers_cat = self.add_noise(centers_cat, 0.1)
        
        ### Encode agents
        
        linear_output = self.linear_embedding(displ_cat)
        pos_encoding = self.pos_encoder(linear_output)
        pos_encoding = pos_encoding + linear_output

        out_transformer = self.encoder_transformer(pos_encoding, agents_per_sample)
        out_agent_gnn = self.agent_gnn(out_transformer, centers_cat, agents_per_sample)
        out_agent_gnn = torch.stack([x[0] for x in out_agent_gnn]) # Only the focal agent
        
        ## Physical
        
        ### Get relevant centerlines (non-padded) per scenario
        
        rel_candidate_centerlines = batch["rel_candidate_centerlines"]
        rel_candidate_centerlines = torch.stack(rel_candidate_centerlines,dim=0)
        
        ### Get the map latent vector associated 

        _, num_centerlines, points_centerline, data_dim = rel_candidate_centerlines.shape
        rel_candidate_centerlines = rel_candidate_centerlines.contiguous().view(-1, points_centerline, data_dim)

        non_empty_mask = rel_candidate_centerlines.abs().sum(dim=1).sum(dim=1) # A padded-centerline must sum 0.0
        # in each dimension, and after that both dimensions together
        rows_mask = torch.where(non_empty_mask == 0.0)[0]
        non_masked_centerlines = rel_candidate_centerlines.shape[0] - len(rows_mask)
        
        rel_candidate_centerlines_mask = torch.zeros([rel_candidate_centerlines.shape[0]], dtype=torch.bool) # False
        rel_candidate_centerlines_mask[rows_mask] = True
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
        rel_candidate_centerlines = rel_candidate_centerlines[rel_candidate_centerlines_mask_inverted,:,:]

        aux_physical_info = self.physical_encoder(rel_candidate_centerlines, centerlines_per_sample)
        physical_info = torch.zeros((batch_size*num_centerlines, self.args.map_latent_size), 
                                    device=aux_physical_info.device)
        physical_info[rel_candidate_centerlines_mask_inverted] = aux_physical_info
        physical_info = physical_info.contiguous().view(batch_size,self.args.map_latent_size * self.args.num_centerlines)
        physical_info = self.mlp_latentmap(physical_info)

        merged_info = torch.cat([out_agent_gnn, 
                                 physical_info], 
                                 dim=1)

        if torch.any(torch.isnan(merged_info)):
            pdb.set_trace()
            
        # Decoder

        # out_decoder = self.decoder_residual(merged_info, self.is_frozen)
        
        traj_agent_abs_rel = displ_cat[focal_agent_id,:,:self.args.data_dim]
        last_obs_agent = centers_cat[focal_agent_id,:]
        
        decoder_h = merged_info.unsqueeze(0)
        decoder_c = torch.zeros(tuple(decoder_h.shape)).to(decoder_h)
        state_tuple = (decoder_h, decoder_c)
          
        pred_traj_rel, conf = self.decoder(traj_agent_abs_rel, state_tuple)
        
        # Convert relative displacements to absolute coordinates (around origin)

        pred_traj = relative_to_abs_multimodal(pred_traj_rel, last_obs_agent)
        
        # In this model we are only predicting
        # the focal agent. We would actually 
        # have batch_size x num_agents x num_modes x pred_len x data_dim
        num_agents = 1
        pred_traj = pred_traj.contiguous().view(batch_size, num_agents, -1, self.args.pred_len, 2) 
        conf = conf.view(batch_size, num_agents, -1)
        
        # Iterate over each batch and transform predictions into the global coordinate frame, Matrix product of two tensors.
        for i in range(len(pred_traj)):
            pred_traj[i] = torch.matmul(pred_traj[i], rotation[i]) + origin[i].contiguous().view(
                1, 1, 1, -1
            )
        return pred_traj, conf

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
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.decoder_residual.unfreeze_layers()

        self.is_frozen = True

    # def prediction_loss(self, preds, gts, conf):
    #     # Stack all the predicted trajectories of the target agent
    #     num_mods = preds.shape[2]
    #     # [0] is required to remove the unneeded dimensions
    #     preds = torch.cat([x[0] for x in preds], 0)

    #     # Stack all the true trajectories of the target agent
    #     # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
    #     # to the target agent
    #     gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0)
    #     gt_target = torch.repeat_interleave(gt_target, num_mods, dim=0) # repeate the gt for all ks 

    #     loss_single = self.reg_loss(preds, gt_target)
    #     loss_single = torch.sum(torch.sum(loss_single, dim=2), dim=1)

    #     loss_single = torch.split(loss_single, num_mods)

    #     # Tuple to tensor
    #     loss_single = torch.stack(list(loss_single), dim=0)

    #     min_loss_index = torch.argmin(loss_single, dim=1)

    #     min_loss_combined = [x[min_loss_index[i]]
    #                          for i, x in enumerate(loss_single)]

    #     loss_out = torch.sum(torch.stack(min_loss_combined))

    #     return loss_out
    
    def prediction_loss(self, pred, gt, conf):
        """_summary_

        Args:
            preds (torch.tensor): batch_size x num_agents x num_modes x pred_len x data_dim
                            OBS: At this moment, num_agents = 1 since we are only predicting the focal agent 
            gts (list): list of gt of each scenario (num_agents x pred_len x 2)
            conf (torch.tensor): batch_size x num_agents x 1

        Returns:
            _type_: _description_
        """
        
        # Stack all the predicted trajectories of the target agent
        
        pred = pred.squeeze(1)
        conf = conf.squeeze(1)
        batch_size, num_modes, pred_len, data_dim = pred.shape
        
        # Stack all the true trajectories of the target agent
        # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
        # to the target agent
        
        gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gt], 0) # batch_size x pred_len x data_dim
        gt_target_repeated = gt_target.unsqueeze(1).repeat(1,pred.shape[1],1,1) # repeate the gt for all ks 
                                                                                # batch_size x num_modes x pred_len x data_dim

        fde_k = torch.sqrt((pred[:, :, -1, 0] - gt_target_repeated[:, :, -1, 0]) ** 2 + # x
                           (pred[:, :, -1, 1] - gt_target_repeated[:, :, -1, 1]) ** 2)  # y
        k_hat = torch.argmin(fde_k, dim=1) 

        index = torch.tensor(range(pred.shape[0]), dtype=torch.long)
        pred_fut_traj = pred[index, k_hat] # Best trajectory in terms of FDE per sequence  

        batch_size, pred_len, _ = pred_fut_traj.shape
        num_modes = pred.shape[1]
        
        # Regression loss
        
        reg_loss = torch.zeros(1, dtype=torch.float32).to(pred)

        mse_loss = F.mse_loss(pred_fut_traj, gt_target, reduction='none')
        mse_loss = mse_loss.sum(dim=2)
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

    def configure_optimizers(self):
        # if self.current_epoch == self.args.mod_freeze_epoch:
        if self.lr <= self.min_lr:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), weight_decay=self.args.wd)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=0.25,
                                                                   patience=math.ceil(self.args.num_epochs/10),
                                                                   min_lr=1e-6,
                                                                   verbose=True)
        else:
            optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.args.wd)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=0.25,
                                                                   patience=math.ceil(self.args.num_epochs/10),
                                                                   min_lr=1e-6,
                                                                   verbose=True)
        # return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "ade_val"}
    
    def on_train_epoch_start(self):

        # Get learning rate according to current epoch
        for single_param in self.optimizers().param_groups:
            self.lr = single_param["lr"]
            self.log("lr", self.lr, prog_bar=True, sync_dist=True)
        
        # Trigger weight freeze and optimizer reinit if current Learning Rate (LR) is equal or lower than min LR
        # if self.lr <= self.min_lr:
        #     self.freeze()
        #     self.trainer.strategy.setup_optimizers(self.trainer)

    def training_step(self, train_batch, batch_idx):

        pred_traj, conf = self.forward(train_batch)
        loss = self.prediction_loss(pred_traj, train_batch["gt"], conf)
        self.log("loss_train", loss, sync_dist=True)
        return loss

    # def get_lr(self, epoch):
    #     lr_index = 0
    #     for lr_epoch in self.args.lr_step_epochs:
    #         if epoch < lr_epoch:
    #             break
    #         lr_index += 1
    #     return self.args.lr_values[lr_index]
    
    def get_lr(self, optimizer):
        """
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def validation_step(self, val_batch, batch_idx):
        pred_traj, conf = self.forward(val_batch)
        loss = self.prediction_loss(pred_traj, val_batch["gt"], conf)
        self.log("loss_val", loss, sync_dist=True)

        # Extract target agent only
        pdb.set_trace()
        pred = [x[0].detach().cpu().numpy() for x in pred_traj]
        gt = [x[0].detach().cpu().numpy() for x in val_batch["gt"]]
        conf = [x[0].detach().cpu().numpy() for x in conf]
        pdb.set_trace()
        return {"predictions": pred, 
                "groundtruth": gt, 
                "confidences": conf} # = validation_outputs

    def validation_epoch_end(self, validation_outputs):
        """_summary_

        Args:
            validation_outputs (_type_): _description_
        """
        # Extract predictions
        pdb.set_trace()
        pred = [out[0] for out in validation_outputs]
        pred = np.concatenate(pred, 0)
        gt = [out[1] for out in validation_outputs]
        gt = np.concatenate(gt, 0)
        conf = [out[2] for out in validation_outputs]
        conf = np.concatenate(conf, 0)
        
        ade1, fde1, ade, fde = self.calc_prediction_metrics(pred, gt, conf)
        
        if ade <= self.min_ade:
            self.min_ade = ade
            
        self.log("ade1_val", ade1, prog_bar=True, sync_dist=True)
        self.log("fde1_val", fde1, prog_bar=True, sync_dist=True)
        self.log("ade_val", ade, prog_bar=True, sync_dist=True)
        self.log("fde_val", fde, prog_bar=True, sync_dist=True)

    def calc_prediction_metrics(self, preds, gts, conf):
        # Calculate prediction error for each mode
        # Output has shape (batch_size, n_modes, n_timesteps)
        pdb.set_trace()
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

        return ade_1, fde_1, ade, fde

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
        self.nhead = self.args.num_attention_heads
        # self.nhead = self.args.social_latent_size
        self.d_hid = 1 ## dimension of the feedforward network model in nn.TransformerEncoder
        self.num_layers = 1
        self.dropout = self.args.apply_dropout[-1]

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid , self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, transformer_in, agents_per_sample):

        transformer_out = F.relu(self.transformer_encoder(transformer_in))
        return transformer_out[:,-1,:]

class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
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

def make_mlp(dim_list, activation_function="ReLU", batch_norm=False, dropout=0.0, model_output=False):
    """
    Generates MLP network:
    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_function: str, activation function for all layers TODO: Different AF for every layer?
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)
    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    index = 0
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))

        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))

        if model_output and (index == len(dim_list) - 1): # Not apply Activation Function for the last
                                                          # layer if it is the model output
            pass
        else:
            if activation_function == "ReLU":
                layers.append(nn.ReLU())
            elif activation_function == "GELU":
                layers.append(nn.GELU())
            elif activation_function == "Tanh":
                layers.append(nn.Tanh())
            elif activation_function == "LeakyReLU":
                layers.append(nn.LeakyReLU())
                
        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))

        index += 1
    return nn.Sequential(*layers)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, args, h_dim):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = args.num_attention_heads
        self.dropout = args.apply_dropout[-1]
        self.latent_size = h_dim

        self.multihead_attention = nn.MultiheadAttention(self.latent_size, 
                                                         self.num_heads, 
                                                         dropout=self.dropout)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        max_agents = max(agents_per_sample)

        padded_att_in = torch.zeros((len(agents_per_sample), max_agents, self.latent_size), device=att_in[0].device)

        mask = torch.arange(max_agents).to(att_in[0].device) < torch.tensor(agents_per_sample).to(att_in[0].device)[:, None]

        padded_att_in[mask] = att_in
        mask_inverted = ~mask
        mask_inverted = mask_inverted.to(att_in.device)

        padded_att_in_swapped = torch.swapaxes(padded_att_in, 0, 1)

        padded_att_in_swapped, _ = self.multihead_attention(
            padded_att_in_swapped, padded_att_in_swapped, padded_att_in_swapped, key_padding_mask=mask_inverted)

        padded_att_in_reswapped = torch.swapaxes(
            padded_att_in_swapped, 0, 1)

        att_out_batch = [x[0:agents_per_sample[i]]
                            for i, x in enumerate(padded_att_in_reswapped)]

        return att_out_batch
    
class MapSubNet(nn.Module):

    def __init__(self, args):
        super(MapSubNet, self).__init__()
        
        self.h_dim = args.map_latent_size  
        self.num_attention_heads = args.num_attention_heads  
        self.dropout = args.apply_dropout[-1]
        self.num_centerlines = args.num_centerlines
        self.depth = 2
        input_dim = 2 # Each point of the centerline has two attributes (xy)
        mid_dim = (input_dim * args.centerline_length + self.h_dim) // 2
        
        self.MLPs = nn.ModuleList([make_mlp([input_dim*args.centerline_length, mid_dim], batch_norm=True, dropout=self.dropout), 
                                   make_mlp([mid_dim, self.h_dim], batch_norm=True, dropout=self.dropout)])
        self.Attn = nn.ModuleList([MultiHeadSelfAttention(args, h_dim=mid_dim),
                                   MultiHeadSelfAttention(args, h_dim=self.h_dim)])
        self.Norms = nn.ModuleList([nn.LayerNorm(mid_dim), nn.LayerNorm(self.h_dim)])

        self.final_layer = nn.Linear(self.num_centerlines*self.h_dim,self.h_dim)
        
    def forward(self, centerlines, centerlines_per_sample):

        hidden_states_batch = centerlines.contiguous().view(centerlines.shape[0],-1) # rel_candidate_centerlines x (length · data_dim)
        
        for layer_index, layer in enumerate(self.Attn):
            hidden_states_batch = self.MLPs[layer_index](hidden_states_batch)
            temp = hidden_states_batch
            hidden_states_batch = torch.cat(layer(hidden_states_batch, centerlines_per_sample),dim=0) 
            hidden_states_batch = temp + hidden_states_batch
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)

        return hidden_states_batch
    
class DecoderResidual(nn.Module):
    def __init__(self, args):
        super(DecoderResidual, self).__init__()

        self.args = args

        output = []
        for i in range(sum(args.mod_steps)):
            output.append(PredictionNet(args))

        self.output = nn.ModuleList(output) # is just like a Python list. It was designed to store any desired number of nn.Module’s

    def forward(self, decoder_in, is_frozen):
        sample_wise_out = []

        if self.training is False:
            for out_subnet in self.output:
                sample_wise_out.append(out_subnet(decoder_in))
        elif is_frozen:
            for i in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
                sample_wise_out.append(self.output[i](decoder_in))
        else:

            sample_wise_out.append(self.output[0](decoder_in))
            
        decoder_out = torch.stack(sample_wise_out)
        decoder_out = torch.swapaxes(decoder_out, 0, 1)

        return decoder_out

    def unfreeze_layers(self):
        for layer in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
            for param in self.output[layer].parameters():
                param.requires_grad = True

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
        
        # self.latent_predictions = nn.Linear(self.args.num_modes*self.args.pred_len*self.args.data_dim,
        #                                     self.args.social_latent_size + self.args.map_latent_size)
        # self.confidences = nn.Sequential(LinearRes(self.decoder_h_dim*2, self.decoder_h_dim*2, norm=norm, ng=ng), 
        #                                  nn.Linear(self.decoder_h_dim*2, self.num_modes))
        self.confidences = nn.Sequential(LinearRes(self.decoder_h_dim, self.decoder_h_dim, norm=norm, ng=ng), 
                                         nn.Linear(self.decoder_h_dim, self.num_modes))
        
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
            decoder_input = F.dropout(decoder_input, p=self.args.apply_dropout[-1], training=self.training)
        
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
                decoder_input = F.dropout(decoder_input, p=self.args.apply_dropout[-1], training=self.training)
        
            curr_pred_traj_fake_rel = torch.stack(curr_pred_traj_fake_rel,dim=0)
            curr_pred_traj_fake_rel = curr_pred_traj_fake_rel.permute(1,0,2)
            pred_traj_fake_rel.append(curr_pred_traj_fake_rel)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0) # num_modes, batch_size, pred_len, data_dim
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1,0,2,3) # batch_size, num_modes, pred_len, data_dim

        # Obtain confidences based on the initial latent state and the predictions

        # predictions_latent = self.latent_predictions(pred_traj_fake_rel.contiguous().view(batch_size, -1))
        # state_tuple_h = state_tuple_h.squeeze(0)
        # conf_latent = torch.cat([state_tuple_h,
        #                          predictions_latent],
        #                          dim=1)
  
        conf_latent = state_tuple_h.squeeze(0)
        conf = self.confidences(conf_latent)
        conf = torch.softmax(conf.view(batch_size,-1), dim=1) # batch_size, num_modes
        if not torch.allclose(torch.sum(conf, dim=1), conf.new_ones((batch_size,))):
            pdb.set_trace()
        
        return pred_traj_fake_rel, conf
    
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
