
import numpy as np
import os
import pdb

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix

from scipy import sparse
import math

# Get the paths of the repository
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(file_path))

#######################################

class TMFModel(nn.Module):
    def __init__(self, args):
        super(TMFModel, self).__init__() # allows us to avoid using the base class name explicitly.
        self.args = args
                
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
        
        self.decoder_residual = DecoderResidual(self.args)

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
        parser_training.add_argument("--num_epochs", type=int, default=72)
        parser_training.add_argument(
            "--lr_values", type=list, default=[1e-3, 6e-4, 3.6e-4 , 2e-4, 1e-4, 5e-5])
        parser_training.add_argument(
            "--lr_step_epochs", type=list, default=[36, 42, 48, 54, 60, 66])
        # parser_training.add_argument(
        #     "--lr_step_epochs", type=list, default=[18, 36, 54])
        parser_training.add_argument("--wd", type=float, default=0.01)
        parser_training.add_argument("--batch_size", type=int, default=32)
        parser_training.add_argument("--val_batch_size", type=int, default=32)
        parser_training.add_argument("--workers", type=int, default=0)
        parser_training.add_argument("--val_workers", type=int, default=0)
        parser_training.add_argument("--gpus", type=int, default=1)

        parser_model = parent_parser.add_argument_group("model")
        parser_model.add_argument("--obs_len", type=int, default=50)
        parser_model.add_argument("--pred_len", type=int, default=60)
        parser_model.add_argument("--centerline_length", type=int, default=40)
        parser_model.add_argument("--num_centerlines", type=int, default=6)
        parser_model.add_argument("--num_attention_heads", type=int, default=4)
        parser_model.add_argument("--apply_dropout", type=list, default=[True,0.25])
        parser_model.add_argument("--social_latent_size", type=int, default=128)
        parser_model.add_argument("--map_latent_size", type=int, default=128)
        parser_model.add_argument("--decoder_latent_size", type=int, default=256)
        parser_model.add_argument("--mod_steps", type=list, default=[1, 5]) # First unimodal -> Freeze -> Multimodal
        parser_model.add_argument("--mod_freeze_epoch", type=int, default=42)
        # parser_model.add_argument("--mod_freeze_epoch", type=int, default=36)

        return parent_parser

    def forward(self, batch, temperature=1):
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
        
        ### Convert the list of tensors to tensors
        
        displ_cat = torch.cat(displ, dim=0)
        centers_cat = torch.cat(centers, dim=0)
        
        ### Encode agents
        
        linear_output = self.linear_embedding(displ_cat)
        pos_encoding = self.pos_encoder(linear_output)
        pos_encoding = pos_encoding + linear_output

        out_transformer = self.encoder_transformer(pos_encoding, agents_per_sample)
        out_agent_gnn = self.agent_gnn(out_transformer, centers_cat, agents_per_sample)
        out_agent_gnn= torch.stack([x[0] for x in out_agent_gnn])
        
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

        # Decoder

        out_linear = self.decoder_residual(merged_info, self.is_frozen)
        out = out_linear.contiguous().view(len(displ), 1, -1, self.args.pred_len, 2)

        # Iterate over each batch and transform predictions into the global coordinate frame, Matrix product of two tensors.
        for i in range(len(out)):
            out[i] = torch.matmul(out[i], rotation[i]) + origin[i].contiguous().view(
                1, 1, 1, -1
            )
        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.decoder_residual.unfreeze_layers()

        self.is_frozen = True

    def prediction_loss(self, preds, gts):
        # Stack all the predicted trajectories of the target agent
        num_mods = preds.shape[2]
        # [0] is required to remove the unneeded dimensions
        preds = torch.cat([x[0] for x in preds], 0)

        # Stack all the true trajectories of the target agent
        # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
        # to the target agent
        gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0)
        gt_target = torch.repeat_interleave(gt_target, num_mods, dim=0) # repeate the gt for all ks 

        loss_single = self.reg_loss(preds, gt_target)
        loss_single = torch.sum(torch.sum(loss_single, dim=2), dim=1)

        loss_single = torch.split(loss_single, num_mods)

        # Tuple to tensor
        loss_single = torch.stack(list(loss_single), dim=0)

        min_loss_index = torch.argmin(loss_single, dim=1)

        min_loss_combined = [x[min_loss_index[i]]
                             for i, x in enumerate(loss_single)]

        loss_out = torch.sum(torch.stack(min_loss_combined))

        return loss_out

    def calc_prediction_metrics(self, preds, gts):
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
        # self.nhead = self.args.num_attention_heads # 128
        self.nhead = self.args.social_latent_size
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
