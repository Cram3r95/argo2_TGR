#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 17:55:12 2023
@author: Carlos Gómez-Huélamo
"""

import glob
import pickle
import pdb
import numpy as np

import torch.utils.data

from data.argoverse.utils.extractor_proc import ArgoDataExtractor

class ArgoCSVDataset(torch.utils.data.Dataset):
    def __init__(self, input_folder, input_preprocessed, args, input_preprocessed_map=None):
        
        self.input_preprocessed_social = input_preprocessed # social
        self.input_preprocessed_map = input_preprocessed_map # map
        self.args = args

        if args.use_preprocessed:
            # Social
            
            with open(self.input_preprocessed_social, 'rb') as f:
                self.data = pickle.load(f)
                
            # Map

            if self.input_preprocessed_map:
                with open(self.input_preprocessed_map, 'rb') as f:
                    self.data_map = pickle.load(f) 
        
                # Merge info
                
                keys_social = []
                for data_social in self.data:
                    keys_social.append(data_social['argo_id'])
                social_indeces_sorted = np.argsort(keys_social)
                
                keys_map = []
                for data_map in self.data_map:
                    keys_map.append(data_map['argo_id'])
                
                map_indeces_sorted = np.argsort(keys_map)
                
                merged_data = []
                for i in range(len(self.data)):
                    sample_social = self.data[social_indeces_sorted[i]]
                    sample_map = self.data_map[map_indeces_sorted[i]]
                    sample_map['rel_candidate_centerlines'] = np.float32(sample_map['rel_candidate_centerlines'])
                    
                    assert sample_social['argo_id'] == sample_map['argo_id'], "The scenario id do not match"
                    
                    # Merge both dicts
                    
                    sample_social.update(sample_map) # This dict contains both dicts without repeating the keys
                    merged_data.append(sample_social)
                
                self.data = merged_data

        else:
            # TODO: Integrate the map preprocessing here, in addition to the social data
            self.files = glob.glob(f"{input_folder}/**/*.parquet")
            if args.reduce_dataset_size > 0:
                self.files = self.files[:args.reduce_dataset_size]

            self.argo_reader = ArgoDataExtractor(args)

    def __getitem__(self, idx):
        """Get preprocessed data at idx or preprocess the data at idx

        Args:
            idx: Index of the sample to return

        Returns:
            Preprocessed sample
        """

        if self.args.use_preprocessed:
            return self.data[idx]
        else:
            return self.argo_reader.extract_data(self.files[idx])

    def __len__(self):
        """Get number of samples in the dataset

        Returns:
            Number of samples in the dataset
        """
        if self.args.use_preprocessed:
            return len(self.data)
        else:
            return len(self.files)
