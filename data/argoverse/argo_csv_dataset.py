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
import time
import os
from pathlib import Path

import torch.utils.data

from data.argoverse.utils.extractor_proc import ArgoDataExtractor

class ArgoCSVDataset(torch.utils.data.Dataset):
    def __init__(self, input_folder, input_preprocessed_social, args, 
                       input_preprocessed_map=None, input_preprocessed_full=None):
        
        self.input_preprocessed_social = input_preprocessed_social # social
        self.input_preprocessed_map = input_preprocessed_map # map
        self.input_preprocessed_full = input_preprocessed_full
        self.args = args
        
        if args.use_preprocessed:
            # We have already merged (social+map) and dumped the data
            # Not none and the file exists
            if self.input_preprocessed_full and os.path.exists(self.input_preprocessed_full): 
                print(f"Loading preprocess full information from: {self.input_preprocessed_full}")
                with open(self.input_preprocessed_full, 'rb') as f:
                    self.data = pickle.load(f)
            else:
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
                    start = time.time()
                    for i in range(len(self.data)):
                        sample_social = self.data[social_indeces_sorted[i]]
                        sample_map = self.data_map[map_indeces_sorted[i]]
                        
                        assert sample_social['argo_id'] == sample_map['argo_id'], "The scenario ids do not match"
    
                        # Extract map data 
                        
                        keys_of_interest = ["rel_centerline", "centerline_type", "is_intersection", 
                                            "rel_left_bound", "left_type",
                                            "rel_right_bound", "right_type"]

                        CENTERLINE_LENGTH = 40 # TODO: Avoid this hard-code. All arrays should have exactly the same
                                            # size, including the relative displacements!!!!!!

                        for key_interest in keys_of_interest:
                            aux = []
                            for key in sample_map.keys(): # Iterate over the different agents
                                if key != "argo_id":
                                    aux_ = []
                                    for index in range(len(sample_map[key])):
                                        array = sample_map[key][index][key_interest]

                                        if array.shape[0] == CENTERLINE_LENGTH - 1:  
                                            array = np.concatenate([np.zeros((1,2)),array]) 

                                        aux_.append(array)
                                    
                                    aux_ = np.array(aux_)
                                    aux.append(aux_)

                            aux = np.array(aux)
                            sample_social[key_interest] = np.float32(aux)

                        # Update social dict with map information
                    
                        merged_data.append(sample_social)
                    
                    self.data = merged_data
                    
                    end = time.time()
                    split = input_folder.split("/")[-1]
                    print(f"Finish {split} merging data: {end-start}")

                    # Save data as pkl file (After it is finished)
        
                    aux_path = Path(self.input_preprocessed_full)
                    PREPROCESSED_FULL_DIR = aux_path.parent.absolute()
                    os.makedirs(PREPROCESSED_FULL_DIR,exist_ok=True)
                    
                    filename = self.input_preprocessed_full
                    print(f"Save data in {self.input_preprocessed_full}")
                    
                    with open(filename, 'wb') as f:
                        pickle.dump(self.data, f)
                    
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
