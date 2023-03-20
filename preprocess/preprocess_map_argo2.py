#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 22 13:29:27 2023
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import pdb
import time
import os
from pathlib import Path
import git
import sys
import pickle

# DL & Math imports

import pandas as pd
import math
import numpy as np

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from map_features_utils_argo2 import MapFeaturesUtils, ScenarioMap
from file_utils import load_list_from_folder

## Argoverse 1

from argoverse.map_representation.map_api import ArgoverseMap

## Argoverse 2

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

# Global variables

OBS_LEN = 50
PRED_LEN = 60
VIZ = True
limit_qualitative_results = 150
MODE = "test" # "train","test" 
# if train -> compute the best candidate (oracle), only using the "competition" algorithm
# if test, return N plausible candidates. Choose between "competition", "map_api" and "get_around" algorithms
ALGORITHM = "map_api" # competition, map_api, get_around
                      # TODO: At this moment, only the "map_api" algorithm is prepared
                      # to retrieve the centerlines (both relevant and oracle) correctly
ALIGN = "x-axis" # If x-axis, the focal agent's last observation is aligned with the x-axis (to the right), 
                 # so the social and map information is rotated according to this orientation.
                 # y-axis is the same but facing up  
MAX_CENTERLINES = 3
INTERPOLATE_CENTERLINE_POINTS = 40
RELATIVE_DISPLACEMENTS = True
SAVE_DIR = os.path.join(BASE_DIR,"preprocess/computed_relevant_centerlines_examples")  
DATASETS_DIR = "/home/robesafe/shared_home/benchmarks/argoverse2/motion-forecasting"
os.makedirs(SAVE_DIR, exist_ok=True)

#######################################

# Global variables

avm = ArgoverseMap()
mfu = MapFeaturesUtils()

# Specify the splits you want to preprocess

                         # Split,  Flag, Split percentage
splits_to_process = dict({"train":[True,0.01], # 0.01 (1 %), 0.1 (10 %), 1.0 (100 %)
                          "val":  [False,1.0],
                          "test": [False,1.0]})

# Preprocess the corresponding folder

for split_name,features in splits_to_process.items():
    if features[0]:
        print(f"Analyzing physical information of {split_name} split ...")

        folder = os.path.join(DATASETS_DIR,split_name)
        folder_list, num_folders = load_list_from_folder(folder)

        num_folders = int(features[1]*num_folders)
        print("Num folders to analyze: ", num_folders)
        
        folder_list = folder_list[:num_folders]
                
        check_every = 0.1
        check_every_n_files = math.ceil(len(folder_list)*check_every)
        print(f"Check remaining time every {check_every_n_files} files")
        time_per_iteration = float(0)
        aux_time = float(0)
        viz_ = VIZ

        preprocessed = []
        
        for i, folder_ in enumerate(folder_list):
            if limit_qualitative_results != -1 and i+1 > limit_qualitative_results:
                viz_ = False
                        
            folders_remaining = len(folder_list) - (i+1)
            
            scenario_id = folder_.split("/")[-1]
            
            start = time.time()

            scenario_path = os.path.join(folder_,f"scenario_{scenario_id}.parquet") # social
            static_map_path = os.path.join(folder_,f"log_map_archive_{scenario_id}.json") # map

            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            df = pd.read_parquet(scenario_path)
            focal_agent = df.loc[df["track_id"] == scenario.focal_track_id]
            focal_agent_track = focal_agent[["position_x","position_y","heading","velocity_x","velocity_y"]].to_numpy()
            focal_agent_xy = focal_agent_track[:,:2]

            map_json = ScenarioMap(static_map_path)

            filename = os.path.join(SAVE_DIR,f"candidates_{MAX_CENTERLINES}_{scenario_id}.png")

            # Get most relevant physical information
            
            candidate_centerlines, rel_candidate_centerlines_array = mfu.get_candidate_centerlines_for_trajectory(focal_agent_xy,
                                                                                map_json,
                                                                                filename,
                                                                                avm,
                                                                                viz=viz_,
                                                                                max_candidates = MAX_CENTERLINES,
                                                                                mode=MODE,
                                                                                algorithm=ALGORITHM,
                                                                                normalize_rotation=ALIGN,
                                                                                interpolate_centerline_points=INTERPOLATE_CENTERLINE_POINTS,
                                                                                relative_displacements=RELATIVE_DISPLACEMENTS)
            
            # Store info in dict to save in pkl format
            pdb.set_trace()
            sample = dict()
            sample["argo_id"] = scenario_id
            sample["rel_candidate_centerlines"] = rel_candidate_centerlines_array
            preprocessed.append(sample)
                
            end = time.time()

            aux_time += (end-start)
            time_per_iteration = aux_time/(i+1)
            
            if i % check_every_n_files == 0:
                print(f"Time per iteration: {time_per_iteration} s. \n \
                        Estimated time to finish ({folders_remaining} files): {round(time_per_iteration*folders_remaining/60)} min")
                
        # Save data as pkl file
        pdb.set_trace()
        filename = os.path.join(DATASETS_DIR,"processed_map",f"{split_name}_map_data_rot_right_x_multi_agent.pkl")
        print(f"Save data in {filename}")
        
        with open(filename, 'wb') as f:
            pickle.dump(preprocessed, f)