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
import git
import sys
import pickle

from pathlib import Path

# DL & Math imports

import pandas as pd
import math
import numpy as np

from shapely.geometry import LineString, Polygon

# Plot imports

import matplotlib
import matplotlib.pyplot as plt

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from map_features_utils_argo2 import MapFeaturesUtils, ScenarioMap, \
                                     get_agent_velocity_and_acceleration, get_yaw, rotz2D, apply_rotation
from file_utils import load_list_from_folder

## Argoverse 1

from argoverse.map_representation.map_api import ArgoverseMap
# from argoverse.utils.mpl_plotting_utils import visualize_centerline

## Argoverse 2

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

# Global variables

OBS_LEN = 50
PRED_LEN = 60
FREQUENCY = 10 # Hz
PERIOD = float(1 / FREQUENCY) # s
VIZ = True
limit_qualitative_results = 150
MODE = "test" # "train","test" 
# if train -> compute the best candidate (oracle), only using the "competition" algorithm
# if test, return N plausible candidates. Choose between "competition", "map_api" and "get_around" algorithms
ALGORITHM = "competition" # competition, map_api, get_around
                      # TODO: At this moment, only the "map_api" algorithm is prepared
                      # to retrieve the centerlines (both relevant and oracle) correctly
ALIGN = "x-axis" # If x-axis, the focal agent's last observation is aligned with the x-axis (to the right), 
                 # so the social and map information is rotated according to this orientation.
                 # y-axis is the same but facing up  
MAX_CENTERLINES = 3
MIN_POINTS_INTERP = 4 # to perform a cubic interpolation you need at least 3 points
RELATIVE_DISPLACEMENTS = True
SAVE_DIR = os.path.join(BASE_DIR,"preprocess/computed_relevant_centerlines_examples")  
DATASETS_DIR = "/home/robesafe/shared_home/benchmarks/argoverse2/motion-forecasting"
os.makedirs(SAVE_DIR, exist_ok=True)

#######################################

# Global variables

avm = ArgoverseMap() # Argo1 map

TYPE_MARKER_DICT = {"DYNAMIC": (0.0,0.0,1.0), # blue
                    "STATIC":  (0.0,1.0,0.0), # green
                    "UNKNOWN": (0.0,0.0,0.0)} # black
RELEVANT_OPACITY_DICT = {"0": 1.0, # FOCAL_TRACK
                         "1": 0.7, # SCORED_TRACK
                         "2": 0.4, # UNSCORED_TRACK
                         "3": 0.1} # TRACK_FRAGMENT

DYNAMIC_OBJECT_TYPES = ("AGENT", "AV", "vehicle", "pedestrian", "motorcyclist", "cyclist", "bus")
STATIC_OBJECT_TYPES = ("static", "background", "construction", "riderless_bicycle")
UNKNOWN_OBJECT_TYPES = ("unknown")
                        
# Aux functions

line_types = {"cl":["--"],"bound":["-"]}

def visualize_centerline(centerline: LineString, 
                         color: tuple,
                         alpha: float,
                         line_type: str) -> None:
    """Visualize the computed centerline.

    Args:
        centerline: Sequence of coordinates forming the centerline
    """
    
    
    line_coords = list(zip(*centerline))
    lineX = line_coords[0]
    lineY = line_coords[1]
    plt.plot(lineX, 
             lineY, 
             line_types[line_type][0], 
             color=color, 
             alpha=alpha, 
             linewidth=1, 
             zorder=0)
    
    if line_type == "cl":
        plt.text(lineX[0], lineY[0], "s")
        plt.text(lineX[-1], lineY[-1], "e")
    plt.axis("equal")
    
# Specify the splits you want to preprocess

                         # Split,  Flag, Split percentage
splits_to_process = dict({"train":[False,1.0], # 0.01 (1 %), 0.1 (10 %), 1.0 (100 %)
                          "val":  [False,1.0],
                          "test": [True,1.0]})

# Preprocess the corresponding folder

for split_name,features in splits_to_process.items():
    if features[0]:
        print(f"Analyzing physical information of {split_name} split ...")

        folder = os.path.join(DATASETS_DIR,split_name)
        folder_list, num_folders = load_list_from_folder(folder)

        num_folders = int(features[1]*num_folders)
        print("Num folders to analyze: ", num_folders)
        
        folder_list = folder_list[:num_folders]
                
        check_every = 0.2
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

            # Social features
            
            scenario_path = os.path.join(folder_,f"scenario_{scenario_id}.parquet") # social
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            df = pd.read_parquet(scenario_path)
            
            # Map features
            
            static_map_path = os.path.join(folder_,f"log_map_archive_{scenario_id}.json") # map
            mfu = MapFeaturesUtils(static_map_path) # My own map functions
            
            # Get all agents
            
            agt_ts = np.sort(np.unique(df["timestep"].values))
     
            mapping = dict()
            for j, ts in enumerate(agt_ts):
                mapping[ts] = j
            
            trajs = np.concatenate((
                df.position_x.to_numpy().reshape(-1, 1),
                df.position_y.to_numpy().reshape(-1, 1)), 1)
            
            steps = [mapping[x] for x in df["timestep"].values]
            steps = np.asarray(steps, np.int64)

            # Replace focal_track_id and AV in object_type
            
            df['object_type']= df.apply(lambda row: 'AGENT' if row['track_id']==row['focal_track_id'] else row['object_type'],axis=1)
            df['object_type']= df.apply(lambda row: 'AV' if row['track_id']=='AV' else row['object_type'],axis=1)

            # objs = df.groupby(["track_id", "object_type"]).groups
            # keys = list(objs.keys())
            objs = df.groupby(["track_id", "object_type", "object_category"]).groups
            keys = list(objs.keys())
        
            obj_type = [x[1] for x in keys]
        
            agnt_key = keys.pop(obj_type.index("AGENT"))
            av_key = keys.pop(obj_type.index("AV")-1)
            keys = [agnt_key, av_key] + keys 
            
            # For each sequence, we always set the focal (target) agent as the first agent
            # of the scene, then our ego-vehicle (AV) and finally the remanining agents
        
            # Compute the most relevant centerlines for each relevant agent and align these map features with 
            # the FOCAL AGENT last orientation
            # OBS: An agent is only relevant if it is present in the obs-len-th timestamp (last observation frame)
            
            filename = os.path.join(SAVE_DIR,f"{split_name}_candidates_{MAX_CENTERLINES}_{scenario_id}.png")
            sample = dict()
            sample["argo_id"] = scenario_id
            scene_yaw = None
            map_origin = None
            
            plt.figure(0, figsize=(8, 7))
            
            relevant_obstacles = 0
            track_fragment_obstacles = 0
            unscored_track_obstacles = 0
            scored_track_obstacles = 0
            focal_track_obstacles = 0
        
            for agent_index,key in enumerate(keys):
                idcs = objs[key]    
                ts = steps[idcs]

                # Condition to include the agent
                
                current_object_type = key[1]
                current_object_category = key[2]
                
                if ((OBS_LEN - 1) not in ts
                    or current_object_type not in DYNAMIC_OBJECT_TYPES
                    or current_object_category == 0):
                    continue
                
                relevant_obstacles += 1
                if key[2] == 0:
                    track_fragment_obstacles += 1
                elif key[2] == 1:
                    unscored_track_obstacles += 1
                elif key[2] == 2:
                    scored_track_obstacles += 1
                elif key[2] == 3:
                    focal_track_obstacles += 1
                
                curr_obs_len = np.where(ts == (OBS_LEN - 1))[0].item() # Get index of the (obs_len - 1)-th timestamp
                agent = df.loc[df["track_id"] == key[0]]
                agent_track = agent[["position_x","position_y","heading","velocity_x","velocity_y"]].to_numpy()
                agent_track_full_xy = agent_track[:,:2]
                    
                xy = agent_track_full_xy[:curr_obs_len+1,:2]

                ## Filter agent's trajectory (smooth)

                vel, acc, xy_filtered, extended_xy_filtered = get_agent_velocity_and_acceleration(xy,period=PERIOD)
                
                ## Compute agent's orientation in the last observation frame

                # Filter the focal agent and compute the orientation (reference for the whole sequence)
                if agent_index == 0:
                    assert key[1] == "AGENT", print("The focal agent is not the first key")
                    lane_dir_vector, scene_yaw = get_yaw(xy_filtered, curr_obs_len)
                    
                    yaw = scene_yaw
                    map_origin = xy[-1] # origin for the remaining agents and centerlines
                else:
                    if xy_filtered.shape[0] > 1:
                        lane_dir_vector, yaw = get_yaw(xy_filtered, curr_obs_len)
                    else:
                        yaw = 0

                # Get most relevant physical information
                # candidate_centerlines, rel_candidate_centerlines_array
                # TODO: Get these arrays as np.float32, not np.float64!!
                candidate_hdmap_info = mfu.get_candidate_centerlines_for_trajectory( 
                                                                                filename,
                                                                                [agent_track_full_xy,xy_filtered,extended_xy_filtered],
                                                                                [vel,acc,yaw],
                                                                                map_origin,
                                                                                avm,
                                                                                max_candidates = MAX_CENTERLINES,
                                                                                mode=MODE,
                                                                                algorithm=ALGORITHM,
                                                                                time_variables=[curr_obs_len,PRED_LEN,FREQUENCY],
                                                                                normalize_rotation=ALIGN,
                                                                                scene_yaw=scene_yaw,
                                                                                relative_displacements=RELATIVE_DISPLACEMENTS,
                                                                                agent_index=agent_index)
                
                sample[key] = candidate_hdmap_info

                # Visualize agents with their corresponding relevant centerlines
                if viz_:
                    if key[1] == "AGENT":
                        color = (1.0,0.0,0.0)
                        opacity = RELEVANT_OPACITY_DICT["0"]
                        
                    else:
                        # color = (np.random.random(), np.random.random(), np.random.random()) 
                        
                        current_object_type = key[1]
                        if current_object_type in DYNAMIC_OBJECT_TYPES:
                            current_object_type_ = "DYNAMIC"
                        elif current_object_type in STATIC_OBJECT_TYPES:
                            current_object_type_ = "STATIC"
                        elif current_object_type in UNKNOWN_OBJECT_TYPES:
                            current_object_type_ = "UNKNOWN"
                        else:
                            raise Exception("Sorry, unknown object type")
                         
                        color = TYPE_MARKER_DICT[current_object_type_]
                        opacity = RELEVANT_OPACITY_DICT[str(key[2])]  

                    for candidate_hdmap_info_ in candidate_hdmap_info:
                        if not np.any(candidate_hdmap_info_["centerline"]): # Do not visualize empty HDMap info
                            continue
                        # Centerline
                        visualize_centerline(candidate_hdmap_info_["centerline"],color,opacity,"cl")
                        # Left bound
                        visualize_centerline(candidate_hdmap_info_["left_bound"],color,opacity,"bound")
                        # Right bound
                        visualize_centerline(candidate_hdmap_info_["right_bound"],color,opacity,"bound")
                        
                    # Observation 
                
                    ## Rotate trajectory

                    R = rotz2D(scene_yaw)
                    agent_track_full_xy = np.subtract(agent_track_full_xy,map_origin)
                    agent_track_full_xy = apply_rotation(agent_track_full_xy,R)

                    plt.plot(
                        agent_track_full_xy[:curr_obs_len, 0],
                        agent_track_full_xy[:curr_obs_len, 1],
                        "-",
                        color=color,# color="#d33e4c",
                        alpha=opacity,
                        linewidth=3,
                        zorder=1,
                    )

                    final_x = agent_track_full_xy[curr_obs_len, 0]
                    final_y = agent_track_full_xy[curr_obs_len, 1]

                    plt.plot(
                        final_x,
                        final_y,
                        "o",
                        color=color,#color="#d33e4c",
                        alpha=opacity,
                        markersize=5,
                        zorder=1,
                    )
                    
                    plt.text(
                        final_x + 1,
                        final_y + 1,
                        f"{agent_index}",
                        fontsize=5,
                        zorder=2
                        )
                    
                    # Ground-truth prediction
                    
                    plt.plot(
                        agent_track_full_xy[curr_obs_len:, 0],
                        agent_track_full_xy[curr_obs_len:, 1],
                        "-",
                        color=color,#color="blue",
                        alpha=opacity,
                        linewidth=3,
                        zorder=1,
                    )

                    final_x = agent_track_full_xy[-1, 0]
                    final_y = agent_track_full_xy[-1, 1]

                    plt.plot(
                        final_x,
                        final_y,
                        "D",
                        color=color,#color="blue",
                        alpha=opacity,
                        markersize=5,
                        zorder=1,
                    )
                    
                    # Uncomment this to obtain each particular each
                    
                    filename_agent = os.path.join(SAVE_DIR,f"candidates_{MAX_CENTERLINES}_{scenario_id}_{agent_index}.pdf") # .png, .pdf
                    plt.xlabel("Map X")
                    plt.ylabel("Map Y")
                    # plt.axis("off")
                    # plt.title(f"Number of candidates = {len(candidate_centerlines)}")
                    plt.savefig(filename_agent, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)
                    plt.close('all')
            pdb.set_trace()
            if viz_: 
                # Uncomment this to obtain the whole scene
                
                plt.xlabel("Map X")
                plt.ylabel("Map Y")
                # plt.axis("off")
                 
                plt.title(f"Relevant obstacles = {relevant_obstacles}\n\
                            Track fragments = {track_fragment_obstacles}\n\
                            Unscored obstacles = {unscored_track_obstacles}\n\
                            Scored obstacles = {scored_track_obstacles}\n\
                            Focal obstacles = {focal_track_obstacles}")
                            
                plt.savefig(filename, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)
                plt.close('all')

            preprocessed.append(sample)
            
            end = time.time()

            aux_time += (end-start)
            time_per_iteration = aux_time/(i+1)

            NUM_SCENES = 10 # consider these iterations to compute the Estimated Time to Finalize
            if ((i > 0 and i % check_every_n_files == 0)
             or i == NUM_SCENES):
                print(f"Time per iteration: {time_per_iteration} s. \n \
                        Estimated time to finish ({folders_remaining} files): {round(time_per_iteration*folders_remaining/60)} min")
                
                if i != NUM_SCENES:
                    filename = os.path.join(DATASETS_DIR,"processed_map",f"{split_name}_map_data_rot_right_x_multi_agent_{i}_{len(folder_list)}.pkl")
                    print(f"Save backup data in {filename}")
                    
                    with open(filename, 'wb') as f:
                        pickle.dump(preprocessed, f)
            
        # Save data as pkl file
        
        filename = os.path.join(DATASETS_DIR,"processed_map",f"{split_name}_map_data_rot_right_x_multi_agent.pkl")
        print(f"Save data in {filename}")
        
        with open(filename, 'wb') as f:
            pickle.dump(preprocessed, f)
            
