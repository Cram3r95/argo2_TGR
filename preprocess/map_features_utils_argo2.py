#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 22 13:29:27 2023
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import pdb
import os
import copy
from pathlib import Path
import git
import sys

from typing import Any, Dict, List, Tuple, Union, Mapping, Iterable, cast
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from collections import defaultdict

# DL & Math imports

import math
import torch
import cv2
import scipy as sp
import numpy as np
import pandas as pd
import json

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

## Argoverse 1

from argoverse.utils.geometry import point_inside_polygon
from argoverse.utils.centerline_utils import (
    centerline_to_polygon,
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    remove_overlapping_lane_seq
)
from argoverse.utils.manhattan_search import (
    find_all_polygon_bboxes_overlapping_query_bbox
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline

## Argoverse 2

from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType

# Global variables 

StaticMapElements = Dict[str, List[Any]]

#######################################

# Aux functions

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.sqrt(np.sum((array-value)**2,axis=1)).argmin()
    return idx

# Main classes

class MapFeaturesUtils:
    """Utils for computation of map-based features."""
    def __init__(self, static_map_path):
        """Initialize class."""
        self._MANHATTAN_THRESHOLD = 10.0
        self._DFS_THRESHOLD_FRONT_SCALE = 45.0
        self._DFS_THRESHOLD_BACK_SCALE = 40.0
        self._MAX_SEARCH_RADIUS_CENTERLINES = 50.0 # 50.0
        self._MAX_CENTERLINE_CANDIDATES_TEST = 6
        self._INTERPOLATE_CENTERLINE_POINTS = 40
        
        self.map_json = ScenarioMap(static_map_path, self._INTERPOLATE_CENTERLINE_POINTS) # Auxiliar Argo2 map with the heuristic of Argo1

    def get_point_in_polygon_score(self, 
                                   lane_seq,
                                   xy_seq,
                                   avm) -> int:
        """Get the number of coordinates that lie insde the lane seq polygon.

        Args:
            lane_seq: Sequence of lane ids
            xy_seq: Trajectory coordinates
            avm: Argoverse map_api instance
        Returns:
            point_in_polygon_score: Number of coordinates in the trajectory that lie within the
            lane sequence
        """
        lane_seq_polygon = unary_union([
            Polygon(self.map_json.get_lane_segment_polygon(lane)).buffer(0)
            for lane in lane_seq if len(self.map_json.get_lane_segment_centerline(lane)) > 1
        ])
        point_in_polygon_score = 0
        for xy in xy_seq:        
            point_in_polygon_score += lane_seq_polygon.contains(Point(xy))
        return point_in_polygon_score

    def sort_lanes_based_on_point_in_polygon_score(
            self,
            lane_seqs,
            xy_seq,
            avm,
    ):
        """Filter lane_seqs based on the number of coordinates inside the bounding polygon of lanes.

        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            avm: Argoverse map_api instance
        Returns:
            sorted_lane_seqs: Sequences of lane sequences sorted based on the point_in_polygon score

        """
        point_in_polygon_scores = []
        for lane_seq in lane_seqs:
            point_in_polygon_scores.append(
                self.get_point_in_polygon_score(lane_seq, xy_seq, avm))
            
        randomized_tiebreaker = np.random.random(len(point_in_polygon_scores))
        sorted_point_in_polygon_scores_idx = np.lexsort(
            (randomized_tiebreaker, np.array(point_in_polygon_scores)))[::-1]
        sorted_lane_seqs = [
            lane_seqs[i] for i in sorted_point_in_polygon_scores_idx
        ]
        sorted_scores = [
            point_in_polygon_scores[i]
            for i in sorted_point_in_polygon_scores_idx
        ]
        return sorted_lane_seqs, sorted_scores

    def get_heuristic_centerlines_for_test_set(
            self,
            lane_seqs,
            xy_seq,
            avm,
            max_candidates,
            scores,
    ):
        """Sort based on distance along centerline and return the centerlines.
        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            avm: Argoverse map_api instance
            max_candidates: Maximum number of centerlines to return
        Return:
            sorted_candidate_centerlines: Centerlines in the order of their score
        """
        aligned_centerlines = []
        diverse_centerlines = []
        diverse_scores = []

        # Get first half as aligned centerlines
        
        aligned_cl_count = 0
        for i in range(len(lane_seqs)):
            lane_seq = lane_seqs[i]
            score = scores[i]
            diverse = True
            centerline = self.map_json.get_cl_from_lane_seq([lane_seq])[0] # Index 0 since the list only has 1 element here

            if aligned_cl_count < int(max_candidates / 2):
                start_dist = LineString(centerline).project(Point(xy_seq[0]))
                end_dist = LineString(centerline).project(Point(xy_seq[-1]))
                if end_dist > start_dist:
                    aligned_cl_count += 1
                    aligned_centerlines.append(centerline)
                    diverse = False
            if diverse:
                diverse_centerlines.append(centerline)
                diverse_scores.append(score)

        num_diverse_centerlines = min(len(diverse_centerlines),
                                      max_candidates - aligned_cl_count)

        test_centerlines = aligned_centerlines
        if num_diverse_centerlines > 0:
            probabilities = ([
                float(score + 1) / (sum(diverse_scores) + len(diverse_scores))
                for score in diverse_scores
            ] if sum(diverse_scores) > 0 else [1.0 / len(diverse_scores)] *
                             len(diverse_scores))
            diverse_centerlines_idx = np.random.choice(
                range(len(probabilities)),
                num_diverse_centerlines,
                replace=False,
                p=probabilities,
            )
            diverse_centerlines = [
                diverse_centerlines[i] for i in diverse_centerlines_idx
            ]

            test_centerlines += diverse_centerlines

        return test_centerlines
    
    def get_heuristic_hdmap_info_for_test_set(
            self,
            lane_seqs,
            xy_seq,
            avm,
            max_candidates,
            scores,
    ):
        """Sort based on distance along centerline and return the centerlines.

        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            avm: Argoverse map_api instance
            max_candidates: Maximum number of centerlines to return
        Return:
            sorted_candidate_hdmap_info: HDMap info in the order of their score

        """
        aligned_hdmap = []
        diverse_hdmap = []
        diverse_scores = []

        # Get first half as aligned centerlines
        
        aligned_cl_count = 0
        for i in range(len(lane_seqs)):
            lane_seq = lane_seqs[i]
            score = scores[i]
            diverse = True

            hdmap_info = self.map_json.get_cl_and_boundaries_from_lane_seq([lane_seq])[0]
            centerline = hdmap_info["centerline"]
            
            if aligned_cl_count < int(max_candidates / 2):
                start_dist = LineString(centerline).project(Point(xy_seq[0]))
                end_dist = LineString(centerline).project(Point(xy_seq[-1]))
                if end_dist > start_dist:
                    aligned_cl_count += 1
                    aligned_hdmap.append(hdmap_info)
                    diverse = False
            if diverse:
                diverse_hdmap.append(hdmap_info)
                diverse_scores.append(score)

        num_diverse_centerlines = min(len(diverse_hdmap),
                                      max_candidates - aligned_cl_count)

        test_hdmap = aligned_hdmap
        
        if num_diverse_centerlines > 0:
            probabilities = ([
                float(score + 1) / (sum(diverse_scores) + len(diverse_scores))
                for score in diverse_scores
            ] if sum(diverse_scores) > 0 else [1.0 / len(diverse_scores)] *
                             len(diverse_scores))
            diverse_centerlines_idx = np.random.choice(
                range(len(probabilities)),
                num_diverse_centerlines,
                replace=False,
                p=probabilities,
            )
            diverse_hdmap = [
                diverse_hdmap[i] for i in diverse_centerlines_idx
            ]

            test_hdmap += diverse_hdmap

        return test_hdmap

    def get_candidate_centerlines_for_trajectory(
            self,
            filename,
            tracks,
            derivatives,
            map_origin,
            avm,
            max_search_radius: float = 50.0,
            max_candidates: int = 10,
            mode: str = "test",
            algorithm: str = "map_api",
            time_variables: list = [50,60,10], # obs_len, pred_len, frequency
            min_dist_around: float = 15,
            normalize_rotation: str = "not_apply",
            scene_yaw: int = None,
            relative_displacements: bool = False,
            agent_index: int = -1,
            debug: bool = True
    ) -> List[np.ndarray]:
        """Get centerline candidates upto a threshold.

        General algorithm:
        1. Take the lanes in the bubble of last observed coordinate
        2. Extend before and after considering all possible candidates
        3. Get centerlines based on point in polygon score.

        Args:
            filename,
            tracks: [agent_track,extended_agent_track]
            derivates: [vel,acc,yaw]
            map_origin:
            avm: Argoverse map_api instance, 
            viz: Visualize candidate centerlines, 
            max_search_radius: Max search radius for finding nearby lanes in meters,
            max_candidates: Maximum number of centerlines to return, 
            mode: train/val/test mode
            time_variables:
            min_dist_around:
            normalize_rotation:
            relative_displacements:
            debug:
        Returns:
            candidate_centerlines: List of candidate centerlines

        """
        
        max_search_radius = self._MAX_SEARCH_RADIUS_CENTERLINES
        
        # 1. Preprocess observation data
        
        obs_len, pred_len, frequency = time_variables
        
        full_xy, xy_filtered, extended_xy_filtered = tracks

        debug = True
        if debug: # If debug, compute the centerlines with only the observation data 
                  # (even with train and val)
            xy = full_xy[:obs_len,:2]

        vel, acc, yaw = derivatives

        ## Estimate travelled distance
        
        dist_around = vel * (pred_len / frequency) + 1/2 * acc * (pred_len / frequency)**2
        
        if dist_around < min_dist_around:
            dist_around = min_dist_around
        
        ## Roughly compute the distance travelled with naive polynomial extension

        distance_travelled = 0
        max_dist = 100 # Hypothesis: max distance in 3 s

        index_max_dist = -1

        for i in range(extended_xy_filtered.shape[0]-1):
            if i >= obs_len:
                dist = np.linalg.norm((extended_xy_filtered[i+1,:] - extended_xy_filtered[i,:]))
                distance_travelled += dist

                if distance_travelled > max_dist and index_max_dist == -1:
                    index_max_dist = i

        reference_point = extended_xy_filtered[index_max_dist,:] # Reference point assuming naive prediction
        # reference_point = extended_xy_filtered[obs_len-1,:] # Reference point assuming last observation
        
        # 2. Get centerlines using the corresponding algorithm
        
        if algorithm == "competition":
            # Get all lane candidates within a bubble
            
            curr_lane_candidates = self.map_json.get_lane_ids_in_xy_bbox(
                xy_filtered[-1, 0], xy_filtered[-1, 1], self._MANHATTAN_THRESHOLD)

            # Keep expanding the bubble until at least 1 lane is found
            
            while (len(curr_lane_candidates) < 1
                and self._MANHATTAN_THRESHOLD < max_search_radius):
                self._MANHATTAN_THRESHOLD *= 2
                curr_lane_candidates = self.map_json.get_lane_ids_in_xy_bbox(
                    xy_filtered[-1, 0], xy_filtered[-1, 1], self._MANHATTAN_THRESHOLD)
                
            try:
                assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"
            except:
                while (len(curr_lane_candidates) < 1
                    and self._MANHATTAN_THRESHOLD < max_search_radius*100):
                    self._MANHATTAN_THRESHOLD *= 2
                    curr_lane_candidates = self.map_json.get_lane_ids_in_xy_bbox(
                        xy_filtered[-1, 0], xy_filtered[-1, 1], self._MANHATTAN_THRESHOLD)
                try:
                    assert (len(curr_lane_candidates) > 0)
                except:
                    while (len(curr_lane_candidates) < 1 and self._MANHATTAN_THRESHOLD < max_search_radius*500):
                        self._MANHATTAN_THRESHOLD *= 2
                        curr_lane_candidates = self.map_json.get_lane_ids_in_xy_bbox(
                            xy_filtered[-1, 0], xy_filtered[-1, 1], self._MANHATTAN_THRESHOLD)
            assert (len(curr_lane_candidates) > 0)
            
            # Set dfs threshold
            
            # dfs_threshold_front = 150.0
            # dfs_threshold_back = 150.0
            dfs_threshold_front = dfs_threshold_back = dist_around

            # DFS to get all successor and predecessor candidates
            
            obs_pred_lanes = [] # NOQA
            for lane in curr_lane_candidates:
                candidates_future = self.map_json.dfs(lane, 0,
                                            dfs_threshold_front)
                candidates_past = self.map_json.dfs(lane, 0, dfs_threshold_back,
                                        True)

                # Merge past and future
                for past_lane_seq in candidates_past:
                    for future_lane_seq in candidates_future:
                        assert (
                            past_lane_seq[-1] == future_lane_seq[0]
                        ), "Incorrect DFS for candidate lanes past and future"
                        obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])
                        
            # Removing overlapping lanes
            
            obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)
            
            # Sort lanes based on point in polygon score
            
            obs_pred_lanes, scores = self.sort_lanes_based_on_point_in_polygon_score(
                obs_pred_lanes, xy_filtered, avm)

            # If the best centerline is not along the direction of travel, re-sort

            candidate_hdmap_info = self.get_heuristic_hdmap_info_for_test_set(
                obs_pred_lanes, xy_filtered, avm, max_candidates, scores)

            # (Optional) Sort centerlines based on the distance to a reference point, usually the last observation

            candidate_centerlines = [hdmap_info["centerline"] for hdmap_info in candidate_hdmap_info]
            distances = []
            for centerline in candidate_centerlines:
                distances.append(min(np.linalg.norm((centerline - reference_point),axis=1)))
        
            AVOID_SAME_MIN_DISTANCES = False
            
            if AVOID_SAME_MIN_DISTANCES:
                # Avoid repeating centerlines with the same min distance
            
                unique_distances = list(set(distances))
                unique_distances.sort()
                unique_distances = unique_distances[:max_candidates]

                final_indeces = [np.where(distances == unique_distance)[0][0] for unique_distance in unique_distances]

                final_hdmap_info = []
                for index in final_indeces:
                    final_hdmap_info.append(candidate_hdmap_info[index])
            else:
                sorted_indeces = np.argsort(distances)
                final_hdmap_info = []
                for index in sorted_indeces:
                    final_hdmap_info.append(candidate_hdmap_info[index])
                
            candidate_hdmap_info = final_hdmap_info
        
        # TODO: Refactorize using HDMap info instead of just centerlines
        # elif algorithm == "map_api": 
        # # Compute centerlines using Argoverse Map API

        #     # Get all lane candidates within a bubble
            
        #     curr_lane_candidates = self.map_json.get_lane_ids_in_xy_bbox(
        #         xy[-1, 0], xy[-1, 1], self._MANHATTAN_THRESHOLD)

        #     # Keep expanding the bubble until at least 1 lane is found
            
        #     while (len(curr_lane_candidates) < 1
        #         and self._MANHATTAN_THRESHOLD < max_search_radius):
        #         self._MANHATTAN_THRESHOLD *= 2
        #         curr_lane_candidates = self.map_json.get_lane_ids_in_xy_bbox(
        #             xy_filtered[-1, 0], xy_filtered[-1, 1], self._MANHATTAN_THRESHOLD)
                
        #     try:
        #         assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"
        #     except:
        #         while (len(curr_lane_candidates) < 1
        #             and self._MANHATTAN_THRESHOLD < max_search_radius*100):
        #             self._MANHATTAN_THRESHOLD *= 2
        #             curr_lane_candidates = self.map_json.get_lane_ids_in_xy_bbox(
        #                 xy_filtered[-1, 0], xy_filtered[-1, 1], self._MANHATTAN_THRESHOLD)
        #         try:
        #             assert (len(curr_lane_candidates) > 0)
        #         except:
        #             while (len(curr_lane_candidates) < 1 and self._MANHATTAN_THRESHOLD < max_search_radius*500):
        #                 self._MANHATTAN_THRESHOLD *= 2
        #                 curr_lane_candidates = self.map_json.get_lane_ids_in_xy_bbox(
        #                     xy_filtered[-1, 0], xy_filtered[-1, 1], self._MANHATTAN_THRESHOLD)
        #     assert (len(curr_lane_candidates) > 0)

        #     # displacement = np.sqrt((xy[0, 0] - xy[obs_len-1, 0]) ** 2 + (xy[0, 1] - xy[obs_len-1, 1]) ** 2)
        #     # dfs_threshold = displacement * 2.0
        #     # dfs_threshold_front = dfs_threshold_back = dfs_threshold
                
        #     dfs_threshold_front = dist_around
        #     dfs_threshold_back = dist_around

        #     # DFS to get all successor and predecessor candidates
            
        #     obs_pred_lanes = [] # NOQA
        #     for lane in curr_lane_candidates:
        #         candidates_future = self.map_json.dfs(lane, 0,
        #                                     dfs_threshold_front)
        #         candidates_past = self.map_json.dfs(lane, 0, dfs_threshold_back,
        #                                 True)

        #         # Merge past and future
        #         for past_lane_seq in candidates_past:
        #             for future_lane_seq in candidates_future:
        #                 assert past_lane_seq[-1] == future_lane_seq[0], "Incorrect DFS for candidate lanes past and future"
        #                 obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

        #     # Removing overlapping lanes

        #     obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        #     # Remove unnecessary extended predecessors
            
        #     obs_pred_lanes = self.map_json.remove_extended_predecessors(obs_pred_lanes, xy_filtered)

        #     # Getting candidate centerlines
            
        #     candidate_cl = self.map_json.get_cl_from_lane_seq(obs_pred_lanes)

        #     # Reduce the number of candidates based on distance travelled along the centerline
            
        #     candidate_centerlines = filter_candidate_centerlines(xy_filtered, candidate_cl)

        #     # If no candidate found using above criteria, take the onces along with travel is the maximum
        #     if len(candidate_centerlines) < 1:
        #         candidate_centerlines = get_centerlines_most_aligned_with_trajectory(xy_filtered, candidate_cl)

        #     ## Additional ##

        #     # Sort centerlines based on the distance to a reference point, usually the last observation

        #     distances = []
        #     for centerline in candidate_centerlines:
        #         distances.append(min(np.linalg.norm((centerline - reference_point),axis=1)))
            
        #     AVOID_SAME_MIN_DISTANCES = True
            
        #     if AVOID_SAME_MIN_DISTANCES:
        #         # Avoid repeating centerlines with the same min distance
                
        #         unique_distances = list(set(distances))
        #         unique_distances.sort()
        #         unique_distances = unique_distances[:max_candidates]

        #         final_indeces = [np.where(distances == unique_distance)[0][0] for unique_distance in unique_distances]

        #         final_candidates = []
        #         for index in final_indeces:
        #             final_candidates.append(candidate_centerlines[index])
        #     else:
        #         sorted_indeces = np.argsort(distances)
        #         sorted_indeces = sorted_indeces[:max_candidates]
        #         final_candidates = []
        #         for index in sorted_indeces:
        #             final_candidates.append(candidate_centerlines[index])

        #     candidate_centerlines = final_candidates

        # 3. (Mandatory) Translate to the origin
        
        for candidate_hdmap_info_ in candidate_hdmap_info:
            candidate_hdmap_info_["centerline"] = candidate_hdmap_info_["centerline"] - map_origin    
            candidate_hdmap_info_["left_bound"] = candidate_hdmap_info_["left_bound"] - map_origin    
            candidate_hdmap_info_["right_bound"] = candidate_hdmap_info_["right_bound"] - map_origin    
            
        # 4. (Optional) Rotate centerlines w.r.t. focal agent last observation frame
        
        if normalize_rotation != "not_apply":
            if scene_yaw:
                yaw_aux = scene_yaw
            # TODO: Check this for single-agent prediction!!!!
            # elif normalize_rotation == "x-axis":
            #     yaw_aux = yaw
            # elif normalize_rotation == "y-axis":
            #     yaw_aux = - (math.pi/2 - yaw)
            # else:
            #     print("Rotation not specified")
            #     pdb.set_trace()
 
            R = rotz2D(yaw_aux)

            for candidate_hdmap_info_ in candidate_hdmap_info:
                candidate_hdmap_info_["centerline"] = apply_rotation(candidate_hdmap_info_["centerline"],R)
                candidate_hdmap_info_["left_bound"] = apply_rotation(candidate_hdmap_info_["left_bound"],R)
                candidate_hdmap_info_["right_bound"] = apply_rotation(candidate_hdmap_info_["right_bound"],R)

        # 5. Interpolate centerlines
        
        non_interp_candidate_hdmap_info = copy.deepcopy(candidate_hdmap_info)
        
        if self._INTERPOLATE_CENTERLINE_POINTS > 0:
            for candidate_hdmap_info_ in candidate_hdmap_info:
                candidate_hdmap_info_["centerline"] = centerline_interpolation(candidate_hdmap_info_["centerline"],
                                                                    interp_points=self._INTERPOLATE_CENTERLINE_POINTS)
                candidate_hdmap_info_["left_bound"] = centerline_interpolation(candidate_hdmap_info_["left_bound"],
                                                                    interp_points=self._INTERPOLATE_CENTERLINE_POINTS)
                candidate_hdmap_info_["right_bound"] = centerline_interpolation(candidate_hdmap_info_["right_bound"],
                                                                    interp_points=self._INTERPOLATE_CENTERLINE_POINTS)
        
        # 6. Recalculate type and intersection arrays
        
        for i in range(len(non_interp_candidate_hdmap_info)):
            ref_indeces = non_interp_candidate_hdmap_info[i]["centerline_length"]
            ref_indeces = np.cumsum(ref_indeces)[:] - 1  
            ref_points = non_interp_candidate_hdmap_info[i]["centerline"][ref_indeces,:] # end-point of each segment 
            left_ref_indeces = [find_nearest(non_interp_candidate_hdmap_info[i]["left_bound"],ref_point) for ref_point in ref_points]
            right_ref_indeces = [find_nearest(non_interp_candidate_hdmap_info[i]["right_bound"],ref_point) for ref_point in ref_points]
            
            # Reference points from the original geometry (centerline ref points are above)
            
            left_ref_points = non_interp_candidate_hdmap_info[i]["left_bound"][left_ref_indeces,:]
            right_ref_points = non_interp_candidate_hdmap_info[i]["right_bound"][right_ref_indeces,:]
            
            # Closest indeces to the current geometry (after interpolation)
            
            closest_indeces = [find_nearest(candidate_hdmap_info[i]["centerline"],ref_point) for ref_point in ref_points]
            left_closest_indeces = [find_nearest(candidate_hdmap_info[i]["left_bound"],left_ref_point) for left_ref_point in left_ref_points]
            right_closest_indeces = [find_nearest(candidate_hdmap_info[i]["right_bound"],right_ref_point) for right_ref_point in right_ref_points]
            
            candidate_hdmap_info[i]["centerline_type"] = np.zeros((self._INTERPOLATE_CENTERLINE_POINTS,candidate_hdmap_info[i]["centerline_type"].shape[1]))
            candidate_hdmap_info[i]["is_intersection"] = np.zeros((self._INTERPOLATE_CENTERLINE_POINTS,candidate_hdmap_info[i]["is_intersection"].shape[1]))
            candidate_hdmap_info[i]["left_type"] = np.zeros((self._INTERPOLATE_CENTERLINE_POINTS,candidate_hdmap_info[i]["left_type"].shape[1]))
            candidate_hdmap_info[i]["right_type"] = np.zeros((self._INTERPOLATE_CENTERLINE_POINTS,candidate_hdmap_info[i]["right_type"].shape[1]))
            
            pre_index = 0
            left_pre_index = 0
            right_pre_index = 0

            for j in range(len(closest_indeces)):
                index = closest_indeces[j] + 1
                left_index = left_closest_indeces[j] + 1
                right_index = right_closest_indeces[j] + 1
                
                candidate_hdmap_info[i]["centerline_type"][pre_index:index,:] = \
                    non_interp_candidate_hdmap_info[i]["centerline_type"][ref_indeces[j],:]
                candidate_hdmap_info[i]["is_intersection"][pre_index:index,:] = \
                    non_interp_candidate_hdmap_info[i]["is_intersection"][ref_indeces[j],:]
                    
                candidate_hdmap_info[i]["left_type"][left_pre_index:left_index,:] = \
                    non_interp_candidate_hdmap_info[i]["left_type"][left_ref_indeces[j],:]
                candidate_hdmap_info[i]["right_type"][right_pre_index:right_index,:] = \
                    non_interp_candidate_hdmap_info[i]["right_type"][right_ref_indeces[j],:]
                
                pre_index = index
                left_pre_index = left_index
                right_pre_index = right_index

            del candidate_hdmap_info[i]["centerline_length"] # Now this key is useless
          
        # 6. (Optional) Get relative displacements

        # rel_candidate_centerlines_array = np.array(candidate_centerlines)
        
        # if relative_displacements and candidate_centerlines:
        #     candidate_centerlines_array = np.array(candidate_centerlines)

        #     rel_candidate_centerlines_array = np.zeros(candidate_centerlines_array.shape) 
        #     rel_candidate_centerlines_array[:, 1:, :] = candidate_centerlines_array[:, 1:, :] - candidate_centerlines_array[:, :-1, :] # Get displacements between consecutive steps
        
        for candidate_hdmap_info_ in candidate_hdmap_info:
            candidate_hdmap_info_["rel_centerline"] = np.zeros(candidate_hdmap_info_["centerline"].shape)
            candidate_hdmap_info_["rel_left_bound"] = np.zeros(candidate_hdmap_info_["left_bound"].shape)
            candidate_hdmap_info_["rel_right_bound"] = np.zeros(candidate_hdmap_info_["right_bound"].shape)
            
            candidate_hdmap_info_["rel_centerline"] = candidate_hdmap_info_["centerline"][1:,:] - candidate_hdmap_info_["centerline"][:-1,:]
            candidate_hdmap_info_["rel_left_bound"] = candidate_hdmap_info_["left_bound"][1:,:] - candidate_hdmap_info_["left_bound"][:-1,:]
            candidate_hdmap_info_["rel_right_bound"] = candidate_hdmap_info_["right_bound"][1:,:] - candidate_hdmap_info_["right_bound"][:-1,:]
            
        # 7. Pad centerlines with zeros
        
        # pad_centerlines = True
        
        # if pad_centerlines and candidate_centerlines:
        #     # Determine if there are some repeated centerlines after filtering. Take the unique
        #     # elements. If after this there are less than max_centerlines, pad with zeros

        #     aux_array = copy.deepcopy(rel_candidate_centerlines_array)
        #     vals, idx_start, count = np.unique(rel_candidate_centerlines_array, axis=0, return_counts=True, return_index=True)
        #     rel_candidate_centerlines_array_aux = aux_array[np.sort(idx_start),:,:]
            
        #     centerline_points = 40
        #     data_dim = 2
        #     pad_zeros_centerlines = np.zeros((max_candidates-rel_candidate_centerlines_array_aux.shape[0],centerline_points,data_dim))
        #     rel_candidate_centerlines_array = np.vstack((rel_candidate_centerlines_array_aux,pad_zeros_centerlines))
        # if pad_centerlines:
        #     print(" ")
        # return candidate_centerlines, rel_candidate_centerlines_array
        
        return candidate_hdmap_info
    
class ScenarioMap:
    def __init__(self, root, interpolate_centerline_points):
        """Initialize the Argoverse Map 2 using functions of the original Argoverse Map 1."""
        
        """ TODO: Integrate the StaticMap functions here in order to have only two map objects,
        avm (Argoverse Map API 1) and scenario_map (Argoverse Map API 2, including these functions
        and the Static Map functions"""
        
        self.root = root
        self.render_window_radius = 150
        self.im_scale_factor = 50
        self._INTERPOLATE_CENTERLINE_POINTS = interpolate_centerline_points
        
        self.data = load_static_map_json(root)
        self.city_lane_centerlines_dict, self.predecessors_dict, self.successors_dict = self.build_centerline_index()
        (
            self.city_halluc_bbox_table,
            self.city_halluc_tableidx_to_laneid_map,
        ) = self.build_hallucinated_lane_bbox_index()

        # get hallucinated lane extends and driveable area from binary img
        self.city_to_lane_polygons_dict: Mapping[str, np.ndarray] = {}
        self.city_to_driveable_areas_dict: Mapping[str, np.ndarray] = {}
        self.city_to_lane_bboxes_dict: Mapping[str, np.ndarray] = {}
        self.city_to_da_bboxes_dict: Mapping[str, np.ndarray] = {}

        # for city_name in self.city_name_to_city_id_dict.keys():
        #     lane_polygons = np.array(self.get_vector_map_lane_polygons(city_name), dtype=object)
        #     driveable_areas = np.array(self.get_vector_map_driveable_areas(city_name), dtype=object)
        #     lane_bboxes = compute_polygon_bboxes(lane_polygons)
        #     da_bboxes = compute_polygon_bboxes(driveable_areas)

        #     self.city_to_lane_polygons_dict[city_name] = lane_polygons
        #     self.city_to_driveable_areas_dict[city_name] = driveable_areas
        #     self.city_to_lane_bboxes_dict[city_name] = lane_bboxes
        #     self.city_to_da_bboxes_dict[city_name] = da_bboxes

    def build_centerline_index(self):
        """
        Build dictionary of centerline for each city, with lane_id as key
        Returns:
            city_lane_centerlines_dict:  Keys are city names, values are dictionaries
                                        (k=lane_id, v=lane info)
        """

        city_lane_centerlines_dict = {}
        predecessors_dict = defaultdict(list)
        successors_dict = defaultdict(list)
        for id, lane in self.data['lane_segments'].items():
            city_lane_centerlines_dict[lane['id']] = lane
            for x in lane['successors']:
                successors_dict[lane['id']].append(x)
                predecessors_dict[x].append(lane['id'])
        return city_lane_centerlines_dict, predecessors_dict, successors_dict
    
    def build_hallucinated_lane_bbox_index(self):
        """
        Populate the pre-computed hallucinated extent of each lane polygon, to allow for fast
        queries.
        Returns:
            city_halluc_bbox_table
            city_id_to_halluc_tableidx_map
        """

        city_halluc_bbox_table = []
        city_halluc_tableidx_to_laneid_map = {}

        for id, lane in self.data['lane_segments'].items():
            left_lane_xy_start = np.array([lane['left_lane_boundary'][0]['x'], lane['left_lane_boundary'][0]['y']])
            left_lane_xy_end = np.array([lane['left_lane_boundary'][-1]['x'], lane['left_lane_boundary'][-1]['y']])
            right_lane_xy_start = np.array([lane['right_lane_boundary'][0]['x'], lane['right_lane_boundary'][0]['y']])
            right_lane_xy_end = np.array([lane['right_lane_boundary'][-1]['x'], lane['right_lane_boundary'][-1]['y']])
            area_1 = np.abs(left_lane_xy_start[0] - right_lane_xy_end[0]) * np.abs(left_lane_xy_start[1] - right_lane_xy_end[1])
            area_2 = np.abs(right_lane_xy_start[0] - left_lane_xy_end[0]) * np.abs(right_lane_xy_start[1] - left_lane_xy_end[1])
            city_halluc_tableidx_to_laneid_map[str(len(city_halluc_bbox_table))] = lane['id']
            if area_1 > area_2:
                city_halluc_bbox_table.append([left_lane_xy_start[0], left_lane_xy_start[1], right_lane_xy_end[0], right_lane_xy_end[1]])
            else:
                city_halluc_bbox_table.append([right_lane_xy_start[0], right_lane_xy_start[1], left_lane_xy_end[0], left_lane_xy_end[1]])
        city_halluc_bbox_table = np.array(city_halluc_bbox_table)  
        return city_halluc_bbox_table, city_halluc_tableidx_to_laneid_map

    def get_lane_ids_in_xy_bbox(
        self,
        query_x,
        query_y,
        query_search_range_manhattan = 5.0,
    ):
        """
        Prune away all lane segments based on Manhattan distance. We vectorize this instead
        of using a for-loop. Get all lane IDs within a bounding box in the xy plane.
        This is a approximation of a bubble search for point-to-polygon distance.
        The bounding boxes of small point clouds (lane centerline waypoints) are precomputed in the map.
        We then can perform an efficient search based on manhattan distance search radius from a
        given 2D query point.
        We pre-assign lane segment IDs to indices inside a big lookup array, with precomputed
        hallucinated lane polygon extents.
        Args:
            query_x: representing x coordinate of xy query location
            query_y: representing y coordinate of xy query location
            query_search_range_manhattan: search radius along axes
        Returns:
            lane_ids: lane segment IDs that live within a bubble
        """
        query_min_x = query_x - query_search_range_manhattan
        query_max_x = query_x + query_search_range_manhattan
        query_min_y = query_y - query_search_range_manhattan
        query_max_y = query_y + query_search_range_manhattan

        overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(
            self.city_halluc_bbox_table,
            np.array([query_min_x, query_min_y, query_max_x, query_max_y]),
        )

        if len(overlap_indxs) == 0:
            return []

        neighborhood_lane_ids: List[int] = []
        for overlap_idx in overlap_indxs:
            lane_segment_id = self.city_halluc_tableidx_to_laneid_map[str(overlap_idx)]
            neighborhood_lane_ids.append(lane_segment_id)

        return neighborhood_lane_ids

    def get_lane_segment_predecessor_ids(self, lane_segment_id):
        """
        Get land id for the lane predecessor of the specified lane_segment_id
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
        Returns:
            predecessor_ids: list of integers, representing lane segment IDs of predecessors
        """
        predecessor_ids = self.predecessors_dict[lane_segment_id]
        return predecessor_ids

    def get_lane_segment_successor_ids(self, lane_segment_id):
        """
        Get land id for the lane sucessor of the specified lane_segment_id
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
        Returns:
            successor_ids: list of integers, representing lane segment IDs of successors
        """
        successor_ids = self.successors_dict[lane_segment_id]
        return successor_ids

    def get_lane_segment_centerline(self, lane_segment_id):
        """
        We return a 3D centerline for any particular lane segment.
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
        Returns:
            lane_centerline: Numpy array of shape (N,3)
        """

        try:
            lane_centerline = np.array([[centerline['x'], centerline['y'], centerline['z']] for centerline in self.city_lane_centerlines_dict[lane_segment_id]['centerline']])
            return lane_centerline
        except:
            return None
        
    def get_lane_segment_centerline_and_boundaries(self, lane_segment_id):
        """
        We return a 3D centerline for any particular lane segment.
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
        Returns:
            Whole HDMap information
        """

        try:
            lane_centerline_array = np.array([[centerline['x'], centerline['y'], centerline['z']] for centerline in self.city_lane_centerlines_dict[lane_segment_id]['centerline']])
            num_points = lane_centerline_array.shape[0]
            
            lane_type = self.city_lane_centerlines_dict[lane_segment_id]['lane_type']
            lane_type_array = self.get_lanetype(num_points, lane_type)
            is_intersection = self.city_lane_centerlines_dict[lane_segment_id]['is_intersection']
            is_intersection_array = self.get_lane_in_intersection(num_points, is_intersection)
            
            left_boundary = np.array([[waypoint['x'], waypoint['y'], waypoint['z']] for waypoint in self.city_lane_centerlines_dict[lane_segment_id]['left_lane_boundary']])
            right_boundary = np.array([[waypoint['x'], waypoint['y'], waypoint['z']] for waypoint in self.city_lane_centerlines_dict[lane_segment_id]['right_lane_boundary']])
                        
            left_line_type = self.city_lane_centerlines_dict[lane_segment_id]['left_lane_mark_type']
            left_line_type_array = self.get_mark_type(left_boundary.shape[0], left_line_type)
            right_line_type = self.city_lane_centerlines_dict[lane_segment_id]['right_lane_mark_type']  
            right_line_type_array = self.get_mark_type(right_boundary.shape[0], right_line_type)

            return lane_centerline_array, lane_type_array, is_intersection_array,\
                   left_boundary, right_boundary, left_line_type_array, right_line_type_array
                   
        except:
            pdb.set_trace()
            return None
        
    def get_lane_segment_polygon(self, lane_segment_id: int, eps=1.0) -> np.ndarray:
        """
        Hallucinate a 3d lane polygon based around the centerline. We rely on the average
        lane width within our cities to hallucinate the boundaries. We rely upon the
        rasterized maps to provide heights to points in the xy plane.
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
        Returns:
            lane_polygon: Array of polygon boundary (K,3), with identical and last boundary points
        """
        lane_centerline = self.get_lane_segment_centerline(lane_segment_id)
        lane_polygon = centerline_to_polygon(lane_centerline[:, :2])
        return np.hstack([lane_polygon, np.zeros(lane_polygon.shape[0])[:, np.newaxis] + np.mean(lane_centerline[:, 2])])

    def get_lane_segments_containing_xy(self, query_x: float, query_y: float) -> List[int]:
        """

        Get the occupied lane ids, i.e. given (x,y), list those lane IDs whose hallucinated
        lane polygon contains this (x,y) query point.

        This function performs a "point-in-polygon" test.

        Args:
            query_x: representing x coordinate of xy query location
            query_y: representing y coordinate of xy query location
        Returns:
            occupied_lane_ids: list of integers, representing lane segment IDs containing (x,y)
        """
        neighborhood_lane_ids = self.get_lane_ids_in_xy_bbox(query_x, query_y)

        occupied_lane_ids: List[int] = []
        if neighborhood_lane_ids is not None:
            for lane_id in neighborhood_lane_ids:
                lane_polygon = self.get_lane_segment_polygon(lane_id)
                inside = point_inside_polygon(
                    lane_polygon.shape[0],
                    lane_polygon[:, 0],
                    lane_polygon[:, 1],
                    query_x,
                    query_y,
                )
                if inside:
                    occupied_lane_ids += [lane_id]
        return occupied_lane_ids
    
    def get_lane_in_intersection(self, lane_points, is_intersection):
        return is_intersection * np.ones(lane_points, np.float32)
    
    def get_lanetype(self, lane_points, lane_type):
        x = np.zeros((lane_points,2), np.float32)
        if lane_type == LaneType.VEHICLE:
            x[:, :] = 1
        elif lane_type == LaneType.BUS:
            x[:, 0] = 1
            x[:, 1] = 0
        elif lane_type == LaneType.BIKE:
            x[:, 0] = 0
            x[:, 1] = 1
        else:
            x[:, :] = 0
        return x
            
    def get_mark_type(self, lane_points, mark_type):
        # 0 (Number of lines) -> None = 0, Single_line = 1, Double_line = 2
        # 1 (First line type, if apply) -> Dash = 0, Solid = 1
        # 2 (Second line type, if apply) -> Dash = 0, Solid = 1
        # 3 (Line(s) colour) -> White = 0, Yellow = 1, Blue = 2 
        x = np.zeros((lane_points,4), np.float32)
        if mark_type == LaneMarkType.DASH_SOLID_YELLOW: 
            # Two lines, one is dashed, the other is solid, both are yellow
            x[:, 0] = 2
            x[:, 1] = 0
            x[:, 2] = 1
            x[:, 3] = 1
        elif mark_type == LaneMarkType.DASH_SOLID_WHITE:
            x[:, 0] = 2
            x[:, 1] = 0
            x[:, 2] = 1
            x[:, 3] = 0
        elif mark_type == LaneMarkType.DASHED_WHITE:   
            x[:, 0] = 1
            x[:, 1] = 0
            x[:, 2] = 0
            x[:, 3] = 0
        elif mark_type == LaneMarkType.DASHED_YELLOW:   
            x[:, 0] = 1
            x[:, 1] = 0
            x[:, 2] = 0  
            x[:, 3] = 1
        elif mark_type == LaneMarkType.DOUBLE_SOLID_YELLOW:
            x[:, 0] = 2
            x[:, 1] = 1
            x[:, 2] = 1
            x[:, 3] = 1
        elif mark_type == LaneMarkType.DOUBLE_SOLID_WHITE:   
            x[:, 0] = 2
            x[:, 1] = 1
            x[:, 2] = 1
            x[:, 3] = 0
        elif mark_type == LaneMarkType.DOUBLE_DASH_YELLOW:   
            x[:, 0] = 2
            x[:, 1] = 0
            x[:, 2] = 0  
            x[:, 3] = 1            
        elif mark_type == LaneMarkType.DOUBLE_DASH_WHITE:
            x[:, 0] = 2
            x[:, 1] = 0
            x[:, 2] = 0
            x[:, 3] = 0
        elif mark_type == LaneMarkType.SOLID_YELLOW:   
            x[:, 0] = 1
            x[:, 1] = 1
            x[:, 2] = 1
            x[:, 3] = 1
        elif mark_type == LaneMarkType.SOLID_WHITE:   
            x[:, 0] = 1
            x[:, 1] = 1
            x[:, 2] = 1  
            x[:, 3] = 0
        elif mark_type == LaneMarkType.SOLID_DASH_WHITE:
            x[:, 0] = 2
            x[:, 1] = 1
            x[:, 2] = 0
            x[:, 3] = 0
        elif mark_type == LaneMarkType.SOLID_DASH_YELLOW:   
            x[:, 0] = 2
            x[:, 1] = 1
            x[:, 2] = 0
            x[:, 3] = 1
        elif mark_type == LaneMarkType.SOLID_BLUE:   
            x[:, 0] = 1
            x[:, 1] = 1
            x[:, 2] = 1  
            x[:, 3] = 2 
        elif mark_type == LaneMarkType.UNKNOWN:
            x[:, 0] = 1
            x[:, 1] = 0
            x[:, 2] = 0  
            x[:, 3] = 0 
        elif mark_type == LaneMarkType.NONE:   
            x[:, 0] = 0
            x[:, 1] = 0
            x[:, 2] = 0  
            x[:, 3] = 0 
        return x
    
    def remove_extended_predecessors(
        self, lane_seqs: List[List[int]], xy: np.ndarray
    ) -> List[List[int]]:
        """
        Remove lane_ids which are obtained by finding way too many predecessors from lane sequences.
        If any lane id is an occupied lane id for the first coordinate of the trajectory, ignore all the
        lane ids that occured before that

        Args:
            lane_seqs: List of list of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
            xy: trajectory coordinates
        Returns:
            filtered_lane_seq (list of list of integers): List of list of lane ids obtained after filtering
        """
        filtered_lane_seq = []
        occupied_lane_ids = self.get_lane_segments_containing_xy(xy[0, 0], xy[0, 1])
        for lane_seq in lane_seqs:
            for i in range(len(lane_seq)):
                if lane_seq[i] in occupied_lane_ids:
                    new_lane_seq = lane_seq[i:]
                    break
                new_lane_seq = lane_seq
            filtered_lane_seq.append(new_lane_seq)
        return filtered_lane_seq
    
    def get_cl_from_lane_seq(self, lane_seqs: Iterable[List[int]]) -> List[np.ndarray]:
        """Get centerlines corresponding to each lane sequence in lane_sequences
        Args:
            lane_seqs: Iterable of sequence of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
        Returns:
            candidate_cl: list of numpy arrays for centerline corresponding to each lane sequence
        """

        candidate_cl = []
        for lanes in lane_seqs:
            curr_candidate_cl = np.empty((0, 2))
            for curr_lane in lanes:
                curr_candidate = self.get_lane_segment_centerline(curr_lane)[:, :2]
                curr_candidate_cl = np.vstack((curr_candidate_cl, curr_candidate))
            candidate_cl.append(curr_candidate_cl)
        return candidate_cl
    
    def get_cl_and_boundaries_from_lane_seq(self, lane_seqs: Iterable[List[int]]) -> List[np.ndarray]:
        """Get centerlines corresponding to each lane sequence in lane_sequences
        Args:
            lane_seqs: Iterable of sequence of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
        Returns:
            whole HD-map information related to the current lane seq
        """
        # TODO: Refactorize this function using lambdas
        
        hdmap_info = []
        for lanes in lane_seqs:
            sample = dict()
            
            cl_, cl_length_, cl_type_, is_inters_, left_bound_, right_bound_, left_type_, right_type_ = \
                ([] for _ in range(8))
                
            for curr_lane in lanes:
                cl, cl_type, is_inters, left_bound, right_bound, left_type, right_type = \
                    self.get_lane_segment_centerline_and_boundaries(curr_lane)

                cl_.append(cl[:, :2]) # Avoid z-axis
                cl_length_.append(cl.shape[0])
                cl_type_.append(cl_type)
                is_inters_.append(is_inters)
                left_bound_.append(left_bound[:, :2])
                right_bound_.append(right_bound[:, :2])
                left_type_.append(left_type)
                right_type_.append(right_type)

            sample["centerline"] = np.concatenate(cl_)
            sample["centerline_length"] = np.array(cl_length_)
            sample["centerline_type"] = np.concatenate(cl_type_)
            sample["is_intersection"] = np.concatenate(is_inters_).reshape(-1,1)
            sample["left_bound"] = np.concatenate(left_bound_)
            sample["right_bound"] = np.concatenate(right_bound_)
            sample["left_type"] = np.concatenate(left_type_)
            sample["right_type"] = np.concatenate(right_type_)
            
            hdmap_info.append(sample)
        return hdmap_info

    def dfs(
        self,
        lane_id: int,
        dist: float = 0,
        threshold: float = 30,
        extend_along_predecessor: bool = False,
    ) -> List[List[int]]:
        """
        Perform depth first search over lane graph up to the threshold.
        Args:
            lane_id: Starting lane_id (Eg. 12345)
            dist: Distance of the current path
            threshold: Threshold after which to stop the search
            extend_along_predecessor: if true, dfs over predecessors, else successors
        Returns:
            lanes_to_return (list of list of integers): List of sequence of lane ids
                Eg. [[12345, 12346, 12347], [12345, 12348]]
        """
        if dist > threshold:
            return [[lane_id]]
        else:
            traversed_lanes = []
            child_lanes = (
                self.get_lane_segment_predecessor_ids(lane_id)
                if extend_along_predecessor
                else self.get_lane_segment_successor_ids(lane_id)
            )
            if child_lanes is not None:
                for child in child_lanes:
                    centerline = self.get_lane_segment_centerline(child)
                    if centerline is not None:
                        try:
                            cl_length = LineString(centerline).length
                            curr_lane_ids = self.dfs(
                                child,
                                dist + cl_length,
                                threshold,
                                extend_along_predecessor,
                            )
                            traversed_lanes.extend(curr_lane_ids)
                        except:
                            pass

            if len(traversed_lanes) == 0:
                return [[lane_id]]
            lanes_to_return = []
            for lane_seq in traversed_lanes:
                lanes_to_return.append(lane_seq + [lane_id] if extend_along_predecessor else [lane_id] + lane_seq)
            return lanes_to_return
        
def load_static_map_json(static_map_path: Path) -> StaticMapElements:
    """Load a saved static map from disk.
    Args:
        static_map_path: Path to the saved static map.
    Returns:
        Object representation for the static map elements saved at `static_map_path`.
    """
    with open(static_map_path, "rb") as f:
        static_map_elements = json.load(f)

    return cast(StaticMapElements, static_map_elements)

def get_agent_velocity_and_acceleration(agent_seq, period=0.1, filter_traj="least_squares",debug=False):
    """
    Consider the observation data to calculate an average velocity and 
    acceleration of the agent in the last observation point
    """

    # https://en.wikipedia.org/wiki/Speed_limits_in_the_United_States_by_jurisdiction
    # 1 miles per hour (mph) ~= 1.609 kilometers per hour (kph)
    # Common speed limits in the USA (in highways): 70 - 80 mph -> (112 kph - 129 kph) -> (31.29 m/s - 35.76 m/s)
    # The average is around 120 kph -> 33.33 m/s, so if the vehicle has accelerated strongly (in the last observation has reached the 
    # maximum velocity), we assume the GT will be without acceleration, that is, braking or stop accelerating positively 
    # (e.g. Seq 188893 in train). 
    # The maximum prediction horizon should be around 100 m.

    #                            | (Limit of observation)
    #                            v
    # ... . . .  .  .  .    .    .    .    .    .     .     .     .     .     .     .     .     . (Good interpretation)

    # [     Observation data     ][                       Groundtruth data                      ]  

    # ... . . .  .  .  .    .    .    .     .     .      .       .       .       .       .        .        .          . (Wrong interpretation) 

    x = agent_seq[:,0]
    y = agent_seq[:,1]
    xy = np.vstack((x, y))

    extended_xy_f = np.array([])
    num_points_trajectory = agent_seq.shape[0]

    polynomial_order = 2

    t = np.linspace(1, num_points_trajectory, num_points_trajectory)
    px = np.poly1d(np.polyfit(t,x,polynomial_order))
    py = np.poly1d(np.polyfit(t,y,polynomial_order))

    xy_f = np.vstack((px(t),py(t)))

    seq_len = 50
    t2 = np.linspace(1, seq_len, seq_len)
    extended_xy_f = np.vstack((px(t2),py(t2)))

    obs_seq_f = xy_f

    vel_f = np.zeros((num_points_trajectory-1))
    for i in range(1,obs_seq_f.shape[1]):
        x_pre, y_pre = obs_seq_f[:,i-1]
        x_curr, y_curr = obs_seq_f[:,i]

        dist = math.sqrt(pow(x_curr-x_pre,2)+pow(y_curr-y_pre,2))

        curr_vel = dist / period
        vel_f[i-1] = curr_vel
    
    acc_f = np.zeros((num_points_trajectory-2))
    for i in range(1,len(vel_f)):
        vel_pre = vel_f[i-1]
        vel_curr = vel_f[i]

        delta_vel = vel_curr - vel_pre

        curr_acc = delta_vel / period
        acc_f[i-1] = curr_acc

    min_weight = 1
    max_weight = 4

    if filter_traj == "least_squares":
        # Theoretically, if the points are computed using Least Squares with a Polynomial with order >= 2,
        # the velocity in the last observation should be fine, since you are taking into account the 
        # acceleration (either negative or positive)
        vel_f_averaged = vel_f[-1]
    else:
        vel_f_averaged = np.average(vel_f,weights=np.linspace(min_weight,max_weight,len(vel_f))) 

    acc_f_averaged = np.average(acc_f,weights=np.linspace(min_weight,max_weight,len(acc_f)))
    acc_f_averaged_aux = acc_f_averaged
    if vel_f_averaged > 35 and acc_f_averaged > 5:
        acc_f_averaged = 0 # We assume the vehicle should not drive faster than 35 mps (meter per second)!
                            # To discard the acceleration data (35 mps = 126 kph)
                            # This is an assumption!

    if debug:
        print("Filter: ", filter)
        print("Min weight, max weight: ", min_weight, max_weight)
        print("vel averaged: ", vel_f_averaged)
        print("acc averaged: ", acc_f_averaged)
        
        pred_len = 60
        freq = 10

        dist_around_wo_acc = vel_f_averaged * (pred_len/freq)
        dist_around_acc = vel_f_averaged * (pred_len/freq) + 1/2 * acc_f_averaged_aux * (pred_len/freq)**2

        print("Estimated horizon without acceleration: ", dist_around_wo_acc)
        print("Estimated horizon with acceleration: ", dist_around_acc)

        if extended_xy_f.size > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.plot(
                    xy_f[0, :],
                    xy_f[1, :],
                    ".",
                    color="b",
                    alpha=1,
                    linewidth=3,
                    zorder=1,
                )

            for i in range(xy_f.shape[1]):
                plt.text(xy_f[0,i],xy_f[1,i],i)

            x_min = min(xy_f[0,:])
            x_max = max(xy_f[0,:])
            y_min = min(xy_f[1,:])
            y_max = max(xy_f[1,:])

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            ax2.plot(
                    extended_xy_f[0, :],
                    extended_xy_f[1, :],
                    ".",
                    color="r",
                    alpha=1,
                    linewidth=3,
                    zorder=1,
                )

            for i in range(extended_xy_f.shape[1]):
                plt.text(extended_xy_f[0,i],extended_xy_f[1,i],i)

            x_min = min(extended_xy_f[0,:])
            x_max = max(extended_xy_f[0,:])
            y_min = min(extended_xy_f[1,:])
            y_max = max(extended_xy_f[1,:])

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
        else:
            plt.plot(
                    xy_f[0, :],
                    xy_f[1, :],
                    ".",
                    color="b",
                    alpha=1,
                    linewidth=3,
                    zorder=1,
                )

            for i in range(xy_f.shape[1]):
                plt.text(xy_f[0,i],xy_f[1,i],i)

            x_min = min(xy_f[0,:])
            x_max = max(xy_f[0,:])
            y_min = min(xy_f[1,:])
            y_max = max(xy_f[1,:])

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

        plt.show()
        plt.close('all')

    xy_f = xy_f.T # TODO: Check the above xy_f and transform from 2 x 20 to 20 x 2 (easy to read)
    extended_xy_f = extended_xy_f.T

    return vel_f_averaged, acc_f_averaged, xy_f, extended_xy_f
    
def get_yaw(agent_xy, obs_len):
    """
    Assuming the agent observation has been filtered, determine the agent's orientation
    in the last observation
    
    Angles range from 0 (0º) to pi (180º), and -pi (180º) to -0 (360º)
    """

    lane_dir_vector = agent_xy[obs_len-1,:] - agent_xy[obs_len-2,:]
    yaw = math.atan2(lane_dir_vector[1],lane_dir_vector[0])

    return lane_dir_vector, yaw

def apply_tf(source_location, transform):  
    """_summary_

    Args:
        source_location (_type_): _description_
        transform (_type_): _description_

    Returns:
        _type_: _description_
    """
    centroid = np.array([0.0,0.0,0.0,1.0]).reshape(4,1)

    centroid[0,0] = source_location[0]
    centroid[1,0] = source_location[1]
    centroid[2,0] = source_location[2]

    target_location = np.dot(transform,centroid) # == transform @ centroid

    return target_location 

def rotz2D(yaw):
    """ 
    Rotation about the z-axis
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c,  -s],
                     [s,   c]])
    
def apply_rotation(tensor,R):
    """
    tensor: torch.tensor or np.array
    R: rotation matrix around Z-axis
    """

    # It is faster to compute the rotation using numpy (CPU) instead of torch (GPU)
    return np.matmul(tensor,R)

def centerline_interpolation(centerline,interp_points=40,debug=False):
    """_summary_

    Args:
        centerline (_type_): _description_
        interp_points (int, optional): _description_. Defaults to 40.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    try:
        cx, cy = centerline[:,0], centerline[:,1]
        points = np.arange(cx.shape[0])

        new_points = np.linspace(points.min(), points.max(), interp_points)
    
        new_cx = sp.interpolate.interp1d(points,cx,kind='cubic')(new_points)
        new_cy = sp.interpolate.interp1d(points,cy,kind='cubic')(new_points)
    except:
        if debug: pdb.set_trace()
        return []

    interp_centerline = np.hstack([new_cx.reshape(-1,1),new_cy.reshape(-1,1)])

    return interp_centerline


