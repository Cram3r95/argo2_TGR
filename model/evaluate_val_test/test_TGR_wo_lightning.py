import argparse
import os
import sys
import pdb
from tqdm import tqdm

import torch
from torch.nn import functional as F
from scipy.special import softmax
import numpy as np
import pandas as pd
from av2_validation.challange_submission import ChallengeSubmission

from torch.utils.data import DataLoader
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from argoverse.evaluation.competition_util import generate_forecasting_h5

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.TFMF_TGR_teacher import TMFModel ## from nn.Module
# from model.TFMF_TGR_student import TMFModel ## from nn.Module

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser = TMFModel.init_args(parser)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--ckpt_path", type=str, default="/path/to/checkpoint.ckpt")

def main():

    args = parser.parse_args()

    if args.split == "val":
        dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args, args.val_split_pre_map)
    else:
        dataset = ArgoCSVDataset(args.test_split, args.test_split_pre, args, args.test_split_pre_map)

    data_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        collate_fn=collate_fn_dict,
        shuffle=False,
        pin_memory=True,
    )

    # Load model with weights

    model = TMFModel(args)
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])#, strict=False)
    model.eval() # We do not want to train during inference

    # Iterate over dataset and generate predictions
    
    predictions = dict()
    gts = dict()
    cities = dict()
    probabilities = dict()
    final_out = dict()

    for data in tqdm(data_loader):
        data = dict(data)

        with torch.no_grad():
            output = model(data)
          
            output = [x[0:1].detach().cpu().numpy() for x in output]
        for i, (argo_id, prediction) in enumerate(zip(data["argo_id"], output)):
            # prediction.shape : (1,6,60,2) prediction.squeeze().shape(6,60,2)
            predictions[argo_id] = prediction.squeeze()
            sum_1 = np.sum(prediction.squeeze(),axis=1)
            sum_2 = np.sum(sum_1,axis=1)
            sotmax_out = softmax(sum_2) 
            sum_soft = np.sum(sotmax_out)
            if sum_soft > 1 :
                index_max = np.argmax(sotmax_out, axis=0)
                sotmax_out[index_max] = sotmax_out[index_max] - (sum_soft- 1 )
                
            if sum_soft < 1:
                index_min = np.argmin(sotmax_out, axis=0)
                sotmax_out[index_min] = sotmax_out[index_min] + ( 1 - sum_soft )
          
            probabilities[argo_id] = sotmax_out
            cities[argo_id] = data["city"][i]
            gts[argo_id] = data["gt"][i][0] if args.split == "val" else None
          
            # read parquet file and extract track_id for argo_id
            if args.split == "test":
                df = pd.read_parquet(args.test_split + '/' + argo_id + '/scenario_'+argo_id+'.parquet')
                track_id = df['focal_track_id'].values[0]
                track_id_dict = dict()
                track_id_dict[track_id] = [prediction.squeeze(),sotmax_out]
                final_out[argo_id] = track_id_dict
            
    # Evaluate or submit
    if args.split == "val":
        results_6 = compute_forecasting_metrics(
            predictions, gts, cities, 6, 60, 2, probabilities)
        results_1 = compute_forecasting_metrics(
            predictions, gts, cities, 1, 60, 2, probabilities)
        pdb.set_trace()
    else:
        chSubmission = ChallengeSubmission(final_out)
        chSubmission.to_parquet('submission/TGR.parquet')
   
if __name__ == "__main__":
    main()
    
"""

Important:

Please comment line 217 in the following file: https://github.com/argoai/argoverse-api/blob/master/argoverse/evaluation/eval_forecasting.py 
First, find the path of this file in your system after installing the Argoverse api_1 package. (pip3 install git+https://github.com/argoai/argoverse-api.git) 
Then comment the line 217 which uses the Map information to compute the DAC parameter.

In case you have installed argoverse-api after cloning the git repository (instead of pip install git+https...) go to:

"$HOME/argoverse-api/argoverse/evaluation/eval_forecasting.py", line 217" -> Comment this line

python test_TGR_wo_lightning.py --use_preprocessed=True \
                                --ckpt_path="/home/robesafe/Argoverse2_Motion_Forecasting/lightning_logs/version_9/checkpoints/epoch=9-loss_train=127.62-loss_val=173.21-ade1_val=2.93-fde1_val=7.12-ade_val=2.94-fde_val=6.79.ckpt" \
                                --split=val
"""
