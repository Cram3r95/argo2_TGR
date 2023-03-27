import pdb
from multiprocessing import Pool
import multiprocessing
import os
import argparse
import sys
import pickle
import os
import logging
import git

from tqdm import tqdm

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from model.models.TFMF_TGR import TMFModel

DATASET_DIR = "dataset/argoverse2"

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

log_dir = os.path.dirname(os.path.abspath(__file__))
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser = TMFModel.init_args(parser,BASE_DIR,DATASET_DIR)

parser.add_argument("--n_cpus", type=int, default=multiprocessing.cpu_count())
parser.add_argument("--chunksize", type=int, default=20)


def preprocess_dataset(dataset, n_cpus, chunksize):
    """Parallely preprocess a dataset to a pickle files

    Args:
        dataset: Dataset to be preprocessed
        n_cpus: Number of CPUs to use
        chunksize: Chunksize for parallelization
    """
    with Pool(n_cpus) as p:
        preprocessed = list(tqdm(p.imap(dataset.__getitem__, 
                                 [*range(len(dataset))], chunksize), total=len(dataset)))

    os.makedirs(os.path.dirname(dataset.input_preprocessed_social), exist_ok=True)
    with open(dataset.input_preprocessed_social, 'wb') as f:
        pickle.dump(preprocessed, f)

def main():
    args = parser.parse_args()

    args.use_preprocessed = False

    train_dataset = ArgoCSVDataset(args.train_split, args.train_split_pre_social, args)
    val_dataset = ArgoCSVDataset(args.val_split, args.val_split_pre_social, args)
    test_dataset = ArgoCSVDataset(args.test_split, args.test_split_pre_social, args)
    
    preprocess_dataset(train_dataset, args.n_cpus, args.chunksize)
    preprocess_dataset(val_dataset, args.n_cpus, args.chunksize)
    preprocess_dataset(test_dataset, args.n_cpus, args.chunksize)

if __name__ == "__main__":
    main()