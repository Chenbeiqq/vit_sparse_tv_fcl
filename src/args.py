import os
import random
import numpy as np
import argparse

import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    # DATASETS
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='CIFAR100',
    )
    parser.add_argument(
        "--eval_datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train_dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )

    # MODEL/TRAINING
    parser.add_argument(
        "--model",
        type=str,
        default='VIT-B-16',
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--client', type=int,default=3)
    # LOAD/SAVE PATHS
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='checkpoints/ViT-B-16/cachedir/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    
    # CL SPLITS
    parser.add_argument(
        "--n_splits",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default=None,
        choices=[None, 'data', 'class']
    )
    # parser.add_argument(
    #     "--sequential-finetuning",
    #     action='store_true'
    # )
    parser.add_argument(
        "--fl_magmax",
        type=str,
        default='True'
    )
    parser.add_argument(
        "--sequential-finetuning",
        type=str,
        default='True'
    )
    parser.add_argument(
        "--federated-continual",
        type=str,
        default='True'
    )
    # CL METHODS
    parser.add_argument(
        "--lwf_lamb",
        type=float,
        default=0.0,
        help="LWF lambda"
    )
    parser.add_argument(
        "--ewc_lamb",
        type=float,
        default=0.0,
        help="EWC lambda"
    )
    #FL
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-7,
        help='mask learning rate'
    )
    parser.add_argument(
        '--num_users',
        type=int,
        default=30,
        help="number of users: K")
    parser.add_argument('--iid',
        type=int,
        default=1,
        help='Default set to IID. Set to 0 for non-IID.')

    parser.add_argument('--alpha',
        default=6,
        type=int,
        help='quantity skew')

    parser.add_argument('--beta',
        default=0.1,
        type=float,
        help='distribution skew')
    parser.add_argument('--select_client',
        type=int,
        default=10,
        help='the number of clients in a local training: M')
    parser.add_argument('--niid_type',
        default='Q',
        type=str,
        help='Q or D')
    #Localized_and_stitch
    parser.add_argument(
        '--sigmoid_bias',
        default=5,
        type=int
    )
    parser.add_argument(
        '--l1_strength',
        default=1,
        type=int
    )
    parser.add_argument(
        '--num_train_epochs',
        default=10,
        type=int
    )
    parser.add_argument(
        '--graft_lr',
        default=1e-7,
        type=int
    )
    #稀疏度
    parser.add_argument(
        '--sparsity',
        default=1e-1,
        type=int
    )
    parser.add_argument(
        '--n_shot',
        default=64,
        type=int,
        help='create n_shot valset'
    )
    # OTHER
    parser.add_argument(
        '--seed',
        default=5,
        type=int
    )
    parser.add_argument(
        "--wandb_entity_name",
        type=str,
        default="chenbeiqi833-jiangxi-university-of-finance-and-economics"
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default='checkpoints/ViT-B-16/zeroshot.pt',
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(parsed_args.seed)
    
    assert parsed_args.lwf_lamb == 0.0 or parsed_args.ewc_lamb == 0.0, \
        "Lambda for LWF and EWC are mutually exclusive"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    return parsed_args
