# pip install --upgrade calflops
import glob
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

# from calflops import calculate_flops
from agents import agents, gcsacbc
import ogbench 
import jax

if __name__ == "__main__" : 
    #random.seed(FLAGS.seed)
    #np.random.seed(FLAGS.seed)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    config = gcsacbc.get_config()
    env_name = "humanoidmaze-giant-navigate-oraclerep-v0"
    dataset_dir = "./dataset/humanoidmaze-giant-navigate-100m-v0" 
    datasets = [file for file in sorted(glob.glob(f'{dataset_dir}/*.npz')) if '-val.npz' not in file]
    dataset_idx = 0

    train_dataset, val_dataset = ogbench.make_env_and_datasets(
            env_name, dataset_path=datasets[dataset_idx], compact_dataset=True, dataset_only=True)
    eps = 1e-5
    val_dataset = Dataset.create(**val_dataset)
    val_dataset = val_dataset.copy(add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + eps, 1 - eps)))
    # env, train_dataset, val_dataset = make_env_and_datasets(env_name, dataset_path=datasets[dataset_idx])
   
    dataset_class_dict = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }
    dataset_class = dataset_class_dict[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    val_dataset = dataset_class(Dataset.create(**val_dataset), config)
    example_batch = train_dataset.sample(1)
    # print(example_batch['observations'].shape, example_batch['actor_goals'].shape)

    batch_size = 256
    obs_size = 69
    goal_size = 2
    input_shape = (batch_size, obs_size + goal_size)
    agent_class = agents[config['agent_name']]
    # from utils.networks import MLP
    from utils.transformer_mlp import TransformerMLP
    variant = 'small'
    if config['mlp_class'] == 'Transformer' : 
        if variant == 'small' : 
            config['actor_hidden_dims']=(2048, 16, 128, 4, 128)  # Actor network hidden dimensions. (small)
            config['value_hidden_dims']= (2048, 16, 128, 4, 128)  # Value network hidden dimensions. (small)
        elif variant == 'large' : 
            config['actor_hidden_dims']=(2048, 8, 256, 10, 1024) # Actor network hidden dimensions. (large)
            config['value_hidden_dims']=(2048, 8, 256, 10, 1024) # Value network hidden dimensions. (large)

    agent = agent_class.create(
            seed,
            example_batch,
            config,
        )

    #print(agent.network)

    model = agent.network
    #print(model.network.params)
    # for x in jax.tree_util.tree_leaves(model.params) :
    #     print(x.size)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(model.params))
    print(param_count)

    '''
    MLP : 16286767 (~17M)
    Transformer small : 3000623 (3M)
    Transformer large : 40498223 (~41M)
    '''

    # batch_size = 256
    # input_shape = (batch_size, 3, 224, 224) # change per environment
    # flops, macs, params = calculate_flops(model=model, 
    #                                     input_shape=input_shape,
    #                                     output_as_string=True,
    #                                     output_precision=4)
    # print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))



