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


def _append_xla_flag(flag: str) -> None:
    """Safely append an XLA flag before importing JAX."""
    current = os.environ.get('XLA_FLAGS', '')
    if flag not in current.split():
        os.environ['XLA_FLAGS'] = (current + ' ' + flag).strip()


# Disable Triton GEMM fusion to avoid GPU ptxas crashes when compiling large transformers.
_append_xla_flag('--xla_gpu_enable_triton_gemm=false')

# This script only inspects parameter counts, so run on CPU by default to avoid GPU-specific compiler issues.
if os.environ.get('CHECK_FLOPS_FORCE_CPU', '1') == '1':
    os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

# from calflops import calculate_flops
from agents import agents, gcsacbc
import ogbench 
import jax
from utils.transformer_mlp import TransformerMLP
from utils import mlp_class
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

    str_mlp_class = 'Transformer'
    variant = 'large'
    agent_mlp = mlp_class[str_mlp_class]

    if str_mlp_class == 'Transformer' : 
        config['batch_size'] = 1 # 1024 -> 256
        if variant == 'small' : 
            config['actor_hidden_dims'] = (2048, 16, 128, 4, 128)
            config['value_hidden_dims'] = (2048, 16, 128, 4, 128)
        elif variant == 'large' : 
            config['actor_hidden_dims'] = (2048, 8, 256, 10, 1024)
            config['value_hidden_dims'] = (2048, 8, 256, 10, 1024)

    config['batch_size'] = 1
    agent = agent_class.create(
        seed,
        example_batch,
        config,
        agent_mlp,
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
