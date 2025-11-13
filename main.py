import glob
import json
import os
import random
import time
from collections import defaultdict

def _append_xla_flag(flag: str) -> None:
    """Safely append an XLA flag before importing JAX."""
    current = os.environ.get('XLA_FLAGS', '')
    if flag not in current.split():
        os.environ['XLA_FLAGS'] = (current + ' ' + flag).strip()


# Disable Triton GEMM fusion to avoid GPU ptxas crashes when compiling large transformers.
_append_xla_flag('--xla_gpu_enable_triton_gemm=false')

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils import mlp_class

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'puzzle-4x5-play-oraclerep-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval.')
flags.DEFINE_integer('num_datasets', None, 'Number of datasets to use.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 5000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 10000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 250000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 5000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 15, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

### transformer
flags.DEFINE_string('mlp_class', 'MLP', 'What MLP class to use for agent') # MLP or TransformerMLP
flags.DEFINE_string('tx_variant', 'small', 'Transformer variant large or small') # Transformer variant

config_flags.DEFINE_config_file('agent', 'agents/sharsa.py', lock_config=False)

### profiler ###
import jax
# import jax.numpy as jnp
# # from jax import random, jit, grad
from jax.profiler import start_trace, stop_trace, trace
from pathlib import Path

def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='horizon-reduction', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and datasets.
    config = FLAGS.agent
    if FLAGS.dataset_dir is None:
        datasets = [None]
    else:
        # Dataset directory.
        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
    if FLAGS.num_datasets is not None:
        datasets = datasets[: FLAGS.num_datasets]
    dataset_idx = 0
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, dataset_path=datasets[dataset_idx])

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset_class_dict = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }
    dataset_class = dataset_class_dict[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]

    agent_mlp = mlp_class[FLAGS.mlp_class]

    if FLAGS.mlp_class == 'Transformer' : 
        config['batch_size'] = 1 # 1024 -> 256
        if FLAGS.tx_variant == 'small' : 
            config['actor_hidden_dims'] = (2048, 16, 128, 4, 128)
            config['value_hidden_dims'] = (2048, 16, 128, 4, 128)
        elif FLAGS.tx_variant == 'large' : 
            config['actor_hidden_dims'] = (2048, 8, 256, 10, 1024)
            config['value_hidden_dims'] = (2048, 8, 256, 10, 1024)

    config['batch_size'] = 1
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
        agent_mlp,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    logdir = Path(f"log/{config['agent_name']}")
    os.makedirs(logdir, exist_ok=True)
    start_trace(str(logdir))
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        # print(batch['observations'].shape, batch['actor_goals'].shape) # 69, 2 for humanmaze
        agent, update_info = agent.update(batch)
        if i == 1 :
            stop_trace()
        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            val_batch = val_dataset.sample(config['batch_size'])
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=agent,
                    env=env,
                    env_name=FLAGS.env_name,
                    goal_conditioned=True,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=5)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        if FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0 and len(datasets) > 1:
            dataset_idx = (dataset_idx + 1) % len(datasets)
            train_dataset, val_dataset = make_env_and_datasets(
                FLAGS.env_name, dataset_path=datasets[dataset_idx], dataset_only=True, cur_env=env
            )
            train_dataset = dataset_class(Dataset.create(**train_dataset), config)
            val_dataset = dataset_class(Dataset.create(**val_dataset), config)
    stop_trace()
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
