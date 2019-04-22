"""
Run Prototypical Soft Actor Critic on HalfCheetahEnv.

"""
import numpy as np
import click
import datetime
import pathlib
import os

from rlkit.envs.ant_goal import AntGoalEnv

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.conv_networks import CNN, CNNEncoder
from rlkit.torch.sac.sac import ProtoSoftActorCritic
from rlkit.torch.sac.proto import ProtoAgent
import rlkit.torch.pytorch_util as ptu

import sys

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

sys.excepthook = info


import vizdoom

import numpy as np
import sys
sys.path.append('/home/jabreezus/clones2/meta-vizdoom/')
sys.path.append('/home/jabreezus/clones2/meta-vizdoom/ppo')


# vizdoom args and env from other code
import arguments
import env as grounding_env
from arguments import get_args
from subproc_vec_env import SubprocVecEnv
args = get_args("--num-processes 10 --algo ppo --max-episode-length 50 --difficulty hard2 --lr 0.0001 --fixed-env 0 --dense-dist-reward clipped  --frame-width 64".split())
# args.difficulty = 'hard2'
# n_proc = args.num_processes
print(args)



def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant):

    args.grayscale = variant['grayscale']
    def make_my_env(args, rank):
        def thunk():
            _env = grounding_env.GroundingEnv(args, args.seed + rank,
                img_encoder=None, fixed=False, manual_set_task=True,
                n_stack=variant['n_stack'])
            _env.game_init()
            _env.tasks = _env.sample_tasks(variant['task_params']['n_tasks'], variants=variant['all_tasks'])
            return _env    
        return thunk

    task_params = variant['task_params']
    # env = NormalizedBoxEnv(AntGoalEnv(n_tasks=task_params['n_tasks'], use_low_gear_ratio=task_params['low_gear']))
    env = make_my_env(args, 0)()
    # import time

    # def make_envs():
    #     t0 = time.time()
    #     envs = SubprocVecEnv([make_my_env(args, i) for i in range(10)])
    #     print('TIMING', time.time() - t0)

    # import pdb; pdb.set_trace()

    ptu.set_gpu_mode(variant['use_gpu'], variant['gpu_id'])

    tasks = env.get_all_task_idx()

    pix_dim = int(np.prod(env.observation_space.shape)) 
    obs_dim = variant['algo_params']['obs_emb_dim']
    action_dim = env.action_space.n # int(np.prod(env.action_space.shape))
    latent_dim = 5
    task_enc_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1

    net_size = variant['net_size']
    # start with linear task encoding
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    nchan = 1 if variant['grayscale'] else 3
    n_layers = variant['n_layers']

    cnn_enc = CNNEncoder(
        64, 64, nchan*variant['n_stack'], obs_dim,
        [8, 4, 3, 3],  #kernels
        [256, 64, 64, 64], #channels
        [2, 2, 2, 2], # strides
        [1, 1, 1, 1], # padding
        # hidden_sizes=[256],
        added_fc_input_size=0,
        batch_norm_conv=False,
        batch_norm_fc=False,
        init_w=1e-4,
        # hidden_init=nn.init.xavier_uniform_,
        # hidden_activation=nn.ReLU(),
        # output_activation=identity,
    )

    task_enc = encoder_model(
            hidden_sizes=[200] * n_layers, # deeper net + higher dim space generalize better
            input_size=obs_dim + action_dim + reward_dim,
            output_size=task_enc_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size] * n_layers,
        input_size=obs_dim + latent_dim,
        output_size=action_dim,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size] * n_layers,
        input_size=obs_dim + latent_dim,
        output_size=action_dim,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size] * n_layers,#, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size] * n_layers,# net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )

    agent = ProtoAgent(
        latent_dim,
        [task_enc, cnn_enc, policy, qf1, qf2, vf],
        **variant['algo_params']
    )

    n_eval_tasks = int(variant['task_params']['n_tasks'] * 0.3)

    algorithm = ProtoSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:-n_eval_tasks]),
        eval_tasks=list(tasks[-n_eval_tasks:]),
        nets=[agent, task_enc, policy, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.to()
    algorithm.train()


@click.command()
@click.argument('gpu', default=0)
@click.argument('exp_id', default='vizdoom')
@click.option('--docker', default=0)
def main(gpu, exp_id, docker):
    max_path_length = 50
    lr = 3E-4

    n_trials = 5
    bsz = (256//2) 

    small = False
    if small: # debugging
        n_tasks=10
        meta_batch=2
        num_tasks_sample=5
        num_train_steps_per_itr=10
        num_evals=3
    else:
        bsz = bsz // 2
        n_tasks=100
        meta_batch=20
        num_tasks_sample=40
        num_train_steps_per_itr=200
        num_evals=10

    # noinspection PyTypeChecker
    variant = dict(
        task_params=dict(
            n_tasks=n_tasks, # 20 works pretty well
            randomize_tasks=True,
            low_gear=False,
        ),
        algo_params=dict(
            meta_batch=meta_batch,
            num_iterations=10000,
            num_tasks_sample=num_tasks_sample,
            num_steps_per_task=n_trials*max_path_length,
            num_train_steps_per_itr=num_train_steps_per_itr, #4000,
            num_evals=num_evals,
            num_steps_per_eval=n_trials*max_path_length,  # num transitions to eval on
            embedding_batch_size=bsz*4,
            embedding_mini_batch_size=bsz,
            batch_size=bsz, # to compute training grads from
            obs_emb_dim=256,
            max_path_length=max_path_length,
            discount=0.99,
            
            soft_target_tau=0.001,
            target_update_period=0.001,

            policy_lr=lr,
            qf_lr=lr,
            vf_lr=lr,
            context_lr=lr,
            reward_scale=5.,
            sparse_rewards=False,
            reparameterize=True,
            kl_lambda=0.5,
            use_information_bottleneck=True,  # only supports False for now
            eval_embedding_source='online_exploration_trajectories',
            train_embedding_source='online_exploration_trajectories',
            # eval_embedding_source='online_exploration_trajectories',
            # train_embedding_source='online_exploration_trajectories',
            recurrent=False, # recurrent or averaging encoder
            dump_eval_paths=False,
        ),
        n_layers=2,
        all_tasks=True,
        n_stack=4,
        grayscale=True,
        net_size=300,
        use_gpu=True,
        gpu_id=gpu,
    )
    exp_name = 'pearl'

    log_dir = '/mounts/output' if docker == 1 else 'output'
    # exp_id = 'ant-goal'
    os.makedirs(os.path.join(log_dir, exp_id), exist_ok=True)
    experiment_log_dir = setup_logger(exp_name, variant=variant, exp_id=exp_id, base_log_dir=log_dir)

    # creates directories for pickle outputs of trajectories (point mass)
    pickle_dir = experiment_log_dir + '/eval_trajectories'
    pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)
    variant['algo_params']['output_dir'] = pickle_dir

    # debugging triggers a lot of printing
    DEBUG = 0
    os.environ['DEBUG'] = str(DEBUG)

    experiment(variant)

if __name__ == "__main__":
    main()
