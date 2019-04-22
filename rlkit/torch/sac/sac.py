from collections import OrderedDict
import numpy as np
import pickle

import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify, torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import MetaTorchRLAlgorithm
from rlkit.torch.sac.proto import ProtoAgent

import time
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return torch.from_numpy(elem_or_tuple).float()

def _filter_batch(np_batch):

    if isinstance(np_batch, list):
        for v in np_batch:
            if v.dtype == np.bool:
                yield v.astype(int)
            else:
                yield v
    else:
        for k, v in np_batch.items():
            if v.dtype == np.bool:
                yield k, v.astype(int)
            else:
                yield k, v

def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }

from torch.utils.data import Dataset, DataLoader

class ReplayDataset(Dataset):
    def __init__(self, buffer, batch_size, indices, numpify=True, is_online=False):
        self.buffer = buffer
        self.batch_size = batch_size
        self.numpify = numpify
        self.is_online = is_online
        self.indices = indices

    def __len__(self):
        return 10000000

    def batch(self, idx):
        idx = self.indices[idx % len(self.indices)]
        return _get_batch(idx, self.buffer, self.batch_size, self.numpify, self.is_online)

    def __getitem__(self, idx):
        return self.sample_data(self.indices)

    def sample_data(self, indices):
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            batch = self.batch(idx)

            # o = batch['observations'][None, ...]
            # a = batch['actions'][None, ...]
            # r = batch['rewards'][None, ...]
            # no = batch['next_observations'][None, ...]
            # t = batch['terminals'][None, ...]
            (o, a, r, no, t) = (torch.from_numpy(x).float() for x in batch)

            obs.append(o[0])
            actions.append(a[0])
            rewards.append(r[0])
            next_obs.append(no[0])
            terms.append(t[0])

        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
            # batch = np_to_pytorch_batch(batch)
        # [o.cuda() for o in [obs, actions, rewards, next_obs, terms]]
        return [obs, actions, rewards, next_obs, terms]

def _get_batch(idx, buffer, batch_size, numpify=True, is_online=False):
    ''' get a batch from replay buffer for input into net '''
    if is_online:
        batch = buffer.random_batch(idx, batch_size, trajs=is_online)
    else:
        batch = buffer.random_batch(idx, batch_size)

    # if numpify:
    #     batch = np_to_pytorch_batch(batch)

    # print("HELLO")
    o = batch['observations'][None, ...]
    a = batch['actions'][None, ...]
    r = batch['rewards'][None, ...]
    no = batch['next_observations'][None, ...]
    t = batch['terminals'][None, ...]

    return (o, a, r, no, t)

# def _get_encoding_batch(idx, buffer, batch_size, numpify=True, is_online=False):
#     ''' get a batch from the separate encoding replay buffer '''
#     # n.b. if eval is online, training should sample trajectories rather than unordered batches to better match statistics
#     # is_online = (self.eval_embedding_source == 'online')
#     # if idx is None:
#     #     idx = self.task_idx
#     batch = buffer.random_batch(idx, batch_size, trajs=is_online)
#     print("HELLO")
#     # if eval_task:
#     #     batch = self.eval_enc_replay_buffer.random_batch(idx, self.embedding_batch_size, trajs=is_online)
#     # else:
#     #     batch = self.enc_replay_buffer.random_batch(idx, self.embedding_batch_size, trajs=is_online)
#     return np_to_pytorch_batch(batch) if numpify else batch


class ProtoSoftActorCritic(MetaTorchRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            class_lr=1e-1,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            reparameterize=True,
            use_information_bottleneck=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            target_update_period=1000,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            policy=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )
        deterministic_embedding=False
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.latent_dim = latent_dim
        self.qf_criterion = nn.SmoothL1Loss()
        self.vf_criterion = nn.SmoothL1Loss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.eval_statistics = None
        self.kl_lambda = kl_lambda

        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        
        self.target_update_period = target_update_period

        # TODO consolidate optimizers!
        self.policy_optimizer = optimizer_class(
            self.policy.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.policy.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.policy.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.policy.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.policy.task_enc.parameters(),
            lr=context_lr,
        )

        self.cnn_optimizer = optimizer_class(
            self.policy.cnn_enc.parameters(),
            lr=context_lr,
        )

    # TODO Parallelize 
    def sample_data(self, indices, encoder=False):
        # sample from replay buffer for each task

        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = self.enc_replay_loader.next()
            else:
                batch = self.replay_loader.next()

            # o = batch['observations'][None, ...]
            # a = batch['actions'][None, ...]
            # r = batch['rewards'][None, ...]
            # no = batch['next_observations'][None, ...]
            # t = batch['terminals'][None, ...]
            (o, a, r, no, t) = [x.float().to(ptu.device) for x in batch]

            obs.append(o[0])
            actions.append(a[0])
            rewards.append(r[0])
            next_obs.append(no[0])
            terms.append(t[0])

        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
            # batch = np_to_pytorch_batch(batch)
        # [o.cuda() for o in [obs, actions, rewards, next_obs, terms]]
        return [obs, actions, rewards, next_obs, terms]

    def prepare_encoder_data(self, obs, act, rewards):
        ''' prepare task data for encoding '''
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        if self.sparse_rewards:
            rewards = ptu.sparsify_rewards(rewards)
        obs_dim = np.prod(self.env.observation_space.shape)

        task_data = torch.cat([obs[:, :, obs_dim:], act, rewards], dim=2)
        return task_data

    def _do_training(self, indices, step):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        if step == 0:
            import time
            t0 = time.time()
            self.enc_replay_dataset = ReplayDataset(self.enc_replay_buffer, self.embedding_batch_size, indices, numpify=False, is_online=(self.eval_embedding_source == 'online'))
            self.replay_dataset = ReplayDataset(self.replay_buffer, self.batch_size, indices, numpify=False, is_online=False)

            self.enc_replay_loader = iter(torch.utils.data.DataLoader(self.enc_replay_dataset, batch_size=1, 
                shuffle=False, pin_memory=True, sampler=None, batch_sampler=None, num_workers=4,
                worker_init_fn=None))
            self.replay_loader = iter(torch.utils.data.DataLoader(self.replay_dataset, batch_size=1, 
                shuffle=False, pin_memory=True, sampler=None, batch_sampler=None, num_workers=8,
                worker_init_fn=None))
            print('const dataset took:', time.time() - t0)

        # sample contexts for all RL updates
        batch = [x.cuda() for x in self.enc_replay_loader.next()]

        # zero out context and hidden encoder state
        self.policy.clear_z(num_tasks=len(indices))

        import time
        for i in range(num_updates):
            t0 = time.time()

            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, _, _ = mini_batch
            self._take_step(indices, obs_enc, act_enc, rewards_enc)

            # stop backprop
            # print(time.time() - t0, 'TIME')
            self.policy.detach_z()

    def _take_step(self, indices, obs_enc, act_enc, rewards_enc):

        num_tasks = len(indices)

        import time
        t6 = time.time()

        # data is (task, batch, feat)
        batch = self.replay_loader.next()
        # print('sample', time.time() - t6)
        t7 = time.time()
        obs, actions, rewards, next_obs, terms = [x.cuda() for x in batch]
        # print('to_cuda', time.time() - t7)

        t5 = time.time()
        enc_data = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)
        # print('prep enc data', time.time() - t5)

        self.cnn_optimizer.zero_grad()
        self.qf1_optimizer.zero_grad()
        self.context_optimizer.zero_grad()

        t5 = time.time()

        # run inference in networks
        q1_pred, q1_next_pred, q2_next_pred, policy_outputs, task_z = self.policy(obs, actions, next_obs, enc_data, obs_enc, act_enc)
        #print('policy', time.time() - t5)

        # new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        new_actions = policy_outputs

        # KL constraint on z if probabilistic

        t4 = time.time()
        kl_loss = 0
        if self.use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
        #print('kl', time.time() - t4)

            # kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)

        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        actions = actions.view(self.batch_size * num_tasks, -1)

        t3 = time.time()

        best_action_idxs = q1_next_pred.max(
            1, keepdim=True
        )[1]
        target_q_values = q2_next_pred.gather(
            1, best_action_idxs
        ).detach()

        #print('get actions', time.time() - t3)

        y_target = rewards_flat + (1. - terms_flat) * self.discount * target_q_values
        y_target = y_target.detach()
        t2 = time.time()

        # actions is a one-hot vector
        y_pred = torch.sum(q1_pred * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        #print('compute loss', time.time() - t2)
        t1 = time.time()

        """
        Update networks
        """
        loss = qf_loss + kl_loss
        loss.backward()
        #print('backward', time.time() - t1)
        t0 = time.time()

        self.qf1_optimizer.step()
        self.cnn_optimizer.step()  
        self.context_optimizer.step()
        #print('step', time.time() - t0)

        """
        Soft target network updates
        """
        if self.target_update_period > 1 and self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.policy.qf1, self.policy.qf2, 1
            )
        else:
            ptu.soft_update_from_to(
                self.policy.qf1, self.policy.qf2, self.soft_target_tau,
            )
            
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            # TODO this is kind of annoying and higher variance, why not just average
            # across all the train steps?
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.policy.z_dists[0].mean)))
                z_sig = np.mean(ptu.get_numpy(self.policy.z_dists[0].variance))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            # self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            # self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
            #     policy_loss
            # ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'V Predictions',
            #     ptu.get_numpy(v_pred),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Log Pis',
            #     ptu.get_numpy(log_pi),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Policy mu',
            #     ptu.get_numpy(policy_mean),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Policy log std',
            #     ptu.get_numpy(policy_log_std),
            # ))

    def sample_z_from_prior(self):
        self.policy.clear_z()

    def sample_z_from_posterior(self, idx, eval_task=False):
        batch = self.get_encoding_batch(idx=idx, eval_task=eval_task)
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        in_ = self.prepare_encoder_data(obs, act, rewards)
        self.policy.set_z(in_)

    @property
    def networks(self):
        return self.policy.networks + [self.policy]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf1=self.policy.qf1,
            qf2=self.policy.qf2,
            policy=self.policy.policy,
            vf=self.policy.vf,
            target_vf=self.policy.target_vf,
            task_enc=self.policy.task_enc,
        )
        return snapshot
