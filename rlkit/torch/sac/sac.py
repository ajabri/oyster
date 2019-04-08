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
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.eval_statistics = None
        self.kl_lambda = kl_lambda

        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

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

    def sample_data(self, indices, encoder=False):
        # sample from replay buffer for each task
        # TODO(KR) this is ugly af
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = self.get_encoding_batch(idx=idx)
            else:
                batch = self.get_batch(idx=idx)
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            r = batch['rewards'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
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

    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        batch = self.sample_data(indices, encoder=True)

        # zero out context and hidden encoder state
        self.policy.clear_z(num_tasks=len(indices))

        import time
        for i in range(num_updates):
            t0 = time.time()
            # TODO(KR) argh so ugly
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, _, _ = mini_batch
            self._take_step(indices, obs_enc, act_enc, rewards_enc)

            # print(time.time() - t0, 'TIME')
            # stop backprop
            self.policy.detach_z()

    def _take_step(self, indices, obs_enc, act_enc, rewards_enc):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        enc_data = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)

        self.cnn_optimizer.zero_grad()

        # run inference in networks
        q1_pred, q1_next_pred, q2_next_pred, policy_outputs, task_z = self.policy(obs, actions, next_obs, enc_data, obs_enc, act_enc)
        # new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        new_actions = policy_outputs

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)

        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        actions = actions.view(self.batch_size * num_tasks, -1)

        best_action_idxs = q1_next_pred.max(
            1, keepdim=True
        )[1]
        target_q_values = q2_next_pred.gather(
            1, best_action_idxs
        ).detach()

        y_target = rewards_flat + (1. - terms_flat) * self.discount * target_q_values
        y_target = y_target.detach()

        # actions is a one-hot vector
        y_pred = torch.sum(q1_pred * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        self.cnn_optimizer.zero_grad()      
        qf_loss.backward()
      
        self.qf1_optimizer.step()
        self.cnn_optimizer.step()  
        self.context_optimizer.step()


        """
        Soft target network updates
        """
        # if self._n_train_steps_total % self.target_update_period == 0:
        ptu.soft_update_from_to(
            self.policy.qf1, self.policy.qf2, self.soft_target_tau
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
