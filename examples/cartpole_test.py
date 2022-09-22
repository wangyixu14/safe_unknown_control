import sys
sys.path.append('../Pontryagin-Differentiable-Programming/')

from PDP import PDP
# from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np

import logging
import os
from typing import Sequence

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import time
import matplotlib
import torchsde
import copy
import matplotlib.patches as mpatches

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# # for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
# # It's also possible to use the reduced notation by directly setting font.family:
# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "Helvetica"
# })


seed = 'cartpole'

INI_STATE = [-0.067, -0.55, -0.35, 0.53]

_BARRIER_ITERATIONS = 150
_NUM = 100
_STEPS = 250
_DT = 0.1
real_patch = mpatches.Patch(color='#ff7f0e', label='Ours')
gene_patch = mpatches.Patch(color='#2ca02c', label='Gene')
ppo_patch = mpatches.Patch(color='#1f77b4', label='PPO-L')
focops_path = mpatches.Patch(color='#13eac9', label='FOCOPS')

class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out

class Controller(nn.Module):
    def __init__(self, data_size=4, hidden_size=16, u_size=1):
        super(Controller, self).__init__()
        self.lin1 = nn.Linear(data_size, u_size)
        self.lin1.weight.data = torch.Tensor([[-0.0652, -0.2577, -1.3080, -0.6947]])

    def forward(self, state):
        x = state
        return self.lin1(x)

class BarrierNN(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(BarrierNN, self).__init__()
        self.B_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        nn.init.kaiming_normal_(self.B_net[0].weight, mode='fan_in')
        nn.init.kaiming_normal_(self.B_net[2].weight, mode='fan_in')
        nn.init.kaiming_normal_(self.B_net[4].weight, mode='fan_in')
    def forward(self, state):
        return self.B_net(state)

class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size, u_size=1):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size + u_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

        self.c_net = Controller().to('cuda')


        self.h_net = nn.Sequential(
            nn.Linear(latent_size + u_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1)) 

    def h(self, t, y):
        u = self.c_net(self.projector(y))
        y = torch.cat((y, u), dim=1)
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        us = self.c_net(xs)
        xus = torch.cat((xs, us), dim=2)
        ctx = self.encoder(torch.flip(xus, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + 0.2*qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=1e-2, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)

        _xs = self.projector(zs)
        r = torch.sum(torch.mean(torch.norm(_xs, dim=2), dim=(1)))
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + 35*logqp_path, r

    def sample(self, batch_size, ts, dt=1e-2, bm=None):
        eps = 0.3*torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion':'g'}, dt=dt, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)               
        return _xs

    def setlatentInit(self):
        for _ in range(10):
            initx0 = self.projector(self.pz0_mean)
            target = torch.Tensor([INI_STATE]).to('cuda')
           
            opt = torch.optim.Adam([self.pz0_mean], lr=0.1)
            loss = torch.cdist(initx0, target, p=2)
            loss.backward()
            opt.step()


def metric(xs):
    xs = torch.swapaxes(xs, 0, 1)
    unsafe = 0
    total_return = []
    print(xs.shape)
    for i in range(xs.size(dim=0)):
        reward = 0
        for j in range(xs.size(dim=1)):
            if xs[i][j][0] <= -0.75:
                unsafe += 1
                break
            reward -= torch.norm(xs[i][j][:2])
            reward += 0.3* torch.norm(xs[i][j][:2] - torch.Tensor([-1, 0]).to('cuda'))
            if i == 0:
                print(xs[i][j][:2], reward)
        total_return.append(reward.detach().cpu().numpy())
    print(unsafe, np.mean(total_return), np.std(total_return))
    assert False

def compare(xs, _xs):
    diff = 0.0
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            if _xs[i, j, 0] < xs[i, j, 0]:
                # print(xs[i, j, 0], _xs[i, j, 0])
                diff = np.max([diff, xs[i, j, 0] - _xs[i, j, 0]])
    print(diff)
    assert False

def make_dataset(batch_size, latent_sde, device, Barrier):

    # Barrier = BarrierNN(state_size=4, hidden_size=64).to('cuda')
    Barrier.load_state_dict(torch.load('./train/cartpole/barrier.pth'))

    def _f(x, u):
        """
        Return the control-independent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        f = np.zeros(4)

        pos = x[0]
        theta = x[1]
        v = x[2]
        theta_dot = x[3]

        f[0] = v
        f[1] = theta_dot
        f[2] =  (u + np.sin(theta)*(theta_dot**2 - np.cos(theta))) / (1 + np.sin(theta)**2)
        f[3] = (
                u*np.cos(theta) + 
                theta_dot**2 * np.cos(theta) * np.sin(theta) -
                2*np.sin(theta)) / (1 + np.sin(theta)**2)

        return f

    def next_state(x, u, ctrl_step):
        simulate_step = 1e-2
        steps = int(ctrl_step/simulate_step)
        for i in range(steps):
            x = x + simulate_step * (_f(x, u))
        return x

    demos = np.zeros((batch_size, _STEPS, 4))  # for data storage

    barrier_set = []
    for bs in range(batch_size):
        ini_state = copy.copy(INI_STATE)
        ini_state[0] += np.random.uniform(-0.1, 0.1, 1)[0]
        ini_state[1] += np.random.uniform(-0.05, 0.05, 1)[0]
        state = np.array(ini_state)

        for i in range(_STEPS):
            r_noise = 0.03*np.random.normal(0, np.sqrt(_DT), 1)[0]
            v_noise = 0.03*np.random.normal(0, np.sqrt(_DT), 1)[0]
            diffusion = np.array([r_noise, v_noise, 0, 0])
            demos[bs,i] = state
            curr_u = latent_sde.c_net(torch.from_numpy(state).to(device).float()).detach().cpu().numpy()
            barrier_value = Barrier(torch.from_numpy(state).to(device).float()).detach().cpu().numpy()[0]
            barrier_set.append(barrier_value)
            state = next_state(state, curr_u, _DT) + diffusion

    xs = torch.from_numpy(demos).float().to(device)
    ts = torch.linspace(0, _STEPS*_DT, steps=_STEPS, device=device)
    xs = torch.swapaxes(xs, 0, 1)

    # metric(xs)

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])

    Init = np.load('cartpole_init_max.npy')
    Unsafe = np.load('cartpole_unsafe_min.npy')
    Lie = np.load('cartpole_lie.npy')
    ax00.plot(list(range(len(Lie))),Lie, label='Lie')
    ax00.plot(list(range(len(Unsafe))), Unsafe, label='$\min_{x_{u} \in X_u} B(x_{u})$')
    ax00.plot(list(range(len(Init))), Init, label='$\max_{x_{0} \in X_0} B(x_{0})$')
    ax00.legend(fontsize=25)
    ax00.set_title('Barrier Loss on $\hat{\mathcal{M}}$', fontsize=30)    

    barrier_set = np.array(barrier_set)
    barrier_set = np.reshape(barrier_set, (-1, _STEPS))
    print(barrier_set.shape, len(barrier_set[0]))
    ax01.plot(np.arange(len(barrier_set[0]))*_DT, np.mean(barrier_set, axis=0), label='$B(s)$ on $\mathcal{M}$')
    ax01.fill_between(np.arange(len(barrier_set[0]))*_DT, np.mean(barrier_set, axis=0) - np.std(barrier_set, axis=0), np.mean(barrier_set, axis=0) + np.std(barrier_set, axis=0), alpha=0.3)
    ax01.set_title('Barrier Value on $\mathcal{M}& \hat{\mathcal{M}}$', fontsize=30)
    ax01.set_xlabel('$t/s$', labelpad=0., fontsize=23)

    bm_vis = torchsde.BrownianInterval(
        t0=0, t1=_STEPS*_DT, size=(batch_size, 4,), device=device, levy_area_approximation="space-time")
    _xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).detach().cpu().numpy()
    _gene_barrier = Barrier(torch.from_numpy(_xs).to(device).float()).squeeze().detach().cpu().numpy()
    # compare(xs.detach().cpu().numpy(), _xs)
    print('max Barrier value is: ', np.amax(_gene_barrier))
    assert False

    ax01.plot(np.arange(len(_gene_barrier))*_DT, np.mean(_gene_barrier, axis=1), label='$B(\hat{s})$ on $\hat{\mathcal{M}}$')
    ax01.fill_between(np.arange(len(_gene_barrier))*_DT, np.mean(_gene_barrier, axis=1) - np.std(_gene_barrier, axis=1), np.mean(_gene_barrier, axis=1) + np.std(_gene_barrier, axis=1), alpha=0.3)
    ax01.legend(fontsize=25)

    lab = ax00.get_xticklabels() + ax00.get_yticklabels() + ax01.get_xticklabels() + ax01.get_yticklabels()
    for l in lab:
        l.set_fontsize(18)  

    plt.savefig('cartpole_barrier.pdf', bbox_inches='tight')
    # assert False

    return xs, ts


def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=2):
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 1)
    ax00 = fig.add_subplot(gs[0, 0])

    # Left plot: data. real environment data
    z1, z2 = np.split(xs.cpu().numpy()[:, :, :2], indices_or_sections=2, axis=-1)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0], color='#ff7f0e') for i in range(num_samples)]
    ax00.add_patch(matplotlib.patches.Rectangle((-1.25, -1), 0.5, 2, color='red'))
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x', )
    ax00.set_xlabel('$x$', labelpad=0., fontsize=18)
    ax00.set_ylabel('$\\theta$', labelpad=0, fontsize=18)
    ax00.set_yticks([-1, -0.5, 0, 0.5, 1], minor=False)
    ax00.set_title('Cartpole', fontsize=18)

    # Right plot: samples from learned model.
    xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).detach().cpu().numpy()
    z1, z2 = np.split(xs[:, :, :2], indices_or_sections=2, axis=-1)
    
    [ax00.plot(z1[:, i, 0], z2[:, i, 0], color='#2ca02c') for i in range(num_samples)]
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    # ax00.set_title('Generative', fontsize=30)
    # ax00.set_xlabel('$\hat{x}$', labelpad=0., fontsize=23)
    # ax00.set_ylabel('$\hat{\theta}$', labelpad=.5, fontsize=23)

    ppo_tra = np.load('./train/cartpole/ppo/cartpole_ppo_tra.npy')
    [ax00.plot(ppo_tra[i, :, 0], ppo_tra[i, :, 1], color='#1f77b4') for i in [496, 499]]
    # ax00.scatter(ppo_tra[:num_samples, 0, 0], z2[:num_samples, 0, 1], marker='x')

    focops = np.load('./train/cartpole/ppo/cartpole_focops_tra.npy')
    focops = np.reshape(focops, (2, 100, 4))
    [ax00.plot(focops[i, :, 0], focops[i, :, 1], color='#13eac9') for i in [0, 1]]

    lab = ax00.get_xticklabels() + ax00.get_yticklabels()
    for l in lab:
        l.set_fontsize(15)

    plt.legend(handles=[real_patch, gene_patch, ppo_patch, focops_path], fontsize=10, loc='upper right')

    plt.savefig('cartpole_traj.pdf', bbox_inches='tight')
    plt.close()

def trainBarrier(latent_sde, Barrier,  batch_size=64,  device='cuda', Test=False):
    Barrier.load_state_dict(torch.load('./train/cartpole/barrier0831.pth'))

    optimizer = torch.optim.Adam(Barrier.parameters())
    weight = np.linspace(0, 1, _BARRIER_ITERATIONS)
    ### samples ###
    Lie_list = []
    Unsafe_min_list = []
    Init_max_list = []

    for it in range(_BARRIER_ITERATIONS):
        _x0 = []
        for _ in range(batch_size):
            ini_state = copy.copy(INI_STATE)
            ini_state[0] += np.random.uniform(-0.1, 0.1, 1)[0]
            ini_state[1] += np.random.uniform(-0.05, 0.05, 1)[0]
            _x0.append(np.array(ini_state))
        _x0 = torch.from_numpy(np.array(_x0)).to(device).float()

        _xu = []
        for _ in range(batch_size):
            # random_vec = np.random.normal(0, 1, size=(1, 4))
            # normalize_vec = random_vec / np.linalg.norm(random_vec)
            # length = np.random.uniform(1, 1.5, 1)[0]
            # _xu.append(length * normalize_vec[0])
            # pos = np.random.uniform(-2, -0.75)
            pos = np.random.uniform(-1, -0.58)
            theta = np.random.uniform(-2, 2)
            v = np.random.uniform(-3, 3)
            theta_dot = np.random.uniform(-3, 3)
            _xu.append(np.array([pos, theta, v, theta_dot]))

        _xu = torch.from_numpy(np.array(_xu)).to(device).float()

        
        ts = torch.linspace(0, _STEPS*_DT, steps=_STEPS, device=device)
        _xs = latent_sde.sample(batch_size=_x0.size(0), ts=ts, dt=5e-2)
        
        R = torch.mean(torch.mean(torch.norm(_xs, dim=2), dim=(1)))
   
        optimizer.zero_grad()

        barrier_xs = Barrier(_xs)

        Lie = torch.mean(barrier_xs[-1] - barrier_xs[0])
        Lie_max = torch.max(barrier_xs[-1] - barrier_xs[0])
        for i in range(_xs.size(0) - 1):
            Lie += torch.abs(torch.mean(barrier_xs[i+1] -barrier_xs[i])) 
            Lie_max = max(torch.max(barrier_xs[i+1] -barrier_xs[i]), Lie_max) 

        barrier_xu = Barrier(_xu)
        barrier_x0 = Barrier(_x0)
        Unsafe = torch.mean(1 - barrier_xu)
        Unsafe_min = torch.min(barrier_xu)
        Init = torch.mean(barrier_x0)
        Init_max = torch.max(barrier_x0)

        loss = weight[it]*Lie / 2 + ((1 - Unsafe_min) + Init_max)
        loss.backward()

        Lie_list.append(Lie.item())
        Unsafe_min_list.append(Unsafe_min.item())
        Init_max_list.append(Init_max.item())

        if (it+1) % 1 == 0:
            print('Iter:{}, loss:{:.2f}, Reward:-{:.2f}, Lie:{:.2f}, Unsafe:{:.2f}, Init:{:.2f}, Lie_max:{:.2f}, Unsafe_min:{:.2f}, Init_max:{:.2f}'
                .format(it,  loss.item(), R.item(), Lie.item(), Unsafe.item(), Init.item(), Lie_max.item(), Unsafe_min.item(), Init_max.item()), flush=True)
        optimizer.step()

    np.save('cartpole_lie.npy', np.array(Lie_list))
    np.save('cartpole_unsafe_min.npy', np.array(Unsafe_min_list))
    np.save('cartpole_init_max.npy', np.array(Init_max_list))

    torch.save(Barrier.state_dict(), './train/cartpole/barrier.pth')


def main(
        batch_size=256,
        latent_size=4,
        data_size=4,
        context_size=32,
        hidden_size=32,
        lr_init=1e-2,
        t0=0.,
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=1000,
        pause_every=50,
        noise_std=0.01,
        adjoint=False,
        train_dir='./train/cartpole/',
        method="euler",
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_sde = LatentSDE(
        data_size=data_size,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)

    checkpoint = torch.load('./train/cartpole/model_12600.pth')
    latent_sde.pz0_mean = checkpoint['pz0_mean']
    latent_sde.pz0_logstd = checkpoint['pz0_logstd']
    latent_sde.h_net = checkpoint['h_net']
    latent_sde.projector = checkpoint['projector']
    latent_sde.g_nets = checkpoint['g_nets']
    latent_sde.f_net = checkpoint['f_net']
    latent_sde.qz0_net = checkpoint['qz0_net']
    latent_sde.encoder = checkpoint['encoder']
    latent_sde.c_net = checkpoint['c_net']
    
    ## fine-tune controller
    Barrier = BarrierNN(state_size=4, hidden_size=64).to(device)
    
    total_num = 6200
    for ep in range(200):
    
        # trainBarrier(latent_sde, Barrier,  batch_size=64)          

        xs, ts = make_dataset(batch_size=batch_size, latent_sde=latent_sde, device=device, Barrier=Barrier)
        print(xs[-1, 0, :])
        print(latent_sde.c_net.lin1.weight.data)
        
        optimizer = optim.Adam([
                                    {'params': latent_sde.pz0_mean},
                                    {'params': latent_sde.pz0_logstd},
                                    {'params': latent_sde.h_net.parameters()},
                                    {'params': latent_sde.projector.parameters()},
                                    {'params': latent_sde.g_nets.parameters()},
                                    {'params': latent_sde.f_net.parameters()},
                                    {'params': latent_sde.qz0_net.parameters()},
                                    {'params': latent_sde.encoder.parameters()},
                                ], lr=1e-4)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

        # Fix the same Brownian motion for visualization.
        bm_vis = torchsde.BrownianInterval(
            t0=t0, t1=_STEPS*_DT, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")
        
        num = _NUM
        for global_step in tqdm.tqdm(range(1, num + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()

            optimizer.step()
            scheduler.step()
            kl_scheduler.step()
            # total_num += 1


            if total_num % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.info(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                img_path = os.path.join(train_dir, f'step_'+str(total_num)+'_.png')
                vis(xs, ts, latent_sde, bm_vis, img_path)
                assert False

                model_path = os.path.join(train_dir, 'model_'+str(total_num)+'.pth')
                torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
                    'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
                    'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # policy = Controller().to(device)

    # xs, ts = make_dataset(batch_size=batch_size, control=policy, device=device)
    # latent_sde = LatentSDE(
    #     data_size=data_size,
    #     latent_size=latent_size,
    #     context_size=context_size,
    #     hidden_size=hidden_size,
    # ).to(device)

    # optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    # kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    # # Fix the same Brownian motion for visualization.
    # bm_vis = torchsde.BrownianInterval(
    #     t0=t0, t1=_STEPS*_DT, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

    # for global_step in tqdm.tqdm(range(0, num_iters + 1)):
    #     if global_step % 500 == 0:
    #         latent_sde.setlatentInit()
    #     latent_sde.zero_grad()
    #     log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
    #     loss = -log_pxs + log_ratio * kl_scheduler.val
    #     loss.backward()

    #     optimizer.step()
    #     scheduler.step()
    #     kl_scheduler.step()

    #     if global_step % 100 == 0:
    #         lr_now = optimizer.param_groups[0]['lr']
    #         logging.warning(
    #             f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
    #             f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
    #         )
    #         img_path = os.path.join(train_dir, f'global_step_{global_step:06d}.pdf')
    #         vis(xs, ts, latent_sde, bm_vis, img_path)

    #     if global_step % 100 == 0:
    #         model_path = os.path.join(train_dir, 'nomial_'+str(global_step)+'model.pth')
    #         torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
    #             'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
    #             'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)

        


if __name__ == "__main__":
    fire.Fire(main)


"""
export CUDA_VISIBLE_DEVICES=5
nohup python -m examples.segway1 &>train_log/segway1.log &
echo $! > train_log/pid_segway1.txt
"""