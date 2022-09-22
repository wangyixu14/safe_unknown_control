import sys
sys.path.append('../Pontryagin-Differentiable-Programming/')

from PDP import PDP
# from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time
from matplotlib import pyplot as plt

import logging
import os
from typing import Sequence

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import time
import matplotlib

import torchsde

seed = 'segway1'

_RX = -2
_RY = 0.0
_RZ = 0.0
_BARRIER_ITERATIONS = 40
_NUM = 100
_STEPS = 100
_DT = 0.1


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
        self.lin1 = nn.Linear(data_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, u_size)
        self.relu=nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = state
        x = self.relu(self.lin1(x))
        # return self.tanh(self.lin2(x))
        return self.lin2(x)

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

        # for param in self.c_net.parameters():
        #     param.requires_grad = False


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
            target = torch.Tensor([[_RX, _RY, 0, 0]]).to('cuda')
           
            opt = torch.optim.Adam([self.pz0_mean], lr=0.1)
            loss = torch.cdist(initx0, target, p=2)
            loss.backward()
            opt.step()


def make_dataset(batch_size, control, device, N):

    def _f(x):
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

        theta = x[1]
        v = x[2]
        theta_dot = x[3]

        f[0] = v
        f[1] = theta_dot
        f[2] = (
            np.cos(theta) * (9.8 * np.sin(theta) + 11.5 * v)
            + 68.4 * v
            - 1.2 * (theta_dot ** 2) * np.sin(theta)
        ) / (np.cos(theta) - 24.7)
        f[3] = (
            -58.8 * v * np.cos(theta)
            - 243.5 * v
            - np.sin(theta) * (208.3 + (theta_dot ** 2) * np.cos(theta))
        ) / (np.cos(theta) ** 2 - 24.7)

        return f
    
    def _g(x):
        """
        Return the control-independent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        g = np.zeros(4)

        # Extract the needed parameters
        # m = params["m"]
        # and state variables
        theta = x[1]

        g[2] = (-1.8 * np.cos(theta) - 10.9) / (
            np.cos(theta) - 24.7
        )
        g[3] = (9.3 * np.cos(theta) + 38.6) / (
            np.cos(theta) ** 2 - 24.7
        )

        return g

    def next_state(x, u, ctrl_step):
        simulate_step = 1e-2
        steps = int(ctrl_step/simulate_step)
        for i in range(steps):
            x = x + simulate_step * (_f(x) + _g(x) * u)
        return x

    demos = np.zeros((batch_size, N, 4))  # for data storage

    for bs in range(batch_size):
        # set initial state
        u = np.random.normal(0, 1)
        v = np.random.normal(0, 1)
        norm = (u**2 + v*v)**(0.5)
        r_I = [u / norm * 0.1  + _RX, v / norm * 0.05  + _RY] 
        ini_v_I = [0.0, 0.0]

        ini_state = r_I + ini_v_I
        state = np.array(ini_state)

        dt = _DT

        # tra = np.zeros((N, len(state)))
        for i in range(N):
            r_noise = 0.02*np.random.normal(0, np.sqrt(dt), 2)
            v_noise = 0.02*np.random.normal(0, np.sqrt(dt), 2)
            diffusion = np.hstack((r_noise, v_noise))
            demos[bs,i] = state
            curr_u = control(torch.from_numpy(state).to(device).float()).detach().cpu().numpy()
            if bs == 0:
                print(state, curr_u)
            state = next_state(state, curr_u[0], dt) + diffusion
    xs = torch.from_numpy(demos).float().to(device)
    ts = torch.linspace(0, N*dt, steps=N, device=device)
    xs = torch.swapaxes(xs, 0, 1)
    return xs, ts


def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=10):
    def coord_convert(p, theta):
        x = p + np.sin(theta)
        y = np.cos(theta)
        return x, y


    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])

    # Left plot: data.
    z1, z2 = np.split(xs.cpu().numpy()[:, :, :2], indices_or_sections=2, axis=-1)
    z1, z2 = coord_convert(z1, z2)

    ax00.add_patch(matplotlib.patches.Circle((0, 1), 0.1, color='pink'))
    [ax00.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    # ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    # ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax00.set_title('Data', fontsize=20)

    # Right plot: samples from learned model.
    xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).detach().cpu().numpy()
    z1, z2 = np.split(xs[:, :, :2], indices_or_sections=2, axis=-1)
    z1, z2 = coord_convert(z1, z2)
    
    ax01.add_patch(matplotlib.patches.Circle((0, 1), 0.1, color='pink'))
    [ax01.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(num_samples)]
    ax01.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    ax01.set_title('Samples', fontsize=20)

    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

def trainBarrier(latent_sde, Barrier,  batch_size=64,  device='cuda', Test=False):

    # Barrier = BarrierNN(state_size=4, hidden_size=64).to(device)

    if Test:
        Barrier.load_state_dict(torch.load('./train/Barrier.pth'))
        print('here')

    optimizer = torch.optim.Adam(Barrier.parameters())
    
    con_opt = torch.optim.Adam(latent_sde.c_net.parameters(), lr=1e-4)
    
    target = torch.from_numpy(np.array([2, 0, 0, 0])).to(device).float()

    ### samples ###
    for it in range(_BARRIER_ITERATIONS):
        _x0 = []
        for _ in range(batch_size):
            u = np.random.normal(0, 1)
            v = np.random.normal(0, 1)
            norm = (u**2 + v*v)**(0.5)
            r_I = [u / norm * 0.1  + _RX, v / norm * 0.05  + _RY] 
            ini_v_I = [0.0, 0.0]
            ini_state = r_I + ini_v_I
            _x0.append(np.array(ini_state))
        _x0 = torch.from_numpy(np.array(_x0)).to(device).float()


        _xu = []
        for _ in range(batch_size):
            u = np.random.normal(0, 1)
            v = abs(np.random.normal(0, 1))
            norm = (u**2 + v*v)**(0.5)
            u = u/norm * 0.1
            v = 1 - v/norm * 0.1
            theta = math.acos(v)
            p = u - math.sin(theta)
            r_I = [p, theta]
            ini_v_I = [np.random.uniform(-1, 1), np.random.uniform(-3, 3)]
            ini_state = r_I + ini_v_I
            _xu.append(np.array(ini_state))
        _xu = torch.from_numpy(np.array(_xu)).to(device).float()

        
        ts = torch.linspace(0, _STEPS*_DT, steps=_STEPS, device=device)
        _xs = latent_sde.sample(batch_size=_x0.size(0), ts=ts, dt=5e-2)
        
        # _xs_reward = reward_weight * _xs
        R = torch.mean(torch.mean(torch.norm(_xs - target, dim=2), dim=(1))) + 10 * torch.mean(torch.norm(_xs[-1] - target, dim=1))
        # R = torch.mean(torch.norm(_xs_reward[-1], dim=1))
   
        optimizer.zero_grad()
        con_opt.zero_grad()

        barrier_xs = Barrier(_xs)
        Lie = torch.sum(torch.mean(barrier_xs[-1]) - torch.mean(barrier_xs, dim=1)) / 10
        Lie_max = torch.max(barrier_xs[1:] - barrier_xs[:-1])
        Lie_max = torch.maximum(Lie_max, torch.max(barrier_xs[-1] - barrier_xs[0]))

        barrier_xu = Barrier(_xu)
        barrier_x0 = Barrier(_x0)
        Unsafe = torch.mean(1 - barrier_xu)
        Unsafe_min = torch.min(barrier_xu)
        Init = torch.mean(barrier_x0)
        Init_max = torch.max(barrier_x0)

        loss = 2*Lie_max + 2*(1 - Unsafe_min) + Init_max + R
        # loss = Lie_max + Lie + (1 - Unsafe_min) + Unsafe + Init + Init_max + R
        # loss = R
        loss.backward()
        if it % 10 == 0:
            print('Iter:{}, loss:{:.2f}, Reward:-{:.2f}, Lie:{:.2f}, Unsafe:{:.2f}, Init:{:.2f}, Lie_max:{:.2f}, Unsafe_min:{:.2f}, Init_max:{:.2f}'
                .format(it,  loss.item(), R.item(), Lie.item(), Unsafe.item(), Init.item(), Lie_max.item(), Unsafe_min.item(), Init_max.item()), flush=True)
        optimizer.step()
        con_opt.step()


def main(
        batch_size=64,
        latent_size=8,
        data_size=4,
        context_size=64,
        hidden_size=32,
        lr_init=1e-2,
        t0=0.,
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=1000,
        pause_every=50,
        noise_std=0.01,
        adjoint=False,
        train_dir='./train/segway/',
        method="euler",
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_sde = LatentSDE(
        data_size=data_size,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)

    checkpoint = torch.load('./train/segway/model_850.pth')
    latent_sde.pz0_mean = checkpoint['pz0_mean']
    latent_sde.pz0_logstd = checkpoint['pz0_logstd']
    latent_sde.h_net = checkpoint['h_net']
    latent_sde.projector = checkpoint['projector']
    latent_sde.g_nets = checkpoint['g_nets']
    latent_sde.f_net = checkpoint['f_net']
    latent_sde.qz0_net = checkpoint['qz0_net']
    latent_sde.encoder = checkpoint['encoder']
    latent_sde.c_net = checkpoint['c_net']
    
    target = torch.from_numpy(np.array([2, 0, 0, 0])).to(device).float()

    _max_r = math.inf
    
    ## fine-tune controller
    Barrier = BarrierNN(state_size=4, hidden_size=64).to(device)
    
    total_num = 0
    for ep in range(100):
    
        trainBarrier(latent_sde, Barrier,  batch_size=64)          

        xs, ts = make_dataset(batch_size=batch_size, control=latent_sde.c_net, device=device, N=_STEPS)
        print(xs[-1, 0, :])
        # optimizer = optim.Adam(params=latent_sde.parameters(), lr=1e-3)
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

        with torch.no_grad():
            # _xs_reward = reward_weight * xs[:30]
            _cur_r = torch.mean(torch.mean(torch.norm(xs - target, dim=2), dim=1)) + 10 * torch.mean(torch.norm(xs[-1] - target, dim=1))
            _cur_r = _cur_r.item()
        print(f"Current rewards: {_cur_r}, save model in this step: {_max_r > _cur_r}")
        
        num = _NUM
        for global_step in tqdm.tqdm(range(1, num + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()

            optimizer.step()
            scheduler.step()
            kl_scheduler.step()
            total_num += 1
            # if global_step % pause_every == 0 and _max_r > _cur_r:
            if total_num % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.info(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                img_path = os.path.join(train_dir, f'step_'+str(total_num)+'_.png')
                vis(xs, ts, latent_sde, bm_vis, img_path)

                model_path = os.path.join(train_dir, 'model_'+str(total_num)+'.pth')
                torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
                    'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
                    'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)


        if _max_r > _cur_r:
            _max_r = _cur_r

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