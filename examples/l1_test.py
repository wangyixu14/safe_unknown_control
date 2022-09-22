# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a latent SDE on data from a stochastic Lorenz attractor.
Reproduce the toy example in Section 7.2 of https://arxiv.org/pdf/2001.01328.pdf
To run this file, first run the following to install extra requirements:
pip install fire
To run, execute:
python -m examples.latent_sde_lorenz --train-dir ./train/
"""
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
import matplotlib.patches as mpatches

seed = '1'
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

class StochasticLorenz(object):
    """Stochastic Lorenz attractor.
    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, control, b: Sequence = (0, 0.2), nominal=False):
        super(StochasticLorenz, self).__init__()
        self.b = b
        self.nominal = nominal
        self.control = control

    def f(self, t, y):
        x1, x2 = torch.split(y, split_size_or_sections=(1, 1), dim=1)

        if self.nominal:
            f1 = 0.8*x2
            f2 = -x1 - x2 - 0.3*x1**3

        else:
            u = self.control(y)

            f1 = x2
            f2 = (u - 0.5*x1**3)

        return torch.cat([f1, f2], dim=1)

    def g(self, t, y):
        x1, x2 = torch.split(y, split_size_or_sections=(1, 1), dim=1)
        b1, b2 = self.b
        if self.nominal:
            g1 = x1 * 0
            g2 = x2 * 0
        else:
            g1 = x1 * b1
            g2 = x2 * b2

        return torch.cat([g1, g2], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
        return xs


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
    def __init__(self, data_size=2, u_size=1):
        super(Controller, self).__init__()
        self.lin = nn.Linear(data_size, u_size, bias=False)
        self.lin.weight.data = torch.Tensor([[-1,  -1]])

    def forward(self, state):
        return self.lin(state)

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

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size + 1, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

        self.c_net = Controller()

        # for param in self.c_net.parameters():
        #     param.requires_grad = False


        self.h_net = nn.Sequential(
            nn.Linear(latent_size + 1, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
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
        self.fixu = False

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1)) 

    def h(self, t, y):
        if self.fixu:
            with torch.no_grad():
                u = self.c_net(self.projector(y))
        else:
            u = self.c_net(self.projector(y))
        y = torch.cat((y, u), dim=1)
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        if self.fixu:
            with torch.no_grad():
                us = self.c_net(xs)
        else:
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
        if not self.fixu:
            eps = 0.1*torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
            z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
            zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion':'g'}, dt=dt, bm=bm)
            # Most of the times in ML, we don't sample the observation noise for visualization purposes.
            _xs = self.projector(zs)
        else:
            with torch.no_grad():
                eps = 0.1*torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
                # img GCN -> encode -> z0 -> sdeint -> zs -> xs y traject
                # img GCN -> NN -> u ->
                z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
                zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion':'g'}, dt=dt, bm=bm)
                # Most of the times in ML, we don't sample the observation noise for visualization purposes.
                _xs = self.projector(zs)                
        return _xs

    def setlatentInit(self):
        for _ in range(10):
            initx0 = self.projector(self.pz0_mean)
            target = torch.Tensor([[-2, 0]]).to('cuda')
           
            opt = torch.optim.Adam([self.pz0_mean], lr=0.1)
            loss = torch.cdist(initx0, target, p=2)
            loss.backward()
            opt.step()


def metric(xs):
    xs = torch.swapaxes(xs, 0, 1)
    unsafe = 0
    total_return = []
    for i in range(xs.size(dim=0)):
        reward = 0
        for j in range(xs.size(dim=1)):
            if -1 <= xs[i][j][0] and xs[i][j][0] <= 0 and 1.2 <= xs[i][j][1] and xs[i][j][1] <= 1.7:
                unsafe += 1
                # break
            reward -= torch.norm(xs[i][j][:])
            reward += 0.3* torch.norm(xs[i][j][:] - torch.Tensor([-0.5, 1.5]).to('cuda'))
        total_return.append(reward.detach().cpu().numpy())
    print(unsafe, np.mean(total_return), np.std(total_return))
    assert False

def compare(xs, _xs):
    xs = xs.detach().cpu().numpy()
    _xs = _xs.detach().cpu().numpy()
    diff = np.array([0, 0])
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            if xs[i, j, 0] > -1.23 and _xs[i, j, 0] > -1.23 and xs[i, j, 1] > _xs[i, j, 1]:
                diff = np.maximum(diff, np.abs(xs[i, j, :] - _xs[i, j, :]))
    print(diff)
    assert False

def make_dataset(t0, t1, batch_size, data_size, noise_std, train_dir, device, control, latent_sde, nominal=False):
    # data_path = os.path.join(train_dir, 'data.pth')

    Init = np.load('2D_Init.npy')
    Unsafe = np.load('2D_Unsafe.npy')
    Lie = np.load('2D_Lie.npy')

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])

    ax00.plot(list(range(len(Lie))),Lie, label='Lie')
    ax00.plot(list(range(len(Unsafe))), Unsafe, label='$\min_{x_{u} \in X_u} B(x_{u})$')
    ax00.plot(list(range(len(Init))), Init, label='$\max_{x_{0} \in X_0} B(x_{0})$')
    ax00.legend(fontsize=25)
    ax00.set_title('Barrier Loss on $\hat{\mathcal{M}}$', fontsize=30)      

    angle = 2* np.pi * torch.rand(size=(batch_size, 1))
    length = torch.sqrt(0.01*torch.rand(size=(batch_size, 1)))
    x = length * np.cos(angle) - 2
    y = length * np.sin(angle)

    _y0 = torch.cat((x, y), dim=1).to(device)

    ts = torch.linspace(t0, t1, steps=100, device=device)
    prev = time.time()
    xs = StochasticLorenz(control=control, nominal=nominal).sample(_y0, ts, noise_std, normalize=False)

    # metric(xs)

    barrier = BarrierNN(state_size=2, hidden_size=64).to(device)
    barrier.load_state_dict(torch.load('./train/Barrier_0919.pth'))

    barvalue = barrier(xs).squeeze().detach().cpu().numpy()

    ax01.plot(np.arange(len(barvalue))*0.06, np.mean(barvalue, axis=1), label='$B(s)$ on $\mathcal{M}$')
    ax01.fill_between(np.arange(len(barvalue))*0.06, np.mean(barvalue, axis=1) - 0.3*np.std(barvalue, axis=1), np.mean(barvalue, axis=1) + 0.3*np.std(barvalue, axis=1), alpha=0.3)

    bm_vis = torchsde.BrownianInterval(
        t0=0, t1=6, size=(batch_size, 4,), device=device, levy_area_approximation="space-time")
    ts = torch.linspace(0, 100*0.06, steps=100, device=device)
    _xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis)
    _gene_barrier = barrier(_xs).squeeze().squeeze().detach().cpu().numpy()
    # compare(xs, _xs)
    print(xs.shape, _xs.shape, np.amax(_gene_barrier))
    assert False

    ax01.plot(np.arange(len(_gene_barrier))*0.06, np.mean(_gene_barrier, axis=1), label='$B(\hat{s})$ on $\hat{\mathcal{M}}$')
    ax01.fill_between(np.arange(len(_gene_barrier))*0.06, np.mean(_gene_barrier, axis=1) - np.std(_gene_barrier, axis=1), np.mean(_gene_barrier, axis=1) + np.std(_gene_barrier, axis=1), alpha=0.3)

    ax01.legend(fontsize=25)
    lab = ax00.get_xticklabels() + ax00.get_yticklabels() + ax01.get_xticklabels() + ax01.get_yticklabels()
    for l in lab:
        l.set_fontsize(18)

    plt.savefig('2D_barrier.pdf', bbox_inches='tight')  


    # print(xs.shape)
    # assert False

    return xs, ts


def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=10):
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 1)
    ax00 = fig.add_subplot(gs[0, 0])
    # ax01 = fig.add_subplot(gs[0, 1])

    # Left plot: real data.
    ax00.add_patch(matplotlib.patches.Rectangle((-1, 1.2), 1, 0.5, color='red'))
    z1, z2 = np.split(xs.cpu().numpy(), indices_or_sections=2, axis=-1)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0], color='#ff7f0e') for i in range(num_samples)]
    # ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    ax00.set_xlabel('$s_1$', labelpad=0., fontsize=18)
    ax00.set_ylabel('$s_2$', labelpad=0, fontsize=18)
    ax00.set_yticks([-1, 0, 1, 2, 3], minor=False)
    ax00.set_title('2D Sys', fontsize=20)

    # ax00.set_xticks(fontsize=18)
    # ax00.set_yticks(fontsize=18)


    # Right plot: samples from learned model.
    xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).cpu().numpy()
    z1, z2 = np.split(xs, indices_or_sections=2, axis=-1)
    
    # ax01.add_patch(matplotlib.patches.Rectangle((-1, 1.2), 1, 0.5, color='pink'))
    [ax00.plot(z1[:, i, 0], z2[:, i, 0], color='#2ca02c') for i in range(num_samples)]
    # ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    # ax00.set_title('Generative', fontsize=30)
    # ax00.set_xlabel('$\hat{s}_1}$', labelpad=0., fontsize=23)
    # ax00.set_ylabel('$\hat{s}_2}$', labelpad=.5, fontsize=23)

    ppo_tra = np.load('./train/new/2D_ppo_tra.npy')
    [ax00.plot(ppo_tra[i, :, 0], ppo_tra[i, :, 1], color='#1f77b4') for i in [0, 332]] 
    focops = np.load('./train/new/twodsys_focops_tra.npy')
    focops = np.reshape(focops, (2, 100, 2))
    [ax00.plot(focops[i, :, 0], focops[i, :, 1], color='#13eac9') for i in [0, 1]] 
    # ax00.plot(ppo_tra[i, :, 0], ppo_tra[i, :, 1], color='#1f77b4') 
    # ax00.set_ylim(-0.25, 2)  
    # ax00.set_xlim(-2.25, 0.25)

    lab = ax00.get_xticklabels() + ax00.get_yticklabels()
    for l in lab:
        l.set_fontsize(15)

    plt.legend(handles=[real_patch, gene_patch, ppo_patch, focops_path], fontsize=10, loc='upper right')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    assert False

def trainBarrier(latent_sde, batch_size=256, device='cuda', Test=False):

    Barrier = BarrierNN(state_size=2, hidden_size=64).to(device)
    Barrier.load_state_dict(torch.load('./train/Barrier_0919.pth'))

    optimizer = torch.optim.Adam(Barrier.parameters())
    # con_opt = torch.optim.Adam(latent_sde.c_net.parameters(), lr=0.01)

    ## TO do, sampling-based Barrier is not sound
    Lie_list = []
    Unsafe_list = []
    Init_list = []
    # weight = np.linspace(0,8, 100)
    weight = np.linspace(1, 6, 200)
    
    ### samples ###
    ratio = 0
    for it in range(200):
        angle = 2* np.pi * torch.rand(size=(batch_size, 1))
        length = torch.sqrt(0.01*torch.rand(size=(batch_size, 1)))
        x = length * np.cos(angle) - 2
        y = length * np.sin(angle)
        _x0 = torch.cat((x, y), dim=1).to(device)
        
        # x = 6*torch.rand(size=(batch_size, 1)) - 3
        # x = -1 + 1*torch.rand(size=(batch_size, 1))
        # y = 1.2 + 0.5*torch.rand(size=(batch_size, 1))
        x = -1.2 + 1.4*torch.rand(size=(batch_size, 1))
        y = 1.05 + 0.65*torch.rand(size=(batch_size, 1))
        _xu = torch.cat((x, y), dim=1).to(device)

        x = 6*torch.rand(size=(batch_size, 1)) - 3
        y = 6*torch.rand(size=(batch_size, 1)) - 3
        _xx = torch.cat((x, y), dim=1).to(device)

        ts = torch.linspace(0, 2.4, steps=40, device=device)
        _xs = latent_sde.sample(batch_size=_xx.size(0), ts=ts)
        
        R = 0.1*torch.sum(torch.mean(torch.norm(_xs, dim=2), dim=(1)))
        # weight = [0.5, 1, 2] 
        optimizer.zero_grad()
        # con_opt.zero_grad()

        # Lie = torch.mean(Barrier(_xs[-1]) - Barrier(_xs[0]))
        # Lie_max = torch.max(Barrier(_xs[-1]) - Barrier(_xs[0]))
        # for i in range(_xs.size(0) - 1):
        #     Lie += torch.mean(Barrier(_xs[-1]) - Barrier(_xs[i+1])) 
        #     Lie_max = max(torch.max(Barrier(_xs[i+1]) - Barrier(_xs[i])), Lie_max) 
        # Lie /= 10

        Lie = torch.mean(Barrier(_xs[-1]) - Barrier(_xs[0]))
        Lie_max = torch.max(Barrier(_xs[-1]) - Barrier(_xs[0]))
        for i in range(_xs.size(0) - 1):
            Lie += torch.abs(torch.mean(Barrier(_xs[i+1]) - Barrier(_xs[i]))) 
            Lie_max = max(torch.max(Barrier(_xs[i+1]) - Barrier(_xs[i])), Lie_max) 

        Unsafe = torch.mean(1 - Barrier(_xu))
        Unsafe_min = torch.min(Barrier(_xu))
        Init = torch.mean(Barrier(_x0))
        Init_max = torch.max(Barrier(_x0))

        # loss = Lie / 2.5 + weight[149-it]*(2*(1 - Unsafe_min) + 4*Init_max)
        loss = Lie / 2  + 2*((1 - Unsafe_min) + Init_max)
        loss.backward()
        # loss = torch.Tensor([0])
        optimizer.step()
        # con_opt.step()

        Lie_list.append(Lie.item())
        Unsafe_list.append(Unsafe_min.item())
        Init_list.append(Init_max.item())

        if (1 - Lie.item()) / Unsafe_min > ratio and Unsafe_min.item() > 0.97:
            ratio = (1 - Lie.item()) / Unsafe_min
            ratio = ratio.item()
            print('safe ratio is: ', ratio, 'saving barrier function')
            torch.save(Barrier.state_dict(), './train/Barrier_0919.pth') 

        if it % 1 == 0:
            print('Iter:{}, loss:{:.2f}, Reward:-{:.2f}, Lie:{:.2f}, Unsafe:{:.2f}, Init:{:.2f}, Lie_max:{:.2f}, Unsafe_min:{:.2f}, Init_max:{:.2f}'
                .format(it,  loss.item(), R.item(), Lie.item(), Unsafe.item(), Init.item(), Lie_max.item(), Unsafe_min.item(), Init_max.item()))
            
    # np.save('2D_Lie.npy', np.array(Lie_list))
    # np.save('2D_Unsafe.npy', np.array(Unsafe_list))
    # np.save('2D_Init.npy', np.array(Init_list))
    # torch.save(Barrier.state_dict(), './train/Barrier_0919.pth')
    assert False

    return False


def main(
        batch_size=500,
        latent_size=4,
        data_size=2,
        context_size=64,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=6,
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=50,
        pause_every=50,
        noise_std=0.01,
        adjoint=False,
        train_dir='./train/',
        method="euler",
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_sde = LatentSDE(
        data_size=data_size,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)

    # checkpoint = torch.load('./train/new/model.pth')
    checkpoint = torch.load('./train/model_1.pth')
    latent_sde.pz0_mean = checkpoint['pz0_mean']
    latent_sde.pz0_logstd = checkpoint['pz0_logstd']
    latent_sde.h_net = checkpoint['h_net']
    latent_sde.projector = checkpoint['projector']
    latent_sde.g_nets = checkpoint['g_nets']
    latent_sde.f_net = checkpoint['f_net']
    latent_sde.qz0_net = checkpoint['qz0_net']
    latent_sde.encoder = checkpoint['encoder']
    latent_sde.c_net = checkpoint['c_net']
    latent_sde.setlatentInit()
    
    ## fine-tune controller
    # Barrier = BarrierNN(state_size=2, hidden_size=64).to(device)
    prev_flag = False
    for _ in range(20):
        latent_sde.setlatentInit()
        latent_sde.fixu = False
        # flag = trainBarrier(latent_sde)            

        latent_sde.fixu = True
        print('controller after control optimization: ', latent_sde.c_net.lin.weight.data)
        xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size, data_size=data_size, noise_std=noise_std, train_dir=train_dir, device=device, control=latent_sde.c_net, latent_sde=latent_sde, nominal=False)
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

        # Fix the same Brownian motion for visualization.
        bm_vis = torchsde.BrownianInterval(
            t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

        for global_step in tqdm.tqdm(range(0, 200 + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
            # print(log_pxs, log_ratio)
            # assert False
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()

            optimizer.step()
            scheduler.step()
            kl_scheduler.step()

            if global_step % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.warning(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                img_path = os.path.join(train_dir, f'2D_tra.pdf')
                vis(xs, ts, latent_sde, bm_vis, img_path)
                assert False
            # if global_step % 100 == 0:
            #     model_path = os.path.join(train_dir, 'model_'+seed+'.pth')
            #     torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
            #         'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
            #         'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)

        print('controller after generative modeling: ', latent_sde.c_net.lin.weight.data)

        if prev_flag and flag:
            # [-0.5309, -3.3263]
            print('learned controller is: ', latent_sde.c_net.lin.weight.data)
            assert False

        prev_flag = flag
    ## training
    ## for training generative modeling of SDE/ learn a starting point
    # optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    # kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    # # Fix the same Brownian motion for visualization.
    # bm_vis = torchsde.BrownianInterval(
    #     t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

    # for global_step in tqdm.tqdm(range(1, num_iters + 1)):
    #     latent_sde.zero_grad()
    #     log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
    #     loss = -log_pxs + log_ratio * kl_scheduler.val
    #     loss.backward()

    #     # r.backward()
    #     # for names, param in latent_sde.c_net.named_parameters():
    #     #     print(names, param.shape, param.grad)
    #     #     assert False

    #     optimizer.step()
    #     scheduler.step()
    #     kl_scheduler.step()

    #     # print(latent_sde.c_net.lin.weight.data)
    #     # assert False

    #     if global_step % pause_every == 0:
    #         lr_now = optimizer.param_groups[0]['lr']
    #         logging.warning(
    #             f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
    #             f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
    #         )
    #         img_path = os.path.join(train_dir, f'global_step_{global_step:06d}.pdf')
    #         vis(xs, ts, latent_sde, bm_vis, img_path)

    #     if global_step % 100 == 0:
    #         model_path = os.path.join(train_dir, 'model.pth')
    #         torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
    #             'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
    #             'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)


if __name__ == "__main__":
    fire.Fire(main)
