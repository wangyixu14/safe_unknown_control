import sys
sys.path.append('../Pontryagin-Differentiable-Programming/')

from PDP import PDP
from JinEnv import JinEnv
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

seed = 'uav3_v2'

_RX = -6.0
_RY = -6.0
_RZ = 0.0
_UNSAFE_CENTER = (-4.5, -3.5)
_UNSAFE_RADIUS = 0.25
_UNSAFE_HEIGHT = 4.0
_BARRIER_ITERATIONS = 20
_NUM = 100
_TOTAL_ITERATIONS = 500


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
    def __init__(self, data_size=13, hidden_size=64, u_size=4):
        super(Controller, self).__init__()
        self.lin1 = nn.Linear(data_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, u_size)
        self.relu=nn.ReLU()

    def forward(self, state):
        x = state
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        return self.lin3(x)

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
        self.encoder = Encoder(input_size=data_size + 4, hidden_size=hidden_size, output_size=context_size)
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
            nn.Linear(latent_size + 4, hidden_size),
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
            eps = 0.3*torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
            z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
            zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion':'g'}, dt=dt, bm=bm)
            # Most of the times in ML, we don't sample the observation noise for visualization purposes.
            _xs = self.projector(zs)
        else:
            with torch.no_grad():
                eps = 0.3*torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
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
            target = torch.Tensor([[_RX, _RY, _RZ, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]).to('cuda')
           
            opt = torch.optim.Adam([self.pz0_mean], lr=0.1)
            loss = torch.cdist(initx0, target, p=2)
            loss.backward()
            opt.step()


def make_dataset(batch_size, control, device, N=85):
    # --------------------------- load environment ----------------------------------------
    uav = JinEnv.Quadrotor()
    uav.initDyn(Jx=1, Jy=1, Jz=1, mass=1, l=1, c=0.01)
    uav.initCost(wr=1, wv=1, wq=1, ww=1, wthrust=0.1)

    demos = []  # for data storage

    for bs in range(batch_size):
        # set initial state
        u = np.random.normal(0, 1)
        v = np.random.normal(0, 1)
        w = np.random.normal(0, 1)
        norm = (u**2 + v*v + w*w)**(0.5)
        r_I = [u / norm * 0.1  + _RX, v / norm * 0.1  + _RY, w / norm * 0.1 + _RZ] 
        ini_v_I = [0.0, 0.0, 0.0]
        ini_q = JinEnv.toQuaternion(0,[1,-1,1])
        ini_w = [0.0, 0.0, 0.0]

        ini_state = r_I + ini_v_I + ini_q + ini_w
        state = np.array(ini_state)
        env_dyn = uav.dyn_fn
        env_rfn = uav.r_fn

        tra = []
        for i in range(N):
            r_noise = 0.1*np.random.normal(0, np.sqrt(uav.dt), 3)
            v_noise = 0.05*np.random.normal(0, np.sqrt(uav.dt), 3)
            qwb_noise = np.array([0]*7)
            diffusion = np.hstack((r_noise, v_noise))
            diffusion = np.hstack((diffusion, qwb_noise))
            tra.append(state)
            curr_u = control(torch.from_numpy(state).to(device).float()).detach().cpu().numpy()
            state = env_dyn(state, curr_u).full().flatten() + diffusion
        tra = np.array(tra)
        demos.append(tra)
    demos = np.array(demos)
    xs = torch.from_numpy(demos).float().to(device)
    ts = torch.linspace(0, N*0.02, steps=N, device=device)
    xs = torch.swapaxes(xs, 0, 1)
    return xs, ts


def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=10):
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(2, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    # Left plot: data.
    ax00.add_patch(matplotlib.patches.Circle(_UNSAFE_CENTER, _UNSAFE_RADIUS, color='pink'))
    z1, z2, z3 = np.split(xs.cpu().numpy()[:, :, :3], indices_or_sections=3, axis=-1)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    # ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    # ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax00.set_title('Data', fontsize=20)
    ax10.add_patch(matplotlib.patches.Rectangle((_UNSAFE_CENTER[0]-_UNSAFE_RADIUS, -_UNSAFE_HEIGHT), 2*_UNSAFE_RADIUS, 2*_UNSAFE_HEIGHT, color='pink'))
    [ax10.plot(z1[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
    ax10.scatter(z1[0, :num_samples, 0], z3[0, :num_samples, 0], marker='x')

    # Right plot: samples from learned model.
    xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).detach().cpu().numpy()
    z1, z2, z3 = np.split(xs[:, :, :3], indices_or_sections=3, axis=-1)
    
    ax01.add_patch(matplotlib.patches.Circle(_UNSAFE_CENTER, _UNSAFE_RADIUS, color='pink'))
    [ax01.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(num_samples)]
    ax01.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    ax01.set_title('Samples', fontsize=20)
    ax11.add_patch(matplotlib.patches.Rectangle((_UNSAFE_CENTER[0]-_UNSAFE_RADIUS, -_UNSAFE_HEIGHT), 2*_UNSAFE_RADIUS, 2*_UNSAFE_HEIGHT, color='pink'))
    [ax11.plot(z1[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
    ax11.scatter(z1[0, :num_samples, 0], z3[0, :num_samples, 0], marker='x')

    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

def trainBarrier(latent_sde, batch_size=128, device='cuda', Test=False, Conly=True):

    Barrier = BarrierNN(state_size=13, hidden_size=64).to(device)

    if Test:
        Barrier.load_state_dict(torch.load('./train/Barrier.pth'))
        print('here')

    optimizer = torch.optim.Adam(Barrier.parameters())
    if Conly:
        con_opt = torch.optim.Adam(latent_sde.c_net.parameters(), lr=3e-5)
    else:
        con_opt = torch.optim.Adam(latent_sde.parameters(), lr=3e-5)

    reward_weight = 0.1 * np.ones(13)
    reward_weight[:2] = 10 * np.ones(2)
    reward_weight[2] = 50
    reward_weight[5] = 5
    reward_weight = torch.from_numpy(reward_weight).to(device).float()

    ### samples ###
    for it in range(_BARRIER_ITERATIONS):
        t0 = time.time()
        _x0 = []
        for _ in range(batch_size):
            u = np.random.normal(0, 1)
            v = np.random.normal(0, 1)
            w = np.random.normal(0, 1)
            norm = (u**2 + v*v + w*w)**(0.5)
            r_I = [u / norm * 0.1  + _RX, v / norm * 0.1 + _RY, w / norm * 0.1 + _RZ] 
            ini_v_I = [0.0, 0.0, 0.0]
            ini_q = [1.0, 0.0, 0.0, 0.0]
            ini_w = [0.0, 0.0, 0.0]
            ini_state = r_I + ini_v_I + ini_q + ini_w
            _x0.append(np.array(ini_state))
        _x0 = torch.from_numpy(np.array(_x0)).to(device).float()


        _xu = []
        for _ in range(batch_size):
            ur = np.random.normal(0, 1)
            vr = np.random.normal(0, 1)
            norm = (ur*ur + vr*vr)**(0.5)
            r_I = [ur / norm * _UNSAFE_RADIUS + _UNSAFE_CENTER[0], vr / norm * _UNSAFE_RADIUS  + _UNSAFE_CENTER[1], np.random.uniform(-_UNSAFE_HEIGHT, _UNSAFE_HEIGHT)] 
            u = np.random.normal(0, 1)
            v = np.random.normal(0, 1)
            p = np.random.normal(0, 1)
            q = np.random.normal(0, 1)
            ini_v_I = [np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
            
            norm2 = (u**2 + v*v + q*q + p*p)**(0.5)
            ini_q = [u/norm2, v/norm2, p/norm2, q/norm2]
            ini_w = [np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
            ini_state = r_I + ini_v_I + ini_q + ini_w
            _xu.append(np.array(ini_state))
        _xu = torch.from_numpy(np.array(_xu)).to(device).float()

        
        # t1 = time.time()
        
        # x = 6*torch.rand(size=(batch_size, 1)) - 3
        # x = -1 + 1*torch.rand(size=(batch_size, 1))
        # y = 1.2 + 0.5*torch.rand(size=(batch_size, 1))
        # _xu = torch.cat((x, y), dim=1).to(device)

        # x = 6*torch.rand(size=(batch_size, 1)) - 3
        # y = 6*torch.rand(size=(batch_size, 1)) - 3
        # _xx = torch.cat((x, y), dim=1).to(device)

        ts = torch.linspace(0, 2, steps=100, device=device)
        _xs = latent_sde.sample(batch_size=_x0.size(0), ts=ts)
        # print(torch.norm(_xs, dim=2).shape,torch.norm(_xs, dim=2), torch.mean(torch.norm(_xs, dim=2), dim=(1)).shape)
        # assert False
        _xs_reward = reward_weight * _xs
        R = torch.sum(torch.mean(torch.norm(_xs_reward, dim=2), dim=(1)))

        # t2 = time.time()
   
        optimizer.zero_grad()
        con_opt.zero_grad()

        
        # Lie = torch.mean(Barrier(_xs[-1]) - Barrier(_xs[0]))
        # Lie_max = torch.max(Barrier(_xs[-1]) - Barrier(_xs[0]))
        # for i in range(_xs.size(0) - 1):
        #     Lie += torch.mean(Barrier(_xs[-1]) - Barrier(_xs[i+1])) 
        #     Lie_max = max(torch.max(Barrier(_xs[i+1]) - Barrier(_xs[i])), Lie_max) 
        # Lie /= 10

        barrier_xs = Barrier(_xs)
        Lie = torch.sum(torch.mean(barrier_xs[-1]) - torch.mean(barrier_xs, dim=1)) / 10
        Lie_max = torch.max(barrier_xs[1:] - barrier_xs[:-1])
        Lie_max = torch.maximum(Lie_max, torch.max(barrier_xs[-1] - barrier_xs[0]))

        # Unsafe = torch.mean(1 - Barrier(_xu))
        # Unsafe_min = torch.min(Barrier(_xu))
        # Init = torch.mean(Barrier(_x0))
        # # print(Barrier(_x0), Barrier(_x0).shape)
        # Init_max = torch.max(Barrier(_x0))

        barrier_xu = Barrier(_xu)
        barrier_x0 = Barrier(_x0)
        Unsafe = torch.mean(1 - barrier_xu)
        Unsafe_min = torch.min(barrier_xu)
        Init = torch.mean(barrier_x0)
        Init_max = torch.max(barrier_x0)

        # t3 = time.time()

        # if Test:
        #     print(Unsafe.item(), Unsafe_min, Init.item(), 
        #         Init_max, Lie.item(), Lie_max)
        #     assert False

        # if Lie_max <= 0.005 and Unsafe_min >= 0.99 and Init_max < 0.02:
        #     print(torch.max(Barrier(_x0)))
        #     print('learned controller is: ', latent_sde.c_net.lin.weight.data)
        #     torch.save(Barrier.state_dict(), './train/Barrier'+seed+'.pth')
            
        #     mean_list = []
        #     std_list = []
        #     for j in range(_xs.size(0)):
        #         value = Barrier(_xs[j]).detach().cpu().numpy()
        #         mean_list.append(np.mean(value))
        #         std_list.append(np.std(value))

        #     mean_list = np.array(mean_list)
        #     std_list = np.array(std_list)
        #     plt.clf()
        #     plt.plot(np.arange(len(mean_list)), mean_list)
        #     plt.fill_between(np.arange(len(mean_list)), mean_list - std_list,  mean_list + std_list, alpha=0.3)
        #     plt.savefig('barrier'+seed+'.png')
        #     return True

        loss = 2*Lie_max + 2*(1 - Unsafe_min) + Init_max + R
        # loss = R
        loss.backward()
        if (it+1) % 10 == 0:
            print('Iter:{}, loss:{:.2f}, Reward:-{:.2f}, Lie:{:.2f}, Unsafe:{:.2f}, Init:{:.2f}, Lie_max:{:.2f}, Unsafe_min:{:.2f}, Init_max:{:.2f}'
                .format(it,  loss.item(), R.item(), Lie.item(), Unsafe.item(), Init.item(), Lie_max.item(), Unsafe_min.item(), Init_max.item()), flush=True)
            # print(latent_sde.c_net.lin.weight.data, latent_sde.c_net.lin.weight.data.grad)

            # print(latent_sde.c_net.lin1.weight.data[:5])
        optimizer.step()
        con_opt.step()

        # t4 = time.time()

        # print((t1-t0)/(t4-t0), (t2-t1)/(t4-t0), (t3-t2)/(t4-t0), (t4-t3)/(t4-t0), t4-t0)


    # return False


def main(
        batch_size=32,
        latent_size=16,
        data_size=13,
        context_size=64,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=6,
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=1000,
        pause_every=50,
        noise_std=0.01,
        adjoint=False,
        train_dir='./train/uav/',
        method="euler",
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_sde = LatentSDE(
        data_size=data_size,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)

    # checkpoint = torch.load('./train/uav/model_uav3_v5.pth')
    # latent_sde.pz0_mean = checkpoint['pz0_mean']
    # latent_sde.pz0_logstd = checkpoint['pz0_logstd']
    # latent_sde.h_net = checkpoint['h_net']
    # latent_sde.projector = checkpoint['projector']
    # latent_sde.g_nets = checkpoint['g_nets']
    # latent_sde.f_net = checkpoint['f_net']
    # latent_sde.qz0_net = checkpoint['qz0_net']
    # latent_sde.encoder = checkpoint['encoder']
    # latent_sde.c_net = checkpoint['c_net']
    # latent_sde.c_net.load_state_dict(torch.load('./train/uav/nominal_control.pth'))
    
    latent_sde.setlatentInit()
    # xs, ts = make_dataset(batch_size=batch_size, control=latent_sde.c_net, device=device)
    # assert False
    
    ## fine-tune controller
    # Barrier = BarrierNN(state_size=2, hidden_size=64).to(device)
    prev_flag = False
    for ep in range(_TOTAL_ITERATIONS):
        latent_sde.setlatentInit()
        latent_sde.fixu = False
        # if ep > 0:
        trainBarrier(latent_sde, batch_size=32)
        # else:
        #     trainBarrier(latent_sde, Conly=True)            

        latent_sde.fixu = True
        # print('controller after control optimization: ', latent_sde.c_net.lin1.weight.data[:1])
        xs, ts = make_dataset(batch_size=batch_size, control=latent_sde.c_net, device=device, N=128)
        print(xs[-1, 0, :])
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=3e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

        # Fix the same Brownian motion for visualization.
        bm_vis = torchsde.BrownianInterval(
            t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

        # if ep == 0:
        #     num = 500
        # else:
        num = _NUM
        for global_step in tqdm.tqdm(range(1, num + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()

            optimizer.step()
            scheduler.step()
            kl_scheduler.step()

            if global_step % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.info(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                img_path = os.path.join(train_dir, f'seed_'+seed+'_step_'+str(global_step)+'_.png')
                vis(xs, ts, latent_sde, bm_vis, img_path)

            if global_step % 50 == 0:
                model_path = os.path.join(train_dir, 'model_'+seed+'.pth')
                torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
                    'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
                    'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)

        # print('controller after generative modeling: ', latent_sde.c_net.lin1.weight.data[:5])

        # if prev_flag and flag:
        #     # [-0.5309, -3.3263]
        #     print('learned controller is: ', latent_sde.c_net.lin.weight.data)
        #     assert False

        # prev_flag = flag

    ## training
    ## for training generative modeling of SDE/ learn a starting point


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # policy = Controller().to(device)

    # xs, ts = make_dataset(batch_size=1024, control=policy, device=device)
    # latent_sde = LatentSDE(
    #     data_size=data_size,
    #     latent_size=latent_size,
    #     context_size=context_size,
    #     hidden_size=hidden_size,
    #     policy=policy
    # ).to(device)
    # optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    # kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    # # Fix the same Brownian motion for visualization.
    # bm_vis = torchsde.BrownianInterval(
    #     t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

    # for global_step in tqdm.tqdm(range(1, num_iters + 1)):
    #     if global_step % 10 == 0:
    #         latent_sde.setlatentInit()
    #     latent_sde.zero_grad()
    #     log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
    #     loss = -log_pxs + log_ratio * kl_scheduler.val
    #     loss.backward()

    #     optimizer.step()
    #     scheduler.step()
    #     kl_scheduler.step()

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


"""
export CUDA_VISIBLE_DEVICES=7
nohup python -m examples.uav3_v5 &>train_log/uav3_v5.log &
echo $! > train_log/pid_uav3_v5.txt
"""