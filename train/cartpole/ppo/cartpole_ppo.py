"""
Proximal Policy Optimization (PPO) version 2
----------------------------
1 actor and 1 critic
Old policy is given by previous actor policy before updating.
Batch size can be larger than episode length, only update when batch size is reached,
therefore the trick of increasing batch size for stabilizing training can be applied.
To run
------
python ***.py --train/test
"""
import argparse
import threading
import time
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy.linalg as LA
import copy

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

# ENV_NAME = 'BipedalWalker-v2'  # environment name
RANDOMSEED = 2  # random seed

EP_MAX = 7000  # total number of episodes for training
EP_LEN = 100  # total number of steps for each episode
GAMMA = 0.9  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH_SIZE = 32  # update batchsize
A_UPDATE_STEPS = 10  # actor update steps
C_UPDATE_STEPS = 10  # critic update steps
ACTION_RANGE = 20.  # if unnormalized, normalized action range should be 1.
EPS = 1e-8  # numerical residual
TEST_EP = 500
# ppo-penalty
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip
EPSILON = 0.2

RENDER = False
PLOT_RESULT = False
ARG_NAME = 'PPO'
METHOD  = ['penalty', 'clip'][0]

INI_STATE = [-0.067, -0.55, -0.35, 0.53]
_DT = 0.1

class Cartpole(object):

    def __init__(self):
        super(Cartpole, self).__init__()
        self.max_iteration = EP_LEN
        self.t = 0

    def reset(self):
        ini_state = copy.copy(INI_STATE)
        ini_state[0] += np.random.uniform(-0.1, 0.1, 1)[0]
        ini_state[1] += np.random.uniform(-0.05, 0.05, 1)[0]
        self.state = np.array(ini_state)        
        return self.state        

    def step(self, u):
        def _f(x, u):
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

        r_noise = 0.03*np.random.normal(0, np.sqrt(_DT), 1)[0]
        v_noise = 0.03*np.random.normal(0, np.sqrt(_DT), 1)[0]
        diffusion = np.array([r_noise, v_noise, 0, 0])
        self.state = next_state(self.state, u, _DT) + diffusion
        self.t += 1
        return self.state, -LA.norm(self.state), self.t == self.max_iteration


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        # self.linear4.weight.data.uniform_(-init_w, init_w)
        # self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=20., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(num_inputs, num_actions)
        
        # self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std = AddBias(torch.zeros(num_actions))  

        self.num_actions = num_actions
        self.action_range = action_range
        
    def forward(self, state):
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))

        mean    =  self.mean_linear(state)
        # print(state.shape, mean.shape)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        # mean[0] = self.action_range * mean[0]
        zeros = torch.zeros(mean.size())
        if state.is_cuda:
            zeros = zeros.cuda()
        log_std = self.log_std(zeros)
        
        std = log_std.exp()
        return mean, std
        
###############################  PPO  ####################################

class PPO(object):
    """
    PPO class
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, method='clip'):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim, ACTION_RANGE).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)
        print(self.actor, self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=A_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=C_LR)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        self.lag_lamb = 1
        self.penalty = 0

    def a_train(self, state, action, adv, old_pi):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        mu, sigma = self.actor(state)
        pi = torch.distributions.Normal(mu, sigma)
        ratio = torch.exp(pi.log_prob(action) - old_pi.log_prob(action))
        surr = ratio * adv
        if self.method == 'penalty':
            kl = torch.distributions.kl_divergence(old_pi, pi)
            kl_mean = kl.mean()
            aloss = -(surr - self.lam * kl).mean()
        else:  # clipping method, find this is better
            aloss = -torch.mean(
                torch.min(
                    surr,
                    torch.clamp(
                        ratio,
                        1. - self.epsilon,
                        1. + self.epsilon
                    ) * adv
                )
            )
        self.actor_opt.zero_grad()
        aloss.backward()
        self.actor_opt.step()

        if self.method == 'kl_pen':
            return kl_mean

    def c_train(self, cumulative_r, state):
        """
        Update actor network
        :param cumulative_r: cumulative reward batch
        :param state: state batch
        :return: None
        """
        advantage = cumulative_r - self.critic(state)
        closs = (advantage ** 2).mean()
        self.critic_opt.zero_grad()
        closs.backward()
        self.critic_opt.step()

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        s = torch.Tensor(self.state_buffer).to(device)
        a = torch.Tensor(self.action_buffer).to(device)
        r = torch.Tensor(self.cumulative_reward_buffer).to(device)
        with torch.no_grad():
            mean, std = self.actor(s)
            pi = torch.distributions.Normal(mean, std)

            _xu = torch.from_numpy(np.array([-1.375, 0, 0, 0 ])).float().to('cuda')
            # print(_xu.device, s.device)

            self.penalty = -torch.mean(torch.norm(s - _xu, dim=1))  + torch.Tensor([3]).to('cuda') 

            adv = r - self.critic(s) - self.penalty*self.lag_lamb
            self.lag_lamb = max(torch.Tensor([0]).to('cuda'), self.lag_lamb + 1e-4*self.penalty)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        if self.method == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv, pi)
                if kl > 4 * self.kl_target:  # this in in google's paper
                    break
            if kl < self.kl_target / 1.5:  # adaptive lambda, this is in OpenAI's paper
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
            self.lam = np.clip(
                self.lam, 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv, pi)

        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def choose_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        s = s[np.newaxis, :].astype(np.float32)
        s = torch.Tensor(s).to(device)
        mean, std = self.actor(s)
        if greedy:
            a = mean.cpu().detach().numpy()[0]
        else:
            pi = torch.distributions.Normal(mean, std)
            a = pi.sample().cpu().numpy()[0]
        return np.clip(a, -self.actor.action_range, self.actor.action_range)

    def save_model(self, ep):
        torch.save(self.actor.state_dict(), './'+ str(ep) + '.pth')
        # torch.save(self.critic.state_dict(), './'+ str(ep) + '.pth')

    def load_model(self, ep):
        self.actor.load_state_dict(torch.load('./'+ str(ep) + '.pth'))
        # self.critic.load_state_dict(torch.load('./'+ str(ep) + '.pth'))

        self.actor.eval()
        self.critic.eval()

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic(torch.Tensor([next_state]).to(device)).cpu().detach().numpy()[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_   # no future reward if next state is terminal
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()


class Drawer:
    def __init__(self, comments=''):
        global update_plot, stop_plot
        update_plot = threading.Event()
        update_plot.set()
        stop_plot = threading.Event()
        stop_plot.clear()
        self.title = ARG_NAME
        if comments:
            self.title += '_' + comments

    def plot(self):
        plt.ion()
        global all_ep_r, update_plot, stop_plot
        all_ep_r = []
        while not stop_plot.is_set():
            if update_plot.is_set():
                plt.cla()
                plt.title(self.title)
                plt.plot(np.arange(len(all_ep_r)), all_ep_r)
                # plt.ylim(-2000, 0)
                plt.xlabel('Episode')
                plt.ylabel('Moving averaged episode reward')
                update_plot.clear()
            plt.draw()
            plt.pause(0.2)
        plt.ioff()
        plt.close()

    def save(self, path='fig'):
        plt.title(ARG_NAME)
        plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        # plt.ylim(-2000, 0)
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        time_array = time.localtime(time.time())
        time_str = time.strftime("%Y%m%d_%H%M%S", time_array)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, self.title + '_' + time_str)
        plt.savefig(path)
        plt.close()

def train():
    env = Cartpole()
    
    state_dim = 4
    action_dim = 1

    # reproducible
    np.random.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)

    ppo = PPO(state_dim, action_dim, method = METHOD)
    ppo.load_model(6900)
    global all_ep_r, update_plot, stop_plot
    all_ep_r = []
    for ep in range(EP_MAX):

        s = env.reset()

        ep_r = 0
        t0 = time.time()
        for t in range(EP_LEN):
            a = ppo.choose_action(s)

            s_, r, done = env.step(a[0])
            # print(s, a, s_)
            r += 2
            
            ppo.store_transition(s, a, r)  # useful for pendulum since the nets are very small, normalization make it easier to learn
            s = s_
            ep_r += r

            # update ppo
            done = t == EP_LEN - 1
            if len(ppo.state_buffer) == BATCH_SIZE:
                ppo.finish_path(s_, done)
                ppo.update()
                # print('here', t, done)

        ppo.finish_path(s_, done)
        print(
            'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f} | lambda: {:.2f} | penalty: {:.2f}'.format(
                ep + 1, EP_MAX, ep_r,
                time.time() - t0, ppo.lag_lamb.cpu().numpy()[0], ppo.penalty.cpu().numpy()[0],
            )
        )
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        if PLOT_RESULT:
            update_plot.set()
        if (ep) % 300 == 0 and ep >= 100:
            ppo.save_model(ep=ep)
    if PLOT_RESULT:
        stop_plot.set()



if __name__ == '__main__':

    # env = Cartpole()
    # state, done = env.reset(), False
    # tra = []
    # while not done:
    #     tra.append(state[:2])
    #     u = np.array([-0.0652, -0.2577, -1.3080, -0.6947]).dot(state)
    #     state, r, done = env.step(u)
    #     print(state, r, done)
    # tra = np.array(tra)
    # plt.plot(tra[:, 0], tra[:, 1])
    # plt.savefig('cartpole_ppo_test.png')
    # assert False

    # train()
    # assert False
    

    # test
    def test():
        env = Cartpole()
        ppo = PPO(4, 1, method = METHOD)
        ppo.load_model(300)
        mean_epoch_reward = []
        state_collection = []
        action_collection = []
        unsafe = 0
        for te in range(TEST_EP):
            ep_reward = 0
            state = env.reset()
            ep_state = []
            for i in range(EP_LEN):
                action = ppo.choose_action(state, True)
                next_state, r, done = env.step(action[0])
                if te == 0:
                    print(state, action)
                if -2 <= state[0] and state[0] <= -0.75 and -2 <= state[1] and state[1] <= 2:
                    unsafe += 1
                    print(te)
                    # break

                ep_state.append(state)
                action_collection.append(action)
                reward = -LA.norm(next_state[:2])
                reward += 0.3*LA.norm(next_state[:2] - np.array([-1, 0]))
                ep_reward += reward
                state = next_state
            state_collection.append(np.array(ep_state))
            mean_epoch_reward.append(ep_reward)

        print(np.mean(mean_epoch_reward))
        print(np.std(mean_epoch_reward))
        print(unsafe)
        state_collection = np.array(state_collection)
        # print(state_collection.shape)
        np.save('cartpole_ppo_tra.npy', state_collection)

    def plot_test():
        state = np.load('cartpole_ppo_tra.npy')
        [plt.plot(state[i, :, 0], state[i, :, 1]) for i in range(TEST_EP)]
        plt.savefig('cartpole_ppo_test.png')

    test()
    # plot_test()



    