# -*- coding:utf-8  -*-
# Time  : 2023/5/20
# Authors: Osher Elhadad, Matan Shamir, Gili Gutfeld

"""
# =================================== Important =========================================
this agent is a pretrained PPO agent , which can fit any env in Jidi platform.
"""

import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
from os import path
father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4,2),
            nn.Flatten()
        )

    def forward(self, view_state):
        # [batch, 128]
        x = self.net(view_state)
        return x

device = 'cpu'

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(Actor, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder().to(device)
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64, cnn=False):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder().to(device)  # 用GPU计算
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value

class CNN_Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size = 64):
        super(CNN_Actor, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2)
        # self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1)
        # self.flatten = nn.Flatten()
        self.net = Net = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, action_space)

    def forward(self, x):
        x = self.net(x)
        x = torch.relu(self.linear1(x))
        action_prob = F.softmax(self.linear2(x), dim = -1)
        return action_prob

class CNN_Critic(nn.Module):
    def __init__(self, state_space, hidden_size = 64):
        super(CNN_Critic, self).__init__()

        self.net = Net = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x



class CNN_CategoricalActor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size = 64):
        super(CNN_CategoricalActor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels=32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(128, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(self.linear1(x))
        action_prob = F.softmax(self.linear2(x), dim = -1)
        c = Categorical(action_prob)
        sampled_action = c.sample()
        greedy_action = torch.argmax(action_prob)
        return sampled_action, action_prob, greedy_action

class CNN_Critic2(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_Critic2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels=32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )
        self.linear1 = nn.Linear(128, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Args:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32
    gamma = 0.99
    lr = 0.0001

    action_space = 36
    # action_space = 3
    state_space = 1600

args = Args()
device = 'cpu'

class PPO:
    clip_param = args.clip_param
    max_grad_norm = args.max_grad_norm
    ppo_update_time = args.ppo_update_time
    buffer_capacity = args.buffer_capacity
    batch_size = args.batch_size
    gamma = args.gamma
    action_space = args.action_space
    state_space = args.state_space
    lr = args.lr
    use_cnn = False

    def __init__(self, run_dir=None):
        super(PPO, self).__init__()
        self.args = args
        if self.use_cnn:
            self.actor_net = CNN_Actor(self.state_space, self.action_space)
            self.critic_net = CNN_Critic(self.state_space)
        else:
            self.actor_net = Actor(self.state_space, self.action_space).to(device)
            self.critic_net = Critic(self.state_space).to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)

        if run_dir is not None:
            self.writer = SummaryWriter(os.path.join(run_dir, "PPO training loss at {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))
        self.IO = True if (run_dir is not None) else False

    def select_action(self, state, train=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state).to(device)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
            # action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(
            device)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1)

                # Compute value function predictions
                V = self.critic_net(state[index].squeeze(1))

                # Compute advantages
                delta = Gt_index - V
                advantage = delta.detach()

                # Update actor network
                action_prob = self.actor_net(state[index].squeeze(1)).gather(1, action[index])  # new policy

                # Compute action loss
                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                action_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic network
                value_loss = F.smooth_l1_loss(V, Gt_index)

                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                if self.IO:
                    self.writer.add_scalar('loss/policy loss', action_loss.item(), self.training_step)
                    self.writer.add_scalar('loss/critic loss', value_loss.item(), self.training_step)

                self.training_step += 1

        self.clear_buffer()


    def clear_buffer(self):
        del self.buffer[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)

    def load(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        print("run_dir: ", run_dir)
        base_path = os.path.dirname(os.path.dirname(__file__))
        print("base_path: ", base_path)
        algo_path = os.path.join(base_path, 'models')
        run_path = os.path.join(algo_path, run_dir)
        run_path = os.path.join(run_path, 'trained_model')
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')


actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}


def my_controller(observation, action_space, is_act_continuous=False):

    model = PPO()
    load_model(model, episode=240)

    action_ctrl_raw, action_prob = model.select_action(observation['obs']['agent_obs'].flatten(), False)
    action_ctrl = actions_map[action_ctrl_raw]
    return [[action] for action in action_ctrl]


def load_model(model, episode):
    print(f'\nBegin to load model: ')
    base_path = os.path.dirname(os.path.dirname(__file__))
    print("base_path: ", base_path)
    model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
    model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
    print(f'Actor path: {model_actor_path}')
    print(f'Critic path: {model_critic_path}')

    if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
        actor = torch.load(model_actor_path, map_location=device)
        critic = torch.load(model_critic_path, map_location=device)
        model.actor_net.load_state_dict(actor)
        model.critic_net.load_state_dict(critic)
        print("Model loaded!")
    else:
        sys.exit(f'Model not founded!')
