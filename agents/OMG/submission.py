# -*- coding:utf-8  -*-
# Time  : 2023/5/20 下午4:14
# Authors: Osher Elhadad, Matan Shamir, Gili Gutfeld

"""
# =================================== Important =========================================
this agent is a pretrained PPO agent , which can fit any env in Jidi platform.
"""

import os
from rl_trainer.algo.ppo import PPO, device
import torch
import sys


def my_controller(observation, action_space, is_act_continuous=False):

    model = PPO()
    load_dir = 'run5'
    load_model(model, load_dir, 'OMG', episode=20)

    agent_action = []
    for i in range(len(action_space)):
        action_ = model.select_action(observation['obs']['agent_obs'].flatten(), False)
        agent_action.append(action_)
    return agent_action


def load_model(model, run_dir, agent, episode):
    print(f'\nBegin to load model: ')
    print("run_dir: ", run_dir)
    base_path = os.path.dirname(os.path.dirname(__file__))
    print("base_path: ", base_path)
    agent_path = os.path.join(base_path, agent)
    algo_path = os.path.join(agent_path, 'models')
    run_path = os.path.join(algo_path, run_dir)
    run_path = os.path.join(run_path, 'trained_model')
    model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
    model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
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
