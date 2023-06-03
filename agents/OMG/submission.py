# -*- coding:utf-8  -*-
# Time  : 2023/5/20
# Authors: Osher Elhadad, Matan Shamir, Gili Gutfeld

"""
# =================================== Important =========================================
this agent is a pretrained PPO agent , which can fit any env in Jidi platform.
"""

import os
from rl_trainer.algo.ppo import PPO, device
import torch
import sys

actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}

def my_controller(observation, action_space, is_act_continuous=False):

    model = PPO()
    load_dir = 'run3'
    load_model(model, load_dir, 'OMG', episode=2000)

    action_ctrl_raw, action_prob = model.select_action(observation['obs']['agent_obs'].flatten(), False)
    action_ctrl = actions_map[action_ctrl_raw]
    return [[action] for action in action_ctrl]


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
