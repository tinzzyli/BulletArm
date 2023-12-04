import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm

import torch
import pyredner

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from bulletarm_baselines.fc_dqn.scripts.create_agent import createAgent
from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.storage.buffer import QLearningBufferExpert, QLearningBuffer
from bulletarm import env_factory
from bulletarm_baselines.fc_dqn.utils.logger import Logger
from bulletarm_baselines.fc_dqn.utils.schedules import LinearSchedule
from bulletarm_baselines.fc_dqn.utils.torch_utils import rand_perlin_2d
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper


ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def getGridPosition(position_index = None):
    x_range = [0.45, 0.55]
    y_range = [-0.05, 0.05]
    num_points = 10
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    return points

def test():
    pyredner.set_print_timing(False)
    plt.style.use('default')
    test_episode = 100
    total = 0
    s = 0
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    agent = createAgent()
    agent.eval()
    if load_model_pre:
        agent.loadModel(load_model_pre) 
    
    gridPositions = getGridPosition()
    
    states, in_hands, obs, _, _ = envs._resetAttack(gridPositions[total])
    states = states.unsqueeze(dim=0).detach()
    in_hands = in_hands.unsqueeze(dim=0).detach()
    obs = obs.unsqueeze(dim=0).detach()
    step_times = []
    pbar = tqdm(total=test_episode)
    while total < 100:
        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActionsAttack(states, in_hands, obs, 0, 0)
        actions_star = actions_star.to(device)
        states = states.to(device)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        actions_star = actions_star.reshape(4)
        states_, in_hands_, obs_, rewards, dones = envs.stepAttack(actions_star.detach(), auto_reset=True)
        
        f1=open("./object_grid_position.txt","a")
        f1.write("index: " + str(object_index) + ", grid_pos: " + str(gridPositions[total]) + ", rewards: " + str(rewards) + "\n")
        
        if dones.sum():
            total += dones.sum().int().item()
        
        s += rewards.sum().int().item()
        if total<100:
            states_, in_hands_, obs_, _, _ = envs._resetAttack(gridPositions[total])
        
        states_ = states_.unsqueeze(dim=0).detach()
        in_hands_ = in_hands_.unsqueeze(dim=0).detach()
        obs_ = obs_.unsqueeze(dim=0).detach()

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)



        pbar.set_description(
            '{}/{}, SR: {:.3f}'
                .format(s, total, float(s) / total if total != 0 else 0)
        )
        pbar.update(dones.sum().int().item())
    return float(s) / total if total != 0 else 0

if __name__ == '__main__':
    sr_value = test()
    print(sr_value)
    print(object_index)
    f=open("./object_info.txt","a")
    f.write("index: " + str(object_index) + ", num: " + str(num_objects) + ", SR: " + str(sr_value) + "\n")