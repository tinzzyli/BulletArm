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

import re

def test(ori_pos, ori_reward):
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
    
    states, in_hands, obs = envs._reset(ori_pos[total])
    states = states.unsqueeze(dim=0).detach()
    in_hands = in_hands.unsqueeze(dim=0).detach()
    obs = obs.unsqueeze(dim=0).detach()
    step_times = []
    pbar = tqdm(total=test_episode)
    while total < 10000:
        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, 0, 0)
        actions_star = actions_star.to(device)
        states = states.to(device)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        actions_star = actions_star.reshape(4)
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star.detach(), auto_reset=True)
        
        f1=open("./deter_stoch_position.txt","a")
        f1.write("index: " + str(object_index) + ", pos: " + str(ori_pos[total]) + ", ori_reward: " + str(ori_reward[total]) +  ", rewards: " + str(rewards) + "\n")
        
        if dones.sum():
            total += dones.sum().int().item()
        
        s += rewards.sum().int().item()
        
        if total<10000:
            envs.setInitializedFalse()
            states_, in_hands_, obs_ = envs._reset(ori_pos[total])
        
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

def read_numeric_values(file_path):
    # Initialize a list to store the numeric values
    numeric_values = []
    
    pos_values = []
    reward_values = []

    # Open the file for reading
    with open(file_path, 'r') as file:
        # Read each line in the file
        for line in file:
            numbers = re.findall(r'-?\d+\.\d+|-?\d+', line)
            # 将匹配到的数字转换为浮点数或整数，并添加到列表中
            # numeric_values.append([float(num) if '.' in num else int(num) for num in numbers])
            numeric = [float(num) if '.' in num else int(num) for num in numbers]
            pos_values.append(numeric[1:3])
            reward_values.append(numeric[3])
            
    return pos_values, reward_values

# def getori_posFromValues(file_path):
#     values = read_numeric_values(file_path)
#     pos = []
#     for v in values:
#         pos.append(v[4:6])
#     return pos

# def getRewardFromValues(file_path):
#     values = read_numeric_values(file_path)
#     reward = []
#     for v in values:
#         reward.append(v[-1])
#     return reward

if __name__ == '__main__':
    file_path = './100_object_original_position.txt'
    
    ori_pos, ori_reward = read_numeric_values(file_path)
    ori_pos = ori_pos[object_index*100: object_index*100 + 10000]
    ori_reward = ori_reward[object_index*100: object_index*100 + 10000]
    
    sr_value = test(ori_pos, ori_reward)
    print(sr_value)
    print(object_index)
    f=open("./object_transfer_info.txt","a")
    f.write("index: " + str(object_index) + ", num: " + str(num_objects) + ", SR: " + str(sr_value) + "\n")