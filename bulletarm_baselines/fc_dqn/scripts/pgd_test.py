import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm
import datetime
import threading
import pyredner
import torch
import torch.nn as nn
from transforms3d import quaternions
import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append('./')
sys.path.append('..')
from PIL import Image
from bulletarm_baselines.fc_dqn.scripts.create_agent import createAgent
from bulletarm_baselines.fc_dqn.storage.buffer import QLearningBufferExpert, QLearningBuffer
from bulletarm_baselines.logger.logger import Logger
from bulletarm_baselines.logger.baseline_logger import BaselineLogger
from bulletarm_baselines.fc_dqn.utils.schedules import LinearSchedule
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper
from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.utils.torch_utils import augmentBuffer, augmentBufferD4
from bulletarm_baselines.fc_dqn.scripts.fill_buffer_deconstruct import fillDeconstructUsingRunner
import re

def getEntries(file_path):
    all_numeric_values = []

    with open(file_path, 'r') as file:
        for line in file:
            # Use regex to extract numeric values from the line
            values = [float(match) for match in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)]
            all_numeric_values.append(values)

    return all_numeric_values

def getIndex(entry):
    return entry[0]

def getReward(entry):
    return entry[1]

def getPos(entry):
    return np.array(entry[2:5])

def getAction(entry):
    return np.array(entry[5:8])

def pgd_test(envs, agent, iters=100, device = device, positions = None):
    pyredner.set_print_timing(False)
    _  = envs._resetAttack()


if __name__ == '__main__':
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    agent = createAgent(test=False)
    agent.eval()
    if load_model_pre:
        agent.loadModel(load_model_pre)
    file_path = '/Users/tingxi/Desktop/object_info_2.txt'  # Replace with the actual file path
    entries = getEntries(file_path)
    
    for e in entries:
        p = 
        reward = pgd_test(envs, agent, iters=100, device = device, positions = p)
    