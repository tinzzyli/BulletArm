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
from bulletarm_baselines.fc_dqn.scripts.create_agent import createAgent
from bulletarm_baselines.fc_dqn.storage.buffer import QLearningBufferExpert, QLearningBuffer
from bulletarm_baselines.logger.logger import Logger
from bulletarm_baselines.logger.baseline_logger import BaselineLogger
from bulletarm_baselines.fc_dqn.utils.schedules import LinearSchedule
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper
from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.utils.torch_utils import augmentBuffer, augmentBufferD4
from bulletarm_baselines.fc_dqn.scripts.fill_buffer_deconstruct import fillDeconstructUsingRunner

def runDemo():
  envs = EnvWrapper(num_processes, env, env_config, planner_config)
  # agent = createAgent(test=False)
  # agent.eval()
  done = False
  cnt = 0
  for _ in range(500):
    states, in_hands, obs, _, _ = envs.resetAttack()
    states = states.unsqueeze(dim=0)
    in_hands = in_hands.unsqueeze(dim=0)
    obs = obs.unsqueeze(dim=0)
    plan_actions = envs.getNextAction()
    # plan_actions = plan_actions.unsqueeze(dim=0)
    plan_actions = plan_actions.to(device)
    # states = states.to(device)
    # in_hands = in_hands.to(device)
    # obs = obs.to(device)
    # planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
    # planner_actions_star = planner_actions_star.to(device)
    # planner_actions_star = torch.cat((planner_actions_star, states.unsqueeze(1)), dim=1)
    # planner_actions_star = planner_actions_star.reshape(4)

    _, _, _, rewards, dones = envs.stepAttack(plan_actions, auto_reset=True)
    print(rewards, dones)
    if rewards == 1:
      cnt += 1
  print(cnt)
  env.close()

if __name__ == '__main__':
  runDemo()
