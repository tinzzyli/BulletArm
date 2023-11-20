import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm

import torch

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

from PIL import Image

def test():
    plt.style.use('default')
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    agent = createAgent()
    agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)
    agent.eval()
    states, in_hands, obs = envs.reset()
    test_episode = 1000
    total = 0
    s = 0
    step_times = []
    pbar = tqdm(total=test_episode)
    
    saveImage(obs, "obs")

    while total < 1000:
        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, 0, 0)

        saveImage(q_value_maps, "q_map")
        saveMatrix(q_value_maps, "q_map")

        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)

        raise ValueError("just a break point")

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = in_hands_

        s += rewards.sum().int().item()

        if dones.sum():
            total += dones.sum().int().item()

        pbar.set_description(
            '{}/{}, SR: {:.3f}'
                .format(s, total, float(s) / total if total != 0 else 0)
        )
        pbar.update(dones.sum().int().item())

def saveImage(obss, text):
    text = str(text)
    for idx,obs in enumerate(obss):
        numpy_array = obs.squeeze().numpy()
        normalized_tensor = ((numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min()) * 255).astype(np.uint8)
        image = Image.fromarray(normalized_tensor)
        image.save(f'{text}_{idx}.png')

def saveMatrix(mats, text):
    text = str(text)
    for idx,mat in enumerate(mats):
        numpy_array = mat.squeeze().numpy()
        np.savetxt(f'{text}_{idx}.txt', numpy_array, fmt='%f')


if __name__ == '__main__':
    test()