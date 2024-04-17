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

def getGridPosition(total_num_of_points):
    x_range = [0.45, 0.55]
    y_range = [-0.05, 0.05]
    num_points = int(np.sqrt(total_num_of_points))
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    return points, num_points**2

def rendering(obj_list):
    
    cam_look_at = torch.tensor([0.5, 0.0, 0.0])
    cam_position = torch.tensor([0.5, 0.0, 10.0])
    camera = pyredner.Camera(position = cam_position,
                        look_at = cam_look_at,
                        up = torch.tensor([-1.0, 0.0, 0.0]),
                        fov = torch.tensor([2.291525676350207]), # in degree
                        clip_near = 1e-2, # needs to > 0
                        resolution = (heightmap_size, heightmap_size),
                        )
    scene = pyredner.Scene(camera = camera, objects = obj_list)
    chan_list = [pyredner.channels.depth]
    depth_img = pyredner.render_generic(scene, chan_list)
    # return depth_img.reshape(heightmap_size,heightmap_size)
    near = 0.09
    far = 0.010
    depth = near * far /(far - depth_img)
    heightmap = torch.abs(depth - torch.max(depth))
    heightmap =  heightmap*37821.71428571428 - 3407.3605408838816
    heightmap = torch.relu(heightmap)
    heightmap = torch.where(heightmap > 1.0, 6e-3, heightmap) 

    return heightmap.reshape(heightmap_size,heightmap_size)

def getPositions(file_path):
    all_numeric_values = []
    with open(file_path, 'r') as file:
        for line in file:
            values = [float(match) for match in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)]
            all_numeric_values.append(values)
    return all_numeric_values

def getGroundTruth(agent, 
                   states,
                   in_hands,
                   object_dir_list, # this variable must not change
                   xyz_position_list,
                   rot_mat_list,
                   scale_list,
                   device):
    
    states = states.unsqueeze(dim = 0).detach() # new variable
    in_hands = in_hands.unsqueeze(dim = 0).detach() # new variable
    object_list = []

    for idx,d in enumerate(object_dir_list):
        o = pyredner.load_obj(d, return_objects=True)[0]

        new_vertices = o.vertices.to(device).detach().clone() # new variable
        scale = scale_list[idx].clone().detach()

        new_vertices *= scale

        new_vertices = torch.matmul(new_vertices, rot_mat_list[idx].T.float())
        new_vertices[:,0:1] += xyz_position_list[idx][0]
        new_vertices[:,1:2] += xyz_position_list[idx][1]
        new_vertices[:,2:3] += xyz_position_list[idx][2]
        o.vertices = new_vertices.clone()

        object_list.append(o)


    tray_dir = "./tray.obj"
    tray = pyredner.load_obj(tray_dir, return_objects=True)[0]
    tray.vertices /= 1000
    tray.vertices[:,0:1] += 0.5
    tray.vertices[:,1:2] += 0.0
    tray.vertices[:,2:3] += 0.0
    object_list.append(tray)

    obs = rendering(obj_list=object_list).reshape(1,1,heightmap_size,heightmap_size)   
    q_value_maps, _, actions = agent.getEGreedyActionsAttack(states, in_hands, obs, 0)
    
    actions = actions.to(device)
    
    states = states.to(device)
    
    return q_value_maps, actions.double()

def collect_data(envs = None, agent = None, device = None, o_info = None):
    pyredner.set_print_timing(False)
    
    ori_x, ori_y = o_info

    states, in_hands, obs, object_dir_list, params = envs._resetAttack(np.array([ori_x, ori_y])) 
    states = states.unsqueeze(dim=0).detach()
    in_hands = in_hands.unsqueeze(dim=0).detach()
    obs = obs.unsqueeze(dim=0).detach()
    
    q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActionsAttack(states, in_hands, obs, 0, 0)
    actions_star = actions_star.to(device)
    states = states.to(device)
    actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
    actions_star = actions_star.reshape(4)
    states_, in_hands_, obs_, rewards, dones = envs.stepAttack(actions_star.detach(), auto_reset=True)
    
    f1=open("./training_data.txt","a")
    f1.write(str([object_index, ori_x, ori_y, actions_star.detach(), rewards]) + "\n")
    
    envs.setInitializedFalse()
    

def main(envs, agent, device, position_list):
    pyredner.set_print_timing(False)
    
    for idx, item in tqdm(enumerate(position_list), total=len(position_list)):        
        o, x, y, reward = item
        states, in_hands, obs, _, _ = envs._resetAttack(np.array([x, y]))
        states = states.unsqueeze(dim=0).detach()
        in_hands = in_hands.unsqueeze(dim=0).detach()
        obs = obs.unsqueeze(dim=0).detach()
        (q_value_maps, q2_output), actions_star_idx, actions_star = agent.getEGreedyActionsAttack(states, in_hands, obs, 0, 0)
        actions_star = actions_star.to(device).reshape(3)
        
        if type(actions_star) != torch.Tensor:
            actions_star = torch.tensor(actions_star).to(device)
        if type(reward) != torch.Tensor:
            reward = torch.tensor([reward]).to(device)
            
        out1 = torch.cat((q_value_maps.view(-1), q2_output.view(-1)), dim=0).to(device)
        out2 = torch.cat((actions_star, reward), dim=0).to(device)
        out3 = torch.cat((out1, out2),dim=0).to(device)
        
        torch.save(out3, f"dataset_1/{int(o)}_{idx}.pt")
        
        envs.setInitializedFalse()
    return True
        
if __name__ == '__main__':
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    agent = createAgent()
    agent.eval()
    if load_model_pre:
        agent.loadModel(load_model_pre) 
    file_path = './obejct_sochastic_position.txt'
    # positions = getPositions(file_path)
    total_num_of_points = 1000
    positions, total_num_of_points = getGridPosition(total_num_of_points=total_num_of_points)
    
    # p_dict = {}
    # for p in positions:
    #     if p[0] not in p_dict:
    #         p_dict[int(p[0])] = []
    #         p_dict[int(p[0])].append(p[1:3])
    #     else:
    #         p_dict[int(p[0])].append(p[1:3])
    # for i in p_dict:
    #     print(i, p_dict[i])
        
    # for i in p_dict:
    #     env_config['object_index'] = int(i)
    #     envs = EnvWrapper(num_processes, env, env_config, planner_config)
    #     print("object_index: ", i)
    #     info = p_dict[i]
    #     reward = main(envs, agent, device, info)
    
    for pos in tqdm(positions, desc="Collecting Data"):
        collect_data(envs, agent, device=device, o_info=pos)
    print("end")
    