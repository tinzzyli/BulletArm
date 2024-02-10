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

def unify(R):
    for idx,r in enumerate(R):
        mod = r[0]**2 + r[1]**2 + r[2]**2
        r /= torch.sqrt(mod)
        R[idx] = r
    return R

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

def read_numeric_values(file_path):
    all_numeric_values = []

    with open(file_path, 'r') as file:
        for line in file:
            # Use regex to extract numeric values from the line
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

def loss_check(envs = None, agent = None, iters=100, device = None, o_info = None):
    pyredner.set_print_timing(False)
    _, ori_x, ori_y, ori_reward = o_info
    
    # states, in_hands, obs, object_dir_list, params = envs._resetAttack(np.array([ori_x, ori_y])) 
    # original_xyz_position_list, original_rot_mat_list, scale_list = params

    # num_objects = len(object_dir_list)
    # xyz_position_list = copy.deepcopy(original_xyz_position_list)
    # rot_mat_list = copy.deepcopy(original_rot_mat_list)
    # scale_list = copy.deepcopy(scale_list)
    
    # leaf_tensor = xyz_position_list[0][:2].clone().to(device)
    # leaf_tensor.requires_grad = True
    # xyz_position_list[0][:2] = leaf_tensor
    
    # q_value_maps, actions = getGroundTruth(agent = agent,
    #                             states = states,
    #                             in_hands = in_hands,
    #                             object_dir_list = object_dir_list,
    #                             xyz_position_list = xyz_position_list,
    #                             rot_mat_list = rot_mat_list,
    #                             scale_list = scale_list,
    #                             device = device)
    
    # actions = actions[0][:2].to(device)
    
    """this block of code is for LOSS"""

    ######
    ori_pos = torch.tensor(o_info[1:3]).requires_grad_()
    
    states, in_hands, obs, _, _ = envs._resetAttack(ori_pos)
    
    states = states.unsqueeze(dim=0)
    in_hands = in_hands.unsqueeze(dim=0)
    obs = obs.unsqueeze(dim=0)
    
    q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActionsAttack(states, in_hands, obs, 0, 0)
    actions_star = actions_star.to(device).double()
    actions = actions_star[0][:2].to(device)
    states = states.to(device)
    actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
    actions_star = actions_star.reshape(4)
    states_, in_hands_, obs_, rewards, dones = envs.stepAttack(actions_star.detach(), auto_reset=True)
    
    
    

    mse_loss = nn.MSELoss()
    q_max_value = torch.max(q_value_maps)
    target_q_value_maps = torch.where(q_value_maps < q_max_value, 0, 1.0)
    loss = - mse_loss(q_value_maps, target_q_value_maps) 
    grad = torch.autograd.grad(outputs=loss, 
                        inputs=ori_pos, 
                        grad_outputs=None, 
                        allow_unused=False, 
                        retain_graph=False, 
                        create_graph=False)
    x_grad, y_grad = grad[0].to(device)
    
    envs.setInitializedFalse()
    
    return loss, grad, actions
    
if __name__ == '__main__':
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    agent = createAgent(test=False)
    agent.eval()
    if load_model_pre:
        agent.loadModel(load_model_pre)
        
    s = 0.
    file_path = './object_original_position.txt'
    all_values = read_numeric_values(file_path)
    object_info = all_values[object_index*100: object_index*100 + 100]
    print("object_index: ", object_index)
    for i in range(100):
        o_info = object_info[i]
        loss, grad, actions = loss_check(envs, agent, iters=100, device = device, o_info = o_info)
        actions = actions.detach().cpu()
        difference = np.sqrt((o_info[1] - actions[0])**2 + (o_info[2] - actions[1])**2)
        f=open('./loss_check.txt', 'a')
        f.write("Info: " + str(o_info) + ", Action: "+ str(actions) + ", Grad: " + str(grad) + ", Loss: " + str(loss) + ", Difference: "+ str(difference) + "\n")
    print("end")
    