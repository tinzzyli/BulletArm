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

def pgd_attack(envs = None, agent = None, epsilon_1 = 0.0005, epsilon_2 = 0.0005, alpha_1 = 0.02, alpha_2 = 0.1, iters=100, device = None, o_info = None):
    pyredner.set_print_timing(False)
    
    _, ori_x, ori_y, ori_reward = o_info

    states, in_hands, obs, object_dir_list, params = envs._resetAttack(np.array([ori_x, ori_y])) 
    original_xyz_position_list, original_rot_mat_list, scale_list = params

    num_objects = len(object_dir_list)
    xyz_position_list = copy.deepcopy(original_xyz_position_list)
    rot_mat_list = copy.deepcopy(original_rot_mat_list)
    scale_list = copy.deepcopy(scale_list)    
    mse_loss = nn.MSELoss()


    for iter in range(iters):
        leaf_tensor = xyz_position_list[0][:2].clone().to(device)
        leaf_tensor.requires_grad = True
        xyz_position_list[0][:2] = leaf_tensor
        # rot_mat_list[0].requires_grad = True


        _, actions = getGroundTruth(agent = agent,
                                    states = states,
                                    in_hands = in_hands,
                                    object_dir_list = object_dir_list,
                                    xyz_position_list = xyz_position_list,
                                    rot_mat_list = rot_mat_list,
                                    scale_list = scale_list,
                                    device = device)
        actions = actions[0][:2].to(device)
        
        """ attack on position """
        loss = - mse_loss(leaf_tensor, actions)      
        grad = torch.autograd.grad(outputs=loss, 
                            inputs=leaf_tensor, 
                            grad_outputs=None, 
                            allow_unused=False, 
                            retain_graph=False, 
                            create_graph=False)
        x_grad, y_grad = grad[0].to(device)
        xyz_position = xyz_position_list[0].detach().clone().to(device)
        # rot_mat = rot_mat_list[0].detach().clone().to(device)
        x,y,z = xyz_position.clone().detach().to(device)
        x_eta = torch.clamp(x_grad, min = -epsilon_1,  max = epsilon_1).to(device)
        y_eta = torch.clamp(y_grad, min = -epsilon_1,  max = epsilon_1).to(device)
        xyz_position = torch.tensor([
            torch.clamp(x + x_eta, min = original_xyz_position_list[0][0].to(device) - alpha_1, max = original_xyz_position_list[0][0].to(device) + alpha_1),
            torch.clamp(y + y_eta, min = original_xyz_position_list[0][1].to(device) - alpha_1, max = original_xyz_position_list[0][1].to(device) + alpha_1),
            z]).to(device)
        """ attack on position """

        """ attack on rotation"""
        # rot_eta = rot_grad.sign() * epsilon_2
        # rot_mat = torch.clamp(unify(rot_mat + rot_eta), min = original_rot_mat_list[0] - alpha_2, max = original_rot_mat_list[0] + alpha_2)
        """ attack on rotation"""

        xyz_position_list[0] = xyz_position.detach().clone().to(device)
        # rot_mat_list[0] = rot_mat.detach().clone().to(device)
        # scale = scale.detach().clone()
        
        
    _, _, _, _, params = envs._resetAttack(xyz_position_list[0].cpu().numpy()) 
    _xyz_position_list, _rot_mat_list, _scale_list = params
    xyz_position_list = copy.deepcopy(_xyz_position_list)
    rot_mat_list = copy.deepcopy(_rot_mat_list)
    scale_list = copy.deepcopy(_scale_list)    

    # xyz_position_list = copy.deepcopy(original_xyz_position_list)
    # rot_mat_list = copy.deepcopy(original_rot_mat_list)
    # scale_list = copy.deepcopy(scale_list)  
    
    _, actions = getGroundTruth(agent = agent,
                                states = states,
                                in_hands = in_hands,
                                object_dir_list = object_dir_list,
                                xyz_position_list = xyz_position_list,
                                rot_mat_list = rot_mat_list,
                                scale_list = scale_list,
                                device = device)
    
    actions = actions.to(device)
    states = states.to(device).unsqueeze(dim=0)
    actions = torch.cat((actions, states.unsqueeze(1)), dim=1)
    actions = actions.reshape(4)
    _, _, _, reward, _ = envs.stepAttack(actions.detach())

    f=open("./object_pgd_attack_positioin.txt","a")
    f.write("index: " + str(object_index) + ", ori_reward: " + str(ori_reward) + ", ori_pos_1: " + str([ori_x, ori_y]) + ", ori_pos_2: " + str(original_xyz_position_list[0]) + ", pos: " + str(xyz_position_list[0]) + ", actions: " + str(actions) + ", reward: " + str(reward) + "\n")

    return reward



def heightmapAttack(envs, agent, epsilon = 1e-5, alpha = 4e-4, iters = 5):

    states, in_hands, obs = envs.reset() 
    loss_function = nn.MSELoss()
    obs = obs.unsqueeze(dim = 0).detach() # new variable
    states = states.unsqueeze(dim = 0).detach() # new variable
    in_hands = in_hands.unsqueeze(dim = 0).detach() # new variable

    original_obs = obs.clone().detach()
    _, _, target = agent.getEGreedyActionsAttack(states, in_hands, obs, 0)

    for _ in range(iters):
        obs.requires_grad = True
        _, _, actions = agent.getEGreedyActionsAttack(states, in_hands, obs, 0)
        loss = loss_function(actions, target)
        grad = torch.autograd.grad(loss, obs)[0]

        eta = grad.sign() * epsilon
        obs = obs.detach()
        obs = torch.clamp(obs + eta, min = original_obs - alpha, max = original_obs + alpha)
    
    _, _, actions = agent.getEGreedyActionsAttack(states, in_hands, obs, 0)
    actions = actions.to(device)
    states = states.to(device)
    actions = torch.cat((actions, states.unsqueeze(1)), dim=1)
    actions = actions.reshape(4)
    _, _, _, reward, _ = envs.step(actions.detach())
    
    return reward

def read_numeric_values(file_path):
    all_numeric_values = []

    with open(file_path, 'r') as file:
        for line in file:
            # Use regex to extract numeric values from the line
            values = [float(match) for match in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)]
            all_numeric_values.append(values)

    return all_numeric_values

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
    #object_info is a list of lists, each sub-list is in such manner:
    #[78.0, 0.4895, -0.0302, 1.0]
    print("object_index: ", object_index)
    for i in range(100):
        o_info = object_info[i]
        reward = pgd_attack(envs, agent, iters=100, device = device, o_info = o_info)
        s += reward
    sr_value = float(s)/100.0
    print("sr_value: ", sr_value)
    f=open("./object_pgd_attack_info.txt","a")
    f.write("index: " + str(object_index) + ", num: " + str(num_objects) + ", SR: " + str(sr_value) + "\n")
    print("end")
    