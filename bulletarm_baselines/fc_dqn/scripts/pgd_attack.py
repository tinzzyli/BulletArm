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

def rendering(obj_list):
    
    cam_look_at = torch.tensor([0.5, 0.0, 0.0])
    cam_position = torch.tensor([0.5, 0.0, 10.0])
    camera = pyredner.Camera(position = cam_position,
                        look_at = cam_look_at,
                        up = torch.tensor([-1.0, 0.0, 0.0]),
                        fov = torch.tensor([2.291525676350207]), # in degree
                        clip_near = 1e-2, # needs to > 0
                        resolution = (128, 128),
                        )
    scene = pyredner.Scene(camera = camera, objects = obj_list)
    chan_list = [pyredner.channels.depth]
    depth_img = pyredner.render_generic(scene, chan_list)
    # return depth_img.reshape(128,128)
    near = 0.09
    far = 0.010
    depth = near * far /(far - depth_img)
    heightmap = torch.abs(depth - torch.max(depth))
    heightmap =  heightmap*37821.71428571428 - 3407.3605408838816
    heightmap = torch.relu(heightmap)
    heightmap = torch.where(heightmap > 1.0, 6e-3, heightmap) 

    return heightmap.reshape(128,128)

def getGroundTruth(agent, 
                   states,
                   in_hands,
                   object_dir_list, # this variable must not change
                   xyz_position,
                   rot_mat,
                   scale,
                   device):
    
    states = states.unsqueeze(dim = 0).detach() # new variable
    in_hands = in_hands.unsqueeze(dim = 0).detach() # new variable
    object_list = []

    for idx, dir in enumerate(object_dir_list):
        obj = pyredner.load_obj(dir, return_objects=True)[0]

        new_vertices = obj.vertices.to(device).detach() # new variable

        scale = scale.clone().detach()

        new_vertices *= scale

        rot_mat_T = rot_mat.T.float()
        new_vertices = torch.matmul(new_vertices, rot_mat_T)

        new_vertices[:,0:1] += xyz_position[0]
        new_vertices[:,1:2] += xyz_position[1]
        new_vertices[:,2:3] += xyz_position[2]
        obj.vertices = new_vertices.clone()

        object_list.append(obj)
    
    tray_dir = "./tray.obj"
    tray = pyredner.load_obj(tray_dir, return_objects=True)[0]
    tray.vertices /= 1000
    tray.vertices[:,0:1] += 0.5
    tray.vertices[:,1:2] += 0.0
    tray.vertices[:,2:3] += 0.0
    object_list.append(tray)

    obs = rendering(obj_list=object_list).reshape(1,1,128,128)   
    q_value_maps, _, actions = agent.getEGreedyActionsAttack(states, in_hands, obs, 0)
    
    actions = actions.to(device)
    states = states.to(device)
    
    return q_value_maps, actions

def pgd_attack(envs, agent, epsilon_1 = 0.002, epsilon_2 = 0.002, alpha_1 = 0.02, alpha_2 = 0.02, iters=10, device = None):
    
    l = logging.getLogger('my_logger')
    l.setLevel(logging.DEBUG)
    log_dir = './outputAttack/VanillaPGD'  
    os.makedirs(log_dir, exist_ok=True)  
    log_file_name = f'auto_generated_log_{int(time.time())}.log'
    log_file_path = os.path.join(log_dir, log_file_name)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    l.addHandler(file_handler)
    # to avoid potential errors, run code in single process

    states, in_hands, obs, object_dir_list, object_param_list = envs.resetAttack() 
    original_xyz_position, original_rot_mat, scale = object_param_list[0]
    xyz_position = original_xyz_position.clone().detach()
    rot_mat = original_rot_mat.clone().detach()
    scale = scale.clone().detach()

    _, target = getGroundTruth(agent = agent,
                               states = states,
                               in_hands = in_hands,
                               object_dir_list = object_dir_list,
                               xyz_position = xyz_position,
                               rot_mat = rot_mat,
                               scale = scale,
                               device = device)

    l.info('\n device: '+str(device)+
           '\n epsilon_1: '+str(epsilon_1)+
           '\n epsilon_2: '+str(epsilon_2)+
           '\n iters: '+str(iters))
    
    loss_function = nn.MSELoss()

    for iter in range(iters):
        l.info('Iteration '+str(iter)+'/'+str(iters))

        xyz_position.requires_grad = True
        rot_mat.requires_grad = True

        q_value_maps, actions = getGroundTruth(agent = agent, 
                                               states = states,
                                               in_hands = in_hands,
                                               object_dir_list = object_dir_list,
                                               xyz_position = xyz_position,
                                               rot_mat = rot_mat,
                                               scale = scale,
                                               device = device)

        """ attack on position """
        loss = loss_function(target, actions)      
        grad = torch.autograd.grad(outputs=loss, 
                                   inputs=(xyz_position, rot_mat), 
                                   grad_outputs=None, 
                                   allow_unused=False, 
                                   retain_graph=True, 
                                   create_graph=False)
        x_grad, y_grad, _ = grad[0]
        rot_grad = grad[1]

        print("loss: ", loss)
        print("grad: ", grad)
        print("actions ", actions)

        x,y,z = xyz_position.clone().detach()
        x_eta = torch.clamp(x_grad, min = -epsilon_1,  max = epsilon_1)
        y_eta = torch.clamp(y_grad, min = -epsilon_1,  max = epsilon_1)
        # coordinate boudary of the object, please do not change these values
        # valid range of x and y is 0.2 while for z the range is 0.000025
        # accumulated change should not exceed the boundaries

        adv_position = torch.tensor([
            torch.clamp(x + x_eta, min = original_xyz_position[0] - alpha_1, max = original_xyz_position[0] + alpha_1),
            torch.clamp(y + y_eta, min = original_xyz_position[1] - alpha_1, max = original_xyz_position[1] + alpha_1),
            z])
        """ attack on position """

        """ attack on rotation"""
        rot_eta = rot_grad.sign() * epsilon_2
        rot_mat = rot_mat.detach()
        rot_mat = torch.clamp(rot_mat + rot_eta, min = original_rot_mat - alpha_2, max = original_rot_mat + alpha_2)
        """ attack on rotation"""

        l.debug("gradient: "+str([x_grad, y_grad]))
        l.debug("OG position: "+str(xyz_position))
        l.debug("eta: "+str([x_eta, y_eta]))
        l.debug("ADV position: "+str(adv_position)) 
        # l.debug("successful grasp: "+str(success))    
        l.debug("actions: "+str(actions))  
        l.debug("rotation: "+str(rot_mat))
        # print("successful grasp: "+str(success))
        print("\n", adv_position, "\n")
        print("\n", rot_mat, "\n")
        
        xyz_position = adv_position.clone().detach()
        rot_mat = rot_mat.clone().detach()
        scale = scale.clone().detach()
    #end of loop
    
    _, actions = getGroundTruth(agent = agent,
                                states = states,
                                in_hands = in_hands,
                                object_dir_list = object_dir_list,
                                xyz_position = xyz_position,
                                rot_mat = rot_mat,
                                scale = scale,
                                device = device)
    
    _, _, actions = agent.getEGreedyActionsAttack(states, in_hands, obs, 0)
    actions = actions.to(device)
    states = states.to(device)
    actions = torch.cat((actions, states.unsqueeze(1)), dim=1)
    actions = actions.reshape(4)
    _, _, _, reward, _ = envs.step(actions.detach())
    
    l.removeHandler(file_handler)
    logging.shutdown()

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


if __name__ == '__main__':
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    agent = createAgent(test=False)
    agent.eval()
    # agent.loadModel("/content/drive/MyDrive/my_archive/ck3/snapshot")
    for _ in range(10):
        reward = pgd_attack(envs, agent, iters=5, device = device)
        print(reward)
    print("end")