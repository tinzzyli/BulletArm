import torch
import os
import pickle
import copy
import numpy as np
import numpy.random as npr
from scipy.ndimage import median_filter
import skimage.transform as sk_transform

import pybullet as pb
import pybullet_data

import bulletarm.pybullet.utils.constants as constants
from bulletarm.pybullet.utils import transformations
import bulletarm.envs.configs as env_configs

from bulletarm.pybullet.robots.ur5_simple import UR5_Simple
from bulletarm.pybullet.robots.ur5_robotiq import UR5_Robotiq
from bulletarm.pybullet.robots.kuka import Kuka
from bulletarm.pybullet.robots.panda import Panda
from bulletarm.pybullet.utils.sensor import Sensor
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
import bulletarm.pybullet.utils.object_generation as pb_obj_generation
from bulletarm.pybullet.utils.constants import NoValidPositionException
from time import sleep

orientation = [pb.getQuaternionFromEuler([0., 0., -np.pi / 4])]
block_scale_range = (0.6, 0.7)
scale = npr.choice(np.arange(block_scale_range[0], block_scale_range[1]+0.01, 0.02))
position = np.array([0.5, -0.0])
for i in range(86):
    handle = pb_obj_generation.generateGraspNetObject(position, orientation, scale, i)
    print(handle.getAABB)