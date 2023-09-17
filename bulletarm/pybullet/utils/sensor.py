import pybullet as pb
import numpy as np
import random
import os
import glob
import torch # We also import PyTorch
import urllib
import zipfile
import random
import torchvision
import math
from typing import Optional, List
import sys
import time
from PIL import Image
from transforms3d import quaternions
from matplotlib.pyplot import imshow
from bulletarm.pybullet.objects import grasp_net_obj
from bulletarm.pybullet.objects.grasp_net_obj import GraspNetObject
import pyredner # pyredner will be the main Python module we import for redner.

class Sensor(object):
  def __init__(self, cam_pos, cam_up_vector, target_pos, target_size, near, far):
    # print(cam_pos, cam_up_vector, target_pos, target_size, near, far)
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=cam_pos,
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    
    self.cam_pos = cam_pos
    self.cam_up_vector = cam_up_vector
    self.target_pos = target_pos
    self.near = near
    self.far = far
    self.fov = np.degrees(2 * np.arctan((target_size / 2) / self.far))
    self.proj_matrix = pb.computeProjectionMatrixFOV(self.fov, 1, self.near, self.far)

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    self.device = torch.device("cuda")

    

  def setCamMatrix(self, cam_pos, cam_up_vector, target_pos):
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=[cam_pos[0], cam_pos[1], cam_pos[2]],
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    self.proj_matrix = pb.computeProjectionMatrixFOV(70, 1, 0.001, 0.3)

  def importSingleObject(self, scale):
    
    # if self.object_index >= 10:
    #   object_index = "0"+str(self.object_index)
    # else:
    #   object_index = "00"+str(self.object_index)
    # dir = "./bulletarm/pybullet/urdf/object/GraspNet1B_object/"+object_index+"/convex.obj"
    dir = "./bulletarm/pybullet/urdf/object/GraspNet1B_object/055/convex.obj"

    o = pyredner.load_obj(dir, return_objects=True, device=self.device)
    new_obj = o[0]

    orien = self.objs[0].getRotation()
    _x, _y, _z, _w = orien
    orien = _w, _x, _y, _z
    quat_rotation = np.array([_w, _x, _y, _z])
    R = quaternions.quat2mat(orien)
    R = torch.Tensor(R)
    R = R.to(self.device)
    R = R.float()

    x = self.objs[0].getXPosition()
    y = self.objs[0].getYPosition()
    z = self.objs[0].getZPosition()
    xyz_position = np.array([x, y, z])

    new_vertices = new_obj.vertices.clone()
    new_vertices = new_vertices.to(self.device)
    new_vertices = new_vertices.float()
    ORI_OBJECT = o[0]
    new_vertices *= scale
    scale = np.copy(scale)
    new_vertices = torch.matmul(new_vertices, R.T)
    new_vertices[:,0:1] += x
    new_vertices[:,1:2] += y
    new_vertices[:,2:3] += z
    new_obj.vertices = new_vertices

    # print(scale, quat_rotation, xyz_position)

    return [new_obj], [ORI_OBJECT], [xyz_position, quat_rotation, scale]
  
  def rendering(self, cam_pos, cam_up_vector, target_pos, fov, obj_list, size):
    
    cam_pos = torch.FloatTensor(cam_pos)
    cam_up_vector = torch.FloatTensor(cam_up_vector)
    target_pos = torch.FloatTensor(target_pos)
    fov = torch.tensor([fov], dtype=torch.float32)

    cam_pos = cam_pos.to(self.device)
    cam_up_vector = cam_up_vector.to(self.device)
    target_pos = target_pos.to(self.device)
    fov = fov.to(self.device)

    camera = pyredner.Camera(position = cam_pos,
                        look_at = target_pos,
                        up = cam_up_vector,
                        fov = fov, # in degree
                        clip_near = 1e-2, # needs to > 0
                        resolution = (size, size)
                        )
    scene = pyredner.Scene(camera = camera, objects = obj_list)
    chan_list = [pyredner.channels.depth]
    depth_img = pyredner.render_generic(scene, chan_list, device=self.device)
    near = 0.09
    far = 0.010
    depth = near * far /(far - depth_img)
    heightmap = torch.abs(depth - torch.max(depth))
    heightmap =  heightmap*37821.71428571428 - 3407.3605408838816
    heightmap = torch.relu(heightmap)
    heightmap = torch.where(heightmap > 1.0, 6e-3, heightmap) 
    return heightmap.reshape(128,128)
  
  def getHeightmap(self, objs, object_index, size, scale):
    self.object_index = object_index
    self.objs = objs
    self.scale = scale
    self.size = size      

    #===HERE IS THE DIFFERENTIABLE RENDERER===#
    rendering_list, _, _ = self.importSingleObject(scale=self.scale)

    tray_dir = "./tray.obj"
    t = pyredner.load_obj(tray_dir, return_objects=True, device=self.device)
    tray = t[0]   
    tray.vertices = tray.vertices.to(self.device)
    tray.vertices /= 1000
    tray.vertices[:,0:1] +=  0.5
    tray.vertices[:,1:2] += -0.0
    tray.vertices[:,2:3] += -0.0
    rendering_list.append(tray)

    img = self.rendering(self.cam_pos, self.cam_up_vector, self.target_pos, self.fov, rendering_list, self.size)
    img = img.cpu().detach().numpy()
    #===HERE IS THE DIFFERENTIABLE RENDERER===#

    #===HERE IS THE ORIGINAL RENDERER===#
    # image_arr = pb.getCameraImage(width=self.size, height=self.size,
    #                               viewMatrix=self.view_matrix,
    #                               projectionMatrix=self.proj_matrix,
    #                               renderer=pb.ER_TINY_RENDERER)
    # depth_img = np.array(image_arr[3])
    # depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)
    # depth = np.abs(depth - np.max(depth)).reshape(self.size, self.size)
    #===HERE IS THE ORIGINAL RENDERER===#
    return img
  
  def getHeightmapAttack(self, objs, object_index, size, scale):
    self.object_index = object_index
    self.objs = objs
    self.scale = scale
    self.size = size      
    rendering_list, ORI_OBJECT_LIST, params = self.importSingleObject(scale=self.scale)
    tray_dir = "./tray.obj"
    t = pyredner.load_obj(tray_dir, return_objects=True, device=self.device)
    tray = t[0]   
    tray.vertices /= 1000
    tray.vertices[:,0:1] +=  0.5
    tray.vertices[:,1:2] += -0.0
    tray.vertices[:,2:3] += -0.0
    rendering_list.append(tray)
    img = self.rendering(self.cam_pos, self.cam_up_vector, self.target_pos, self.fov, rendering_list, self.size)
    img = img.cpu().detach().numpy()
    return img, ORI_OBJECT_LIST, params
  


  def getRGBImg(self, size, objs):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    rgb_img = np.moveaxis(image_arr[2][:, :, :3], 2, 0) / 255
    return rgb_img

  def getDepthImg(self, size):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    depth_img = np.array(image_arr[3])
    depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)
    return depth.reshape(size, size)

  def getPointCloud(self, size, to_numpy=True):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix, 
                                  projectionMatrix=self.proj_matrix, 
                                  renderer=pb.ER_TINY_RENDERER)
    depthImg = np.asarray(image_arr[3])

    # https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
    projectionMatrix = np.asarray(self.proj_matrix).reshape([4,4],order='F')
    viewMatrix = np.asarray(self.view_matrix).reshape([4,4],order='F')
    tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
    pixel_pos = np.mgrid[0:size, 0:size]
    pixel_pos = pixel_pos/(size/2) - 1
    pixel_pos = np.moveaxis(pixel_pos, 1, 2)
    pixel_pos[1] = -pixel_pos[1]
    zs = 2*depthImg.reshape(1, size, size) - 1
    pixel_pos = np.concatenate((pixel_pos, zs))
    pixel_pos = pixel_pos.reshape(3, -1)
    augment = np.ones((1, pixel_pos.shape[1]))
    pixel_pos = np.concatenate((pixel_pos, augment), axis=0)
    position = np.matmul(tran_pix_world, pixel_pos)
    pc = position / position[3]
    points = pc.T[:, :3]

    # if to_numpy:
      # points = np.asnumpy(points)
    return points
  
