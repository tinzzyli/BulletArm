import pybullet as pb
import numpy as np
import random
import os
import glob
import pyredner # pyredner will be the main Python module we import for redner.
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

class Sensor(object):
  def __init__(self, cam_pos, cam_up_vector, target_pos, target_size, near, far):
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

  def setCamMatrix(self, cam_pos, cam_up_vector, target_pos):
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=[cam_pos[0], cam_pos[1], cam_pos[2]],
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    self.proj_matrix = pb.computeProjectionMatrixFOV(70, 1, 0.001, 0.3)

  def getHeightmap(self, objs, object_index, size, scale):
    self.object_index = object_index
    self.objs = objs
    self.scale = scale

    def save_greyscale_image(img):
      """
      input: [128, 128] numpy.array 
      output: N/A
      """
      img = np.array(img)
      img = np.clip(img*255, 0 ,255).astype('uint8')
      filename = f'grayscale_{int(time.time())}.png'
      img = Image.fromarray(img)
      img.save("./bulletarm_baselines/fc_dqn/scripts/heightmapPNG/"+filename)
      time.sleep(1)

    # def store_pos(string, l):
    #   s = str(l)
    #   s = string + s
    #   with open("/Users/tingxi/BulletArm/bulletarm_baselines/fc_dqn/scripts/actions.txt", "a") as f:
    #     f.write(s+"\n")

    # def store_heightmap(heightmap):
    #   heightmap = np.array(heightmap)
    #   with open("/Users/tingxi/BulletArm/bulletarm_baselines/fc_dqn/scripts/heightmap.txt", "a") as f:
    #     np.savetxt(f, heightmap, delimiter=",")
    #     f.write("\n")      

    def setSingleObjPosition():
      """
      In ONE object scenario ONLY:
      set desired object index in config, e.g. 055
      """
      if self.object_index >= 10:
        object_index = "0"+str(self.object_index)
      else:
        object_index = "00"+str(self.object_index)
      
      dir = "./bulletarm/pybullet/urdf/object/GraspNet1B_object/"+object_index+"/convex.obj"
      o = pyredner.load_obj(dir, return_objects=True)
      newObj = o[0]

      """set quaternion"""
      orien = self.objs[0].getRotation()
      _x, _y, _z, _w = orien
      orien = _w, _x, _y, _z
      R = quaternions.quat2mat(orien)
      R = torch.Tensor(R)
      newObj.vertices = torch.matmul(newObj.vertices, R.T)

      """set scale"""
      newObj.vertices *= scale

      """set position"""
      x = self.objs[0].getXPosition()
      y = self.objs[0].getYPosition()
      z = self.objs[0].getZPosition()
      newObj.vertices[:,0:1] += x
      newObj.vertices[:,1:2] += y
      newObj.vertices[:,2:3] += z

      #store_pos("mean position: ", [newObj.vertices[:,0:1].mean(), newObj.vertices[:,1:2].mean(), newObj.vertices[:,2:3].mean()])
      #store_pos("original position: ", [x,y,z])
      """rendering() needs List[Object] as input"""
      return [newObj]

    def rendering(cam_pos, cam_up_vector, target_pos, fov, obj_list):
      
      cam_pos = torch.FloatTensor(cam_pos)
      cam_up_vector = torch.FloatTensor(cam_up_vector)
      target_pos = torch.FloatTensor(target_pos)
      fov = torch.tensor([fov], dtype=torch.float32)

      camera = pyredner.Camera(position = cam_pos,
                          look_at = target_pos,
                          up = cam_up_vector,
                          fov = fov, # in degree
                          clip_near = 1e-2, # needs to > 0
                          resolution = (128, 128)
                          )
      #print("cam_pos: ", cam_pos, "\ncam_up: ", cam_up_vector, "\ntar_pos: ", target_pos, "\nfov: ", fov)
      scene = pyredner.Scene(camera = camera, objects = obj_list)
      chan_list = [pyredner.channels.depth]
      img = pyredner.render_generic(scene, chan_list)
      img = np.array(img)
      """reshape [128,128,1] to [128,128]"""
      img = np.squeeze(img, axis=2)
      return img
    
    #===HERE IS THE DIFFERENTIABLE RENDERER===#
    redner_obj_list = setSingleObjPosition()
    img = rendering(self.cam_pos, self.cam_up_vector, self.target_pos, self.fov, redner_obj_list)
    #===HERE IS THE DIFFERENTIABLE RENDERER===#

    #===HERE IS THE ORIGINAL RENDERER===#
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    depth_img = np.array(image_arr[3])
    depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)
    depth = np.abs(depth - np.max(depth)).reshape(size, size)
    #===HERE IS THE ORIGINAL RENDERER===#

    # store_heightmap(img)
    # store_heightmap(depth*100.0)
    
    save_greyscale_image(img)
    save_greyscale_image(depth*100.0)

    """make value of each pixel in the same range of the original heightmap"""
    return img/100.0

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
  
