import torch
import numpy as np
import math
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


def gaussian_noise(x, severity):
    s=torch.tensor([1., 1.25, 1.5, 1,75, 2.])[severity-1]
    mean = torch.mean(x)
    std = torch.std(x)
    gaussian_noise = torch.normal(mean=mean, std=std, size=x.shape) * s
    return x + gaussian_noise

def poisson_noise(x, severity):
    s = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.])[severity-1]
    std_deviation = torch.std(x)
    poisson_noise = torch.poisson(x + s) * std_deviation
    return x + poisson_noise

def salt_pepper_noise(x, severity):
    total_pixel = x.numel()
    s = torch.tensor([0.025, 0.05, 0.075, 0.1, 0.125])[severity-1]
    salt_num = (total_pixel * s).int()
    pepper_num = (total_pixel * s).int()
    salt_coords = torch.randint(0,total_pixel,(salt_num,))
    pepper_coords = torch.randint(0,total_pixel,(pepper_num,))
    flat_x = x.reshape(total_pixel)
    flat_x[salt_coords] = torch.min(x)
    flat_x[pepper_coords] = torch.max(x)
    x = flat_x.reshape(x.shape)
    return x

def rotation(x, severity):
    angle_degrees = torch.tensor([5., 10., 15., 20., 25])[severity-1]
    assert len(x.shape) == 4
    assert x.shape[1] == 1
    for idx in range(x.shape[0]):

        if random.choice([True, False]):
            angle_degrees *= -1

        heightmap = x[idx][0].clone()
        rotated_heightmap = F.affine_grid(torch.tensor([[math.cos(math.radians(angle_degrees)), math.sin(math.radians(angle_degrees)), 0],
                                                [-math.sin(math.radians(angle_degrees)), math.cos(math.radians(angle_degrees)), 0]]).unsqueeze(0),
                                   torch.Size([1, 1, heightmap.size(0), heightmap.size(1)]),
                                   align_corners=False)
        rotated_heightmap = F.grid_sample(heightmap.unsqueeze(0).unsqueeze(0), rotated_heightmap, align_corners=False)
        rotated_heightmap = rotated_heightmap.squeeze()
        x[idx][0] = rotated_heightmap
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(heightmap, cmap='gray')
        # plt.title('Original Heightmap')
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(rotated_heightmap, cmap='gray')
        # plt.title('Rotated Heightmap')
        # plt.axis('off')

        # plt.show()
    return x

def translation(x, severity):
    s = torch.tensor([0.02, 0.04, 0.06, 0.08, 0.10])[severity-1]
    num_positions = (x.shape[-1] * s).int()
    assert len(x.shape) == 4
    assert x.shape[1] == 1
    for idx in range(x.shape[0]):
        heightmap = x[idx][0].clone()
        direction = random.choice(['Left', 'Right', 'Up', 'Down'])
        if direction == 'Left':
            translated_heightmap = torch.cat((heightmap[:, num_positions:], torch.zeros(heightmap.size(0), num_positions)), dim=1)
        if direction == 'Right':
            translated_heightmap = torch.cat((torch.zeros(heightmap.size(0), num_positions), heightmap[:, :-num_positions]), dim=1)
        if direction == 'Up':
            translated_heightmap = torch.cat((heightmap[num_positions:], torch.zeros(num_positions, heightmap.size(1))), dim=0)
        if direction == 'Down':
            translated_heightmap = torch.cat((torch.zeros(num_positions, heightmap.size(1)), heightmap[:-num_positions]), dim=0)
        x[idx][0] = translated_heightmap
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(heightmap, cmap='gray')
        # plt.title('Original Heightmap')
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(translated_heightmap, cmap='gray')
        # plt.title('translated Heightmap')
        # plt.axis('off')

        # plt.show()
    return x

