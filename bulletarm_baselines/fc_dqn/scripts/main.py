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
from transforms3d import quaternions

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


ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss

def train_step(agent, replay_buffer, logger):
    batch = replay_buffer.sample(batch_size)
    loss, td_error = agent.update(batch)
    logger.logTrainingStep(loss)
    if logger.num_training_steps % target_update_freq == 0:
        agent.updateTarget()

def saveModelAndInfo(logger, agent):
    logger.writeLog()
    logger.exportData()
    agent.saveModel(os.path.join(logger.models_dir, 'snapshot'))


def evaluate(envs, agent, logger):
  states, in_hands, obs = envs.reset()
  evaled = 0
  temp_reward = [[] for _ in range(num_eval_processes)]
  if not no_bar:
    eval_bar = tqdm(total=num_eval_episodes)
  while evaled < num_eval_episodes:
    q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, 0)
    actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
    states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
    rewards = rewards.numpy()
    dones = dones.numpy()
    states = copy.copy(states_)
    in_hands = copy.copy(in_hands_)
    obs = copy.copy(obs_)
    for i, r in enumerate(rewards.reshape(-1)):
      temp_reward[i].append(r)
    evaled += int(np.sum(dones))
    for i, d in enumerate(dones.astype(bool)):
      if d:
        R = 0
        for r in reversed(temp_reward[i]):
          R = r + gamma * R
        logger.logEvalEpisode(temp_reward[i], discounted_return=R)
        # eval_rewards.append(R)
        temp_reward[i] = []
    if not no_bar:
      eval_bar.update(evaled - eval_bar.n)
  # logger.eval_rewards.append(np.mean(eval_rewards[:num_eval_episodes]))
  logger.logEvalInterval()
  logger.writeLog()
  if not no_bar:
    eval_bar.close()

def train():
    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    eval_envs = EnvWrapper(num_eval_processes, env, env_config, planner_config)

    # setup agent
    agent = createAgent()
    eval_agent = createAgent(test=True)

    if load_model_pre:
        agent.loadModel(load_model_pre)
    agent.train()
    eval_agent.train()

    # logging
    base_dir = os.path.join(log_pre, '{}_{}_{}'.format(alg, model, env))
    if note:
        base_dir += '_'
        base_dir += note
    if not log_sub:
        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d.%H:%M:%S')
        log_dir = os.path.join(base_dir, timestamp)
    else:
        log_dir = os.path.join(base_dir, log_sub)

    # logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)

    hyper_parameters['model_shape'] = agent.getModelStr()
    # logger = Logger(log_dir, checkpoint_interval=save_freq, hyperparameters=hyper_parameters)
    logger = BaselineLogger(log_dir, checkpoint_interval=save_freq, num_eval_eps=num_eval_episodes, hyperparameters=hyper_parameters, eval_freq=eval_freq)
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    else:
        replay_buffer = QLearningBuffer(buffer_size)
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)

    states, in_hands, obs = envs.reset()

    if load_sub:
        logger.loadCheckPoint(os.path.join(base_dir, load_sub, 'checkpoint'), agent.loadFromState, replay_buffer.loadFromState)

    if planner_episode > 0 and not load_sub:
        if fill_buffer_deconstruct:
            fillDeconstructUsingRunner(agent, replay_buffer)
        else:
            planner_envs = envs
            planner_num_process = num_processes
            j = 0
            states, in_hands, obs = planner_envs.reset()
            s = 0
            if not no_bar:
                planner_bar = tqdm(total=planner_episode)
            local_transitions = [[] for _ in range(planner_num_process)]
            while j < planner_episode:
                plan_actions = planner_envs.getNextAction()
                # plan_actions = plan_actions.unsqueeze(dim=0)
                planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
                planner_actions_star = torch.cat((planner_actions_star, states.unsqueeze(1)), dim=1)
                states_, in_hands_, obs_, rewards, dones = planner_envs.step(planner_actions_star, auto_reset=True)
                buffer_obs = getCurrentObs(in_hands, obs)
                buffer_obs_ = getCurrentObs(in_hands_, obs_)
                for i in range(planner_num_process):
                  transition = ExpertTransition(states[i], buffer_obs[i], planner_actions_star_idx[i], rewards[i], states_[i],
                                                buffer_obs_[i], dones[i], torch.tensor(100), torch.tensor(1))
                  local_transitions[i].append(transition)
                states = copy.copy(states_)
                obs = copy.copy(obs_)
                in_hands = copy.copy(in_hands_)

                for i in range(planner_num_process):
                  if dones[i] and rewards[i]:
                    for t in local_transitions[i]:
                      replay_buffer.add(t)
                    local_transitions[i] = []
                    j += 1
                    s += 1
                    if not no_bar:
                      planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
                      planner_bar.update(1)
                  elif dones[i]:
                    local_transitions[i] = []

        if expert_aug_n > 0:
            augmentBuffer(replay_buffer, expert_aug_n, agent.rzs)
        elif expert_aug_d4:
            augmentBufferD4(replay_buffer, agent.rzs)

    # pre train
    if pre_train_step > 0:
        pbar = tqdm(total=pre_train_step)
        while logger.num_training_steps < pre_train_step:
            t0 = time.time()
            train_step(agent, replay_buffer, logger)
            if not no_bar:
                pbar.set_description('loss: {:.3f}, time: {:.2f}'.format(float(logger.getCurrentLoss()), time.time()-t0))
                pbar.update(len(logger.num_training_steps)-pbar.n)

            if (time.time() - start_time) / 3600 > time_limit:
                logger.saveCheckPoint(agent.getSaveState(), replay_buffer.getSaveState())
                exit(0)
        pbar.close()
        agent.saveModel(os.path.join(logger.models_dir, 'snapshot_{}'.format('pretrain')))
        # agent.sl = sl

    if not no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    while logger.num_training_steps < max_train_step:
        if fixed_eps:
            eps = final_eps
        else:
            eps = exploration.value(logger.num_eps)
        is_expert = 0
        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, eps)

        buffer_obs = getCurrentObs(in_hands, obs)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        envs.stepAsync(actions_star, auto_reset=False)

        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger)

        states_, in_hands_, obs_, rewards, dones = envs.stepWait()

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                in_hands_[idx] = reset_in_hands_[j]
                obs_[idx] = reset_obs_[j]

        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        for i in range(num_processes):
            replay_buffer.add(
                ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                 buffer_obs_[i], dones[i], torch.tensor(100), torch.tensor(is_expert))
            )
        logger.logStep(rewards.numpy(), dones.numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Action Step:{}; Episode: {}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
              logger.num_steps, logger.num_eps, logger.getAvg(logger.training_eps_rewards, 100),
              np.mean(logger.eval_eps_rewards[-2]) if len(logger.eval_eps_rewards) > 1 and len(logger.eval_eps_rewards[-2]) > 0 else 0, eps, float(logger.getCurrentLoss()),
              timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_training_steps - pbar.n)

        if logger.num_training_steps > 0 and eval_freq > 0 and logger.num_training_steps % eval_freq == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent, logger))
            eval_thread.start()

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    if eval_thread is not None:
        eval_thread.join()

    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(agent.getSaveState(), replay_buffer.getSaveState())
    envs.close()
    eval_envs.close()

def render(obj_list):
    
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




def untargeted_pgd_attack(epsilon=0.002, z_epsilon=None, alpha=5e-13, iters=10):
    torch.set_printoptions(precision=10)
    envs = EnvWrapper(num_processes, env, env_config, planner_config)

    #NOTE: find out the difference between test=True and test=False
    #NOTE: find out which to use, eval() or train(), in this case
    agent = createAgent(test=False) # if test=True, gradient equals to ZERO
    agent.eval()
    states, in_hands, obs, REDNER_OBJ_LIST, OBJ_XYZ_POSITION, OBJ_ROTATION = envs.resetWithGradient() 
    states = states.unsqueeze(dim = 0)
    in_hands = in_hands.unsqueeze(dim = 0)
    obs = obs.unsqueeze(dim = 0)

    q_value_maps_list = []
    position_list = []

    if not z_epsilon:
        z_epsilon = epsilon * 1e-04
    
    print("epsilon: ", epsilon)
    print("z_epsilon: ", z_epsilon)
    print("alpha: ", alpha)
    print("iters: ", iters)

    
    for iter in range(iters):
        print("iter number: ", iter+1)

        q_value_maps, _, _ = agent.getEGreedyActionsWithGradient(states, in_hands, obs, 0)
        loss = q_value_maps.sum()
        # maps -> encode -> decode -> action
        # loss.backward()
        grad = torch.autograd.grad(loss, adv_img, retain_graph=False, create_graph=False)[0]

        # with torch.no_grad():
        x_grad, y_grad, z_grad = OBJ_XYZ_POSITION.grad.clone()
        OBJ_XYZ_POSITION.grad.zero_()
        x,y,z = OBJ_XYZ_POSITION.clone().detach()
        # step length should be within a certain range
        x_eta = torch.clamp(x_grad.sign(), min = -epsilon,   max = epsilon)
        y_eta = torch.clamp(y_grad.sign(), min = -epsilon,   max = epsilon)
        z_eta = torch.clamp(z_grad.sign(), min = -z_epsilon, max = z_epsilon)
        # coordinate boudary of the object, please do not change these values
        # valid range of x and y is 0.2 while for z the range is 0.000025
        # accumulated change should not exceed the boundaries
        adv_position = torch.tensor([
            torch.clamp(x + x_eta, min =  0.4, max = 0.6),
            torch.clamp(y + y_eta, min = -0.1, max = 0.1),
            torch.clamp(z + z_eta, min =  0.013800, max = 0.013825)
        ])
        position_list.append(adv_position)
        q_value_maps_list.append(q_value_maps)

        for net in agent.networks:
            net.zero_grad()
        for net in agent.target_networks:
            net.zero_grad()
        for optimizer in agent.optimizers:
            optimizer.zero_grad()
        
        OBJ_XYZ_POSITION = adv_position.clone()
        OBJ_XYZ_POSITION.requires_grad = True
        x_adv, y_adv, z_adv = OBJ_XYZ_POSITION.clone()

        new_vertices = REDNER_OBJ_LIST[0].vertices.clone().detach()
        new_vertices[:,0:1] += (x_adv - x)
        new_vertices[:,1:2] += (y_adv - y)
        new_vertices[:,2:3] += (z_adv - z)
        REDNER_OBJ_LIST[0].vertices = new_vertices.clone()
        
        
        # obs = render(obj_list=REDNER_OBJ_LIST)
        # obs = obs.reshape(1,1,128,128)
        in_hands = in_hands.clone().detach()
        states = states.clone().detach()
        obs = obs.clone().detach()
        # obs.requires_grad = True
        print("gradient: ", [x_grad, y_grad, z_grad])
        print("OG position: ", [x,y,z])
        print("eta: ", [x_eta, y_eta, z_eta])
        print("ADV position: ", [x_adv, y_adv, z_adv])

    return q_value_maps_list, position_list


if __name__ == '__main__':
    map, pos = untargeted_pgd_attack(iters=25)
    np_pos = [p.numpy() for p in pos]
    np.save("/Users/tingxi/BulletArm/np_pos.txt", np.pos)
    print("end")
    # train()