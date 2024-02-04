from bulletarm import env_factory
import time
import numpy as np
def runDemo():
  env_config = {'render': True, 'num_objects': 1,'object_index': 20}
  env = env_factory.createEnvs(0, 'object_grasping', env_config)
  
  position = np.array([0.5258, 0.0218])
  _, _, _, _, params = env._resetAttack(position) 
  
  a, _rot_mat_list, _scale_list = params
  # print(a)
  
  # _ = env.stepAttack(np.array([0., 0.6, 0.1, 5.49776703]))
  # _, _, _, _, _params = env._resetAttack(np.array([0.5434, -0.0411])) 
  # b, _rot_mat_list, _scale_list = _params
  # print(b)
  
  done = False
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)
    print(action, position)
    print("distance: ", np.sqrt((action[1]-position[0])**2 + (action[2]-position[1])**2))
    

  env.close()

if __name__ == '__main__':
  runDemo()