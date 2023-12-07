from bulletarm import env_factory
import time
import numpy as np
def runDemo():
  env_config = {'render': True, 'num_objects': 1,'object_index': 37}
  env = env_factory.createEnvs(0, 'object_grasping', env_config)
  pos = np.array([0.5, -0.0])
  obs = env._resetAttack(pos)
  done = False
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)
    print(action)

  env.close()

if __name__ == '__main__':
  runDemo()
