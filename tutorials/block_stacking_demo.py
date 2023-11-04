from bulletarm import env_factory
import numpy as np
def runDemo():
  env_config = {'render': False}
  env = env_factory.createEnvs(1, 'house_building_3', env_config)

  obs = env.reset()
  done = False
  # _ = env.step(np.array([[0., 0.7, 0.0, 4.843947]]))
  
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)
    print(action)
  env.close()

if __name__ == '__main__':
  runDemo()
