from bulletarm import env_factory
import time
def runDemo():
  env_config = {'render': True, 'num_objects': 1,'object_index': 37}
  env = env_factory.createEnvs(0, 'object_grasping', env_config)
  obs = env.reset()
  done = False
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)
    print(action)

  env.close()

if __name__ == '__main__':
  runDemo()
