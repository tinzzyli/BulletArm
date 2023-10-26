from bulletarm import env_factory

def runDemo():
  env_config = {'render': True, 'object_index': -1, 'num_objects':5}
  env = env_factory.createEnvs(0, 'object_grasping', env_config)
  obs = env.reset()
  done = False
  # while not done:
  #   action = env.getNextAction()
  #   obs, reward, done = env.step(action)

  for _ in range(5):
    action = env.getNextAction()
    obs, reward, done = env.step(action)
  env.close()

if __name__ == '__main__':
  runDemo()
