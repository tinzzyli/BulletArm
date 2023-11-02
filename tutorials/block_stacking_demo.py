from bulletarm import env_factory

def runDemo():
  env_config = {'render': True, 'object_index': -1}
  env = env_factory.createEnvs(0, 'house_building_3', env_config)
  obs = env.reset()
  done = False
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)

  env.close()

if __name__ == '__main__':
  runDemo()
