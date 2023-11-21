from bulletarm import env_factory

def runDemo():
  env_config = {'render': True}
  env = env_factory.createEnvs(0, 'object_grasping', env_config)
  obs = env.reset()
  done = False
  while not done:
    action = env.getNextAction()
    action[1] *= 1.05
    action[2] *= 1.05
    obs, reward, done = env.step(action)

  env.close()

if __name__ == '__main__':
  runDemo()
