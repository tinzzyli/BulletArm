from bulletarm.planners import constants

def planner_fn(env, planner_type, planner_config):
    return constants.PLANNERS[planner_type](env, planner_config)

def getPlannerFn(env_type, planner_config):
  '''

  '''
  if 'planner_type' in planner_config:
    planner_type = planner_config['planner_type']
  elif env_type in  constants.PLANNERS:
    planner_type = env_type
  else:
    planner_type = 'none'

  if planner_type in constants.PLANNERS:
    # return lambda env: constants.PLANNERS[planner_type](env, planner_config)
    # def planner_fn(env):
    #     return constants.PLANNERS[planner_type](env, planner_config)
    return planner_fn
  else:
    raise ValueError('Invalid planner passed to factory.')
