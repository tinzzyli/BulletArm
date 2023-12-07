  _, _, _, _, params = env._resetAttack(np.array([0.5444, -0.0401])) 
  a, _rot_mat_list, _scale_list = params
  print(a)
  
  # _ = env.resetAttack()
    
  _, _, _, _, _params = env._resetAttack(np.array([0.5434, -0.0411])) 
  b, _rot_mat_list, _scale_list = _params
  print(b)