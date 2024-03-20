from bulletarm import env_factory
import time
import numpy as np
import re

def dummy_bulletarm(position, point, rotation, object_index = None):
    env_config = {'render': False, 'num_objects': 1,'object_index': object_index}
    env = env_factory.createEnvs(0, 'object_grasping', env_config)
    _, _, _, _, params = env._resetAttack(position)
    done = False
    while not done:
        action = np.array([0.0, point[0], point[1], rotation])
        obs, reward, done = env.stepAttack(action)
        
    env.close()
    
    with open("./action_ablation_test.txt", "a") as file:
        file.write(str([object_index, position, point, rotation, reward]))
        
    return done
    
    
def getGridPosition(max_x, min_x, max_y, min_y, total_num_points = 10000):
    
    side_num_points = int(np.sqrt(total_num_points))
    
    mid_x = (max_x + min_x) * 0.50
    mid_y = (max_y + min_y) * 0.50

    side_length = 0.00010 * side_num_points if 0.00010 * side_num_points >= max(max_x-min_x, max_y-min_y) else max(max_x-min_x, max_y-min_y)
        
    x_range = [mid_x - (side_length * 0.50), mid_x + (side_length * 0.50)]
    y_range = [mid_y - (side_length * 0.50), mid_y + (side_length * 0.50)]
    x_values = np.linspace(x_range[0], x_range[1], side_num_points)
    y_values = np.linspace(y_range[0], y_range[1], side_num_points)
    
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    
    return points

def main(file_path):
    max_min_values = []
    unique_rotations = set()
    
    with open(file_path, 'r') as file:
        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')

        entry_count = 0

        for line in file:
            # [84, 0.4648, -0.0111, 0.4625, -0.0125,  2.3562,  0.0,  0,  0]
            # idx, pos_x,   pos_y,  act_x,   act_y,   act_rot,  p,  r1, r2
            numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)
            numeric = [float(num) if '.' in num else int(num) for num in numbers]

            max_x = max(max_x, numeric[3])
            min_x = min(min_x, numeric[3])
            max_y = max(max_y, numeric[4])
            min_y = min(min_y, numeric[4])

            unique_rotations.add(numeric[5])

            entry_count += 1

            if entry_count % 100 == 0:
                max_min_values.append((max_x, min_x, max_y, min_y))

                grid_points = getGridPosition(
                    max_x, 
                    min_x, 
                    max_y, 
                    min_y,
                    total_num_points = 10000
                )
                
                position = np.array([numeric[0], numeric[1]])
                object_index = int(numeric[0])
                for idx_1 in range(len(grid_points)):
                    point = grid_points[idx_1]
                    for idx_2 in range(len(unique_rotations)):
                        rotation = unique_rotations[idx_2]
                        dummy_bulletarm(position, point, rotation, object_index)
                
                
                max_x = float('-inf')
                min_x = float('inf')
                max_y = float('-inf')
                min_y = float('inf')
                
                unique_rotations = set()
                max_min_values = []
                
    return True

if __name__ == "__main__":
    file_path = "./action.txt"
    main(file_path)
    