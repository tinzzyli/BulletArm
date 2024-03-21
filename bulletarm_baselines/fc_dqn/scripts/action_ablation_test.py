from bulletarm import env_factory
import time
import numpy as np
import re
import pyredner
import tqdm
import multiprocessing

# def dummy_bulletarm(position, point, rotation, object_index = None):
#     env_config = {'render': False, 'num_objects': 1,'object_index': object_index}
#     env = env_factory.createEnvs(0, 'object_grasping', env_config)
    
#     _, _, _, _, params = env._resetAttack(position)
#     done = False
#     while not done:
#         action = np.array([0.0, point[0], point[1], rotation])
#         obs, reward, done = env.stepAttack(action)
        
#     env.close()
    
#     with open("./action_ablation_test.txt", "a") as file:
#         file.write(str([object_index, position, point, rotation, reward])+"\n")
        
#     return done

def chunk_file(file_path):
    with open(file_path, 'r') as file:

        entry_count = 0

        chunks = []
        chunk = []
        for line in file:
            entry_count += 1
            # [84, 0.4648, -0.0111, 0.4625, -0.0125,  2.3562,  0.0,  0,  0]
            # idx, pos_x,   pos_y,  act_x,   act_y,   act_rot,  p,  r1, r2
            numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)
            numeric = [float(num) if '.' in num else int(num) for num in numbers]
            chunk.append(numeric)
            if entry_count % 100 == 0:
                chunks.append(chunk)
                chunk = []
            
        return chunks
                
def getData(chunk):
    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    min_y = float('inf')
    unique_rotations = set()
    
    for entry in chunk:
        max_x = max(max_x, entry[3])
        min_x = min(min_x, entry[3])
        max_y = max(max_y, entry[4])
        min_y = min(min_y, entry[4])
        unique_rotations.add(entry[5])
        
    grid_points = getGridPosition(
        max_x, 
        min_x, 
        max_y, 
        min_y,
        total_num_points = 10000)
    position = np.array([float(entry[1]), float(entry[2])])
    object_index = int(entry[0])
    
    return position, grid_points, unique_rotations, object_index
        
def dummy_bulletarm(position=None, 
                    grid_points=None, 
                    unique_rotations=None, 
                    object_index=None,
                    worker_id=None):
    file_path = f"./output/action_ablation_test_worker_{worker_id}.txt"
    env_config = {'render': False, 'num_objects': 1,'object_index': object_index}
    env = env_factory.createEnvs(0, 'object_grasping', env_config)
    
    for idx_1 in range(len(grid_points)):
        point = grid_points[idx_1]
        for idx_2 in range(len(list(unique_rotations))):
            rotation = list(unique_rotations)[idx_2]
            
            _, _, _, _, params = env._resetAttack(position)
            done = False
            while not done:
                action = np.array([0.0, point[0], point[1], rotation])
                obs, reward, done = env.stepAttack(action)
                with open(file_path, "a") as file:
                    file.write(str([object_index, position, point, rotation, reward])+"\n")
    env.close()
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

            (unique_rotations).add(numeric[5])

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
                
                # position = np.array([numeric[1], numeric[2]])
                # object_index = int(numeric[0])
                # for idx_1 in range(len(grid_points)):
                #     point = grid_points[idx_1]
                #     for idx_2 in range(len(list(unique_rotations))):
                #         rotation = list(unique_rotations)[idx_2]
                #         dummy_bulletarm(position, point, rotation, object_index)
                
                position = np.array([numeric[1], numeric[2]])
                object_index = int(numeric[0])
                dummy_bulletarm(position, grid_points, unique_rotations, object_index)
                
                max_x = float('-inf')
                min_x = float('inf')
                max_y = float('-inf')
                min_y = float('inf')
                
                unique_rotations = set()
                max_min_values = []
                print(entry_count)
                
    return True

def worker(param, worker_id):
    position, grid_points, unique_rotations, object_index = param
    try:
        dummy_bulletarm(position=position, 
                        grid_points=grid_points, 
                        unique_rotations=unique_rotations, 
                        object_index=object_index,
                        worker_id=object_index)
    except Exception as e:
        print("exception: \n", e)
    print("worker id job done: "+str(worker_id))
    
if __name__ == "__main__":
    pyredner.set_print_timing(False)
    file_path = "./action.txt"
    
    chunks = chunk_file(file_path)
    
    data = [
        getData(chunk) for chunk in chunks
    ]
    
    num_workers = len(data)
    
    pool = multiprocessing.Pool(processes=num_workers)
    
    for i in range(num_workers):
        pool.apply_async(worker, args=(data[i], i))
    
    pool.close()
    pool.join()