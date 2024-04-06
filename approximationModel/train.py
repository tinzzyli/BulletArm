import torch
import re
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F

def load_data_from_txt(path_to_file):
    features = []
    labels = []
    with open(path_to_file, 'r') as file:
        for line in file:
            values = [float(match) for match in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)]
            features.append(torch.tensor(values[-5:-2]))
            labels.append(torch.tensor(values[-1:]))
            # print(all_numeric_values)
            # break
    return features, labels

def split_data():
    pass

def sample_data():
    pass

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
if __name__ == "__main__":
    features, labels = load_data_from_txt("object_original_position.txt")
    set_random_seed(42)
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    feature_train, feature_val, label_train, label_val = train_test_split(feature_train, label_train, test_size=0.1, random_state=42)
    
    print(feature_train[0], label_train[0])