import torch
import re
import numpy as np
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, tensor_paths, device):
        self.tensor_paths = tensor_paths
        self.device = device
    
    def __len__(self):
        return len(self.tensor_paths)
    
    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        q_value_map, q2_output, action, reward = load_tensor(tensor_path, self.device)
        return q_value_map, q2_output, action, reward
    
class CustomModel(nn.Module):
    def __init__(self, input1_channels, input2_size, input3_size, hidden_size, output_size):
        super(CustomModel, self).__init__()

        # Input1的处理
        self.conv1 = nn.Conv2d(input1_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.maxpooling = nn.MaxPool2d(kernel_size=4)
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, hidden_size)

        # Input2的处理
        self.fc3 = nn.Linear(input2_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        # Input3的处理
        self.fc5 = nn.Linear(input3_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)

        # 输出层
        self.output_layer = nn.Linear(hidden_size * 3, output_size)

    def forward(self, input1, input2, input3):
        # Input1的处理
        x1 = torch.relu(self.conv1(input1))
        x1 = torch.relu(self.conv2(x1))
        x1 = torch.relu(self.conv3(x1))
        x1 = self.maxpooling(x1)
        x1 = torch.flatten(x1, 1)
        x1 = torch.relu(self.fc1(x1))
        x1 = torch.relu(self.fc2(x1))
        print(x1.shape)

        # Input2的处理
        x2 = torch.relu(self.fc3(input2))
        x2 = torch.relu(self.fc4(x2))
        print(x2.shape)

        # Input3的处理
        x3 = torch.relu(self.fc5(input3))
        x3 = torch.relu(self.fc6(x3))

        # 将三个输入连接起来
        x = torch.cat((x1, x2, x3), dim=1)

        # 输出层
        print(x.shape)
        output = self.output_layer(x)

        return output

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
def generate_path_list():
    path_list = [f"dataset/{i}_{j}.pt" for i in range(86) for j in range(100)]
    return path_list

def load_tensor(tensor_path, device):
    tensor = torch.load(tensor_path, map_location=device).detach()
    q_value_map = tensor[:128*128].reshape(1,128,128)
    q2_output = tensor[128*128:-4]
    action = tensor[-4:-1]
    reward = tensor[-1:]
    return q_value_map, q2_output, action, reward


if __name__ == "__main__":
    set_random_seed(42)
    if torch.cuda.is_available():
        device_name = torch.device('cuda')
    else:
        raise ValueError("cannot run on cpu device")
    
    data_path_list = generate_path_list()
    data_num = len(data_path_list)
    random.shuffle(data_path_list)

    input1_channels = 1
    input2_size = 16
    input3_size = 3
    hidden_size = 64
    output_size = 1
    num_epochs = 100
    batch_size = 1
    log_train_loss = []
    log_eval_loss = []
    log_test_loss = []
    best_eval_loss = float('inf')

    MyModel = CustomModel(input1_channels, input2_size, input3_size, hidden_size, output_size)
    MyModel.cuda()
    print(MyModel)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(MyModel.parameters(), lr=0.001)
    
    train_size = int(0.8 * len(data_path_list))
    eval_size = int(0.1 * len(data_path_list))
    test_size = len(data_path_list) - train_size - eval_size

    train_dataset = CustomDataset(data_path_list[:train_size], device_name)
    eval_dataset = CustomDataset(data_path_list[train_size:train_size+eval_size], device_name)
    test_dataset = CustomDataset(data_path_list[train_size+eval_size:], device_name)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for e in range(num_epochs):
        
        MyModel.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {e+1}/{num_epochs}, Training')
        for idx, data in enumerate(train_loader):
            q_value_map, q2_output, action, reward = data
            output = MyModel(q_value_map, q2_output, action)
            loss = criterion(output, reward)
            
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        log_train_loss.append(avg_train_loss)
        
        MyModel.eval()
        eval_loss = 0
        eval_bar = tqdm(eval_loader, desc=f'Epoch {e+1}/{num_epochs}, Evaluating')
        with torch.no_grad():
            for idx, data in enumerate(eval_loader):
                q_value_map, q2_output, action, reward = data
                output = MyModel(q_value_map, q2_output, action)
                loss = criterion(output, reward)
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(eval_loader.dataset)
        log_eval_loss.append(avg_eval_loss)
        
        print(f'Epoch {e+1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}, Average Eval Loss: {avg_eval_loss:.4f}')
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(MyModel.state_dict(), 'best_model_checkpoint.pth')
        
    MyModel.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            q_value_map, q2_output, action, reward = data
            output = MyModel(q_value_map, q2_output, action)
            loss = criterion(output, reward)
            log_test_loss.append(loss.detach().item())
            eval_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'Average Test Loss: {avg_test_loss:.4f}')
    
    plt.plot(log_train_loss, label='Train Loss')
    plt.plot(log_eval_loss, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Evaluation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    
    plt.clf()
    
    plt.plot(log_test_loss, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Testing Loss')
    plt.savefig('test_loss_figure.png')
    
    plt.clf()
    
    print("=== Task Done === ")