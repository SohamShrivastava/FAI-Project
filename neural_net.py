import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    def save_checkpoint(self, filename='dqn_model.pth'):
        folder = './checkpoints'
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, filename)
        torch.save(self.state_dict(), path)