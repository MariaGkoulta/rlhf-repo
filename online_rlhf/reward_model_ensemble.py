import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=16):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, hidden_dim)  # +1 for the action input
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, states, actions):
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
        x = torch.cat([states, actions.float()], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        rewards = torch.sigmoid(self.fc3(x)).squeeze()
        return rewards

    def get_reward(self, state, action):
        with torch.no_grad():
            return self.forward(state, action).item()
