import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim = 64, num_actions = 2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
if __name__ == "__main__":
    # test the policy network
    policy = PolicyNetwork(4)
    state = torch.rand(4)
    action = policy(state)
    print(action)
    print(action.sum())