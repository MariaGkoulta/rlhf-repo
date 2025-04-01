import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)

if __name__ == "__main__":
    model = RewardModel(state_dim=4)
    dummy_state = torch.randn(5, 4)
    pred = model(dummy_state)
    print("Predicted rewards:", pred)
