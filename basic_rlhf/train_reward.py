import numpy as np
import pickle
import torch 
import torch.nn as nn
import torch.optim as optim
from reward_model import RewardModel

def load_preference_data():
    with open('preference_pairs.pkl', 'rb') as f:
        return pickle.load(f)

def compute_trajectory_reward(model, trajectory, device):
    states = [step[0] for step in trajectory]
    states_array = np.array(states)
    states_tensor = torch.tensor(states_array, dtype=torch.float32, device=device)
    rewards = model(states_tensor)
    return rewards.sum()

def train_reward_model(epochs=100, learning_rate=1e-3, device='cpu'):
    pairs = load_preference_data()
    model = RewardModel(state_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for traj1, traj2, label in pairs:
            optimizer.zero_grad()
            r1 = compute_trajectory_reward(model, traj1, device)
            r2 = compute_trajectory_reward(model, traj2, device)
            diff = r1 - r2
            prob = torch.sigmoid(diff)
            target = torch.tensor(float(label), dtype=torch.float32, device=device)
            loss = loss_fn(prob.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss / len(pairs):.4f}")
    
    torch.save(model.state_dict(), 'reward_model.pth')
    print("Reward model saved as 'reward_model.pth'.")

if __name__ == "__main__":
    train_reward_model()