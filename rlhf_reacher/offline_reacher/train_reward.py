import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
from reward_model import RewardModel
from datetime import datetime

INPUT_CLIPS_PATH = "collected_clips_with_rewards.pkl"
NUM_PREFERENCE_PAIRS = 750_000  # Number of preference pairs to generate
MIN_REWARD_DIFFERENCE_FOR_PREFERENCE = 0 # Minimum reward gap between two clips to form a preference
RM_OBS_DIM = None
RM_ACT_DIM = None
RM_BATCH_SIZE = 64
RM_EPOCHS = 100
RM_LEARNING_RATE = 1e-3 # Learning rate for reward model optimizer
RM_VALIDATION_SPLIT = 0.1 # Fraction of data to use for validation
RM_EARLY_STOPPING_PATIENCE = 25 # Epochs to wait for improvement before early stopping
REWARD_MODEL_SAVE_PATH = "trained_reward_model" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pth"


class PreferenceDataset(Dataset):
    """
    Stores preference-labeled clip pairs as tensors, converting each clip only once on addition.
    """
    def __init__(self, device='cpu'):
        self.s1 = []
        self.a1 = []
        self.s2 = []
        self.a2 = []
        self.prefs = []
        self.device = device

    def add(self, clip1, clip2, pref):
        # Expects clip1/clip2 to be dicts {'obs': numpy_array, 'acts': numpy_array}
        s1 = torch.tensor(np.stack(clip1['obs']), dtype=torch.float32, device=self.device)
        a1 = torch.tensor(np.stack(clip1['acts']), dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack(clip2['obs']), dtype=torch.float32, device=self.device)
        a2 = torch.tensor(np.stack(clip2['acts']), dtype=torch.float32, device=self.device)
        p  = torch.tensor(pref, dtype=torch.float32, device=self.device) # Preference label (0 or 1)
        self.s1.append(s1)
        self.a1.append(a1)
        self.s2.append(s2)
        self.a2.append(a2)
        self.prefs.append(p)

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx):
        return self.s1[idx], self.a1[idx], self.s2[idx], self.a2[idx], self.prefs[idx]

# # --- RewardModel Class ---
# class RewardModel(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden=None): # hidden parameter is not used in this specific architecture
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim + act_dim, 128), nn.ReLU(),
#             nn.Linear(128, 64), nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, states, actions):
#         x = torch.cat([states, actions], dim=-1)
#         return self.net(x).squeeze(-1)

#     def predict_reward(self, obs, action):
#         s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#         a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
#         with torch.no_grad():
#             return self.forward(s, a).item()

def generate_preference_pairs_from_loaded_clips(
    loaded_clips_data, 
    num_pairs_to_generate=NUM_PREFERENCE_PAIRS,
    min_reward_gap=MIN_REWARD_DIFFERENCE_FOR_PREFERENCE
):
    """
    Generates preference pairs from a list of clips, each with a pre-calculated scalar reward.
    """
    preference_data_for_dataset = []
    if len(loaded_clips_data) < 2:
        print("Warning: Not enough unique clips to form pairs.")
        return preference_data_for_dataset

    generated_count = 0
    attempts = 0
    max_attempts = num_pairs_to_generate * 5 # Try more times to get desired number of pairs

    pbar = tqdm(total=num_pairs_to_generate, desc="Generating preference pairs")
    while generated_count < num_pairs_to_generate and attempts < max_attempts:
        attempts += 1
        clip_data1, clip_data2 = random.sample(loaded_clips_data, 2)
        reward1 = clip_data1.get('reward')
        reward2 = clip_data2.get('reward')
        if abs(reward1 - reward2) < min_reward_gap:
            continue # Skip if rewards are too close
        clip1_for_ds = {'obs': clip_data1['observations'], 'acts': clip_data1['actions']}
        clip2_for_ds = {'obs': clip_data2['observations'], 'acts': clip_data2['actions']}
        preference_label = 1.0 if reward1 > reward2 else 0.0
        preference_data_for_dataset.append((clip1_for_ds, clip2_for_ds, preference_label))
        generated_count += 1
        pbar.update(1)
    pbar.close()
    return preference_data_for_dataset

def train_reward_model_batched(
    rm_model: RewardModel,
    pref_dataset: PreferenceDataset,
    batch_size=RM_BATCH_SIZE,
    epochs=RM_EPOCHS,
    lr=RM_LEARNING_RATE,
    val_frac=RM_VALIDATION_SPLIT,
    patience=RM_EARLY_STOPPING_PATIENCE,
    device='cpu'
):
    """
    Trains RewardModel using mini-batches from a tensor-based PreferenceDataset.
    Assumes each clip in the dataset has shape [T, obs_dim] for states and [T, act_dim] for actions.
    """
    rm_model.to(device)

    total_samples = len(pref_dataset)
    val_size = int(total_samples * val_frac)
    train_size = total_samples - val_size
    
    if train_size == 0 or val_size == 0:
        print("Error: Dataset too small for train/validation split. Need more preference pairs.")
        return rm_model # Return untrained model

    train_ds, val_ds = random_split(pref_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(rm_model.parameters(), lr=lr)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print(f"\nStarting reward model training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        rm_model.train()
        current_train_losses = []
        current_train_accuracies = []
        for s1_batch, a1_batch, s2_batch, a2_batch, prefs_batch in train_loader:
            N, T, _ = s1_batch.shape 
            s1_flat = s1_batch.reshape(N * T, -1)
            a1_flat = a1_batch.reshape(N * T, -1)
            s2_flat = s2_batch.reshape(N * T, -1)
            a2_flat = a2_batch.reshape(N * T, -1)
            r1_per_segment = rm_model(s1_flat, a1_flat).view(N, T).sum(dim=1)
            r2_per_segment = rm_model(s2_flat, a2_flat).view(N, T).sum(dim=1) 
            logits = r1_per_segment - r2_per_segment
            loss_bce = F.binary_cross_entropy_with_logits(logits, prefs_batch)
            loss_reg = 1e-3 * (r1_per_segment.pow(2).mean() + r2_per_segment.pow(2).mean())
            total_loss = loss_bce + loss_reg
            
            predicted_prefs = (logits > 0).float()
            accuracy = (predicted_prefs == prefs_batch).float().mean()
            
            current_train_losses.append(total_loss.item())
            current_train_accuracies.append(accuracy.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(current_train_losses) if current_train_losses else 0
        avg_train_acc = np.mean(current_train_accuracies) if current_train_accuracies else 0
        
        # --- Validation Phase ---
        rm_model.eval()
        current_val_losses = []
        current_val_accuracies = []
        with torch.no_grad():
            for s1_batch, a1_batch, s2_batch, a2_batch, prefs_batch in val_loader:
                N, T, _ = s1_batch.shape
                s1_flat = s1_batch.reshape(N * T, -1)
                a1_flat = a1_batch.reshape(N * T, -1)
                s2_flat = s2_batch.reshape(N * T, -1)
                a2_flat = a2_batch.reshape(N * T, -1)

                r1_per_segment = rm_model(s1_flat, a1_flat).view(N, T).sum(dim=1)
                r2_per_segment = rm_model(s2_flat, a2_flat).view(N, T).sum(dim=1)
                
                logits = r1_per_segment - r2_per_segment
                loss_bce = F.binary_cross_entropy_with_logits(logits, prefs_batch)
                loss_reg = 1e-3 * (r1_per_segment.pow(2).mean() + r2_per_segment.pow(2).mean())
                val_loss = loss_bce + loss_reg
                
                predicted_prefs = (logits > 0).float()
                accuracy = (predicted_prefs == prefs_batch).float().mean()

                current_val_losses.append(val_loss.item())
                current_val_accuracies.append(accuracy.item())

        avg_val_loss = np.mean(current_val_losses) if current_val_losses else 0
        avg_val_acc = np.mean(current_val_accuracies) if current_val_accuracies else 0
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
                break
    
    rm_model.to('cpu')
    return rm_model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading collected clips from: {INPUT_CLIPS_PATH}")
    if not os.path.exists(INPUT_CLIPS_PATH):
        print(f"Error: Clips file not found at '{INPUT_CLIPS_PATH}'.")
        print("Please ensure clips have been generated and saved to this location.")
        exit(1)
    
    with open(INPUT_CLIPS_PATH, 'rb') as f:
        all_loaded_clips = pickle.load(f)

    if not all_loaded_clips or not isinstance(all_loaded_clips, list):
        print("Error: No clips data found in the file or data is not in expected list format.")
        exit(1)
    
    valid_clips_for_preferences = []
    for i, clip_data in enumerate(all_loaded_clips):
        if isinstance(clip_data, dict) and \
           'observations' in clip_data and isinstance(clip_data['observations'], np.ndarray) and \
           'actions' in clip_data and isinstance(clip_data['actions'], np.ndarray) and \
           'reward' in clip_data and clip_data['reward'] is not None:
            if clip_data['observations'].ndim == 2 and clip_data['actions'].ndim == 2 and \
               clip_data['observations'].shape[0] == clip_data['actions'].shape[0] and \
               clip_data['observations'].shape[0] > 0:
                valid_clips_for_preferences.append(clip_data)
            else:
                print(f"Warning: Clip at index {i} has malformed 'observations' or 'actions' (ndim, shape, or length mismatch). Skipping.")
        else:
            print(f"Warning: Clip at index {i} is missing required keys ('observations', 'actions', 'reward') or 'reward' is None. Skipping.")


    if not valid_clips_for_preferences:
        print("Error: No valid clips available after filtering. Cannot proceed to generate preferences.")
        exit(1)
        
    print(f"Loaded {len(valid_clips_for_preferences)} valid clip segments for preference generation.")

    preference_tuples = generate_preference_pairs_from_loaded_clips(
        valid_clips_for_preferences,
        num_pairs_to_generate=NUM_PREFERENCE_PAIRS,
        min_reward_gap=MIN_REWARD_DIFFERENCE_FOR_PREFERENCE
    )

    if not preference_tuples:
        print("Error: No preference pairs were generated. Cannot train the reward model.")
        exit(1)

    preference_dataset = PreferenceDataset(device=device)
    print("Populating preference dataset...")
    for clip1_dict, clip2_dict, preference_val in tqdm(preference_tuples, desc="Adding to PreferenceDataset"):
        preference_dataset.add(clip1_dict, clip2_dict, preference_val)
    
    print(f"Preference dataset populated with {len(preference_dataset)} pairs.")
    if len(preference_dataset) == 0:
        print("Error: Preference dataset is empty after processing. Cannot train.")
        exit(1)

    try:
        RM_OBS_DIM = preference_dataset.s1[0].shape[1]
        RM_ACT_DIM = preference_dataset.a1[0].shape[1]
        segment_length = preference_dataset.s1[0].shape[0]
        print(f"Inferred from data: Observation Dim = {RM_OBS_DIM}, Action Dim = {RM_ACT_DIM}, Segment Length = {segment_length}")
    except IndexError:
        print("Error: Could not infer dimensions from PreferenceDataset. It might be empty or malformed.")
        exit(1)

    reward_model_instance = RewardModel(obs_dim=RM_OBS_DIM, act_dim=RM_ACT_DIM)

    trained_reward_model = train_reward_model_batched(
        reward_model_instance,
        preference_dataset,
        batch_size=RM_BATCH_SIZE,
        epochs=RM_EPOCHS,
        lr=RM_LEARNING_RATE,
        val_frac=RM_VALIDATION_SPLIT,
        patience=RM_EARLY_STOPPING_PATIENCE,
        device=device
    )

    print(f"\nSaving trained reward model to: {REWARD_MODEL_SAVE_PATH}")
    try:
        torch.save(trained_reward_model.state_dict(), REWARD_MODEL_SAVE_PATH)
        print("Reward model saved successfully.")
    except Exception as e:
        print(f"Error saving reward model: {e}")

    print("Reward model training process finished.")