import pickle
import random

def create_preference_pairs(num_pairs=200): 
    with open('trajectories.pkl', 'rb') as f:
        trajectories_dict = pickle.load(f)
    
    all_trajectories = []
    for level, traj_list in trajectories_dict.items():
        print(f"  {level}: {len(traj_list)} trajectories")
        for traj_data in traj_list:
            traj, reward = traj_data 
            all_trajectories.append((traj, reward))
    
    print(f"Total: {len(all_trajectories)} trajectories")
    
    pair_data = [] 
    for i in range(num_pairs):
        traj_data1, traj_data2 = random.sample(all_trajectories, 2)
        traj1, reward1 = traj_data1
        traj2, reward2 = traj_data2
        label = 1 if reward1 > reward2 else 0
        pair_data.append((traj1, traj2, label))
        
        if (i+1) % 10 == 0:
            print(f"Created {i+1}/{num_pairs} preference pairs...")

    with open('preference_pairs.pkl', 'wb') as f:
        pickle.dump(pair_data, f)
        print(f"Created and saved {len(pair_data)} preference pairs.")

if __name__ == "__main__":
    create_preference_pairs()