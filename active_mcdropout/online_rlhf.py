import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from policy_network import PolicyNetwork
from reward_model import RewardModel
import random
import matplotlib.pyplot as plt
from line_profiler import profile

class OnlineRLHF:

    def __init__(self, env_name='CartPole-v1', device='cpu'):
        self.env = gym.make(env_name)
        self.device = device
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy = PolicyNetwork(state_dim=self.state_dim).to(device)
        self.reward_model = RewardModel(state_dim=self.state_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.reward_loss = nn.BCELoss()

    @profile
    def compute_trajectory_reward(self, trajectory):
        # OPTIMIZATION: Convert directly to tensor without going through numpy
        states = torch.tensor([step[0] for step in trajectory], 
                            dtype=torch.float32, device=self.device)
        actions = torch.tensor([step[1] for step in trajectory], 
                            dtype=torch.long, device=self.device)
        
        rewards = self.reward_model(states, actions)
        return rewards.sum()
    
    @profile
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action, probs

    @profile
    def collect_trajectories(self, num_trajectories=100, max_steps=200):
        trajectories = []
        for i in range(num_trajectories):
            state, _ = self.env.reset()
            trajectory = []
            total_reward = 0

            for step in range(max_steps):
                action, _ = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                trajectory.append((state, action, reward, next_state, done))
                total_reward += reward
                state = next_state

                if done or truncated:
                    break

            trajectories.append((trajectory, total_reward))
        return trajectories
    
    #     Create preference pairs by selecting the most uncertain pairs based on MC dropout
    @profile
    def create_preference_pairs(self, trajectories, num_pairs=20, candidtate_multiplier=3):
        candidate_pairs = []
        num_candidates = num_pairs * candidtate_multiplier
        
        for _ in range(num_candidates):
            idx1, idx2 = random.sample(range(len(trajectories)), 2)
            traj1, reward1 = trajectories[idx1]
            traj2, reward2 = trajectories[idx2]

            if reward1 > reward2:
                preferred = 1
            elif reward2 > reward1:
                preferred = 0
            else: 
                preferred = random.randint(0, 1)

            
            uncertainty = self.estimate_uncertainty(traj1, traj2)
            candidate_pairs.append((traj1, traj2, preferred, uncertainty))
            
        # Sort by uncertainty and select the top num_pairs
        candidate_pairs.sort(key=lambda x: x[3], reverse=True)
        selected_pairs = [(traj1, traj2, preferred) for traj1, traj2, preferred, _ in candidate_pairs[:num_pairs]]
        
        print(f"Selected {num_pairs} most uncertain pairs out of {num_candidates} candidates")
        return selected_pairs
    
    def calculate_discounted_rewards(self, rewards, gamma):
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards.tolist()):
            R = reward + gamma * R
            discounted_rewards.insert(0, R)
            
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        return discounted_rewards

    @profile
    def train_reward_model(self, preference_pairs, epochs=5, batch_size=32):
        epoch_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            indices = np.random.permutation(len(preference_pairs))
            
            for i in range(0, len(preference_pairs), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = [preference_pairs[idx] for idx in batch_indices]
                
                self.reward_optimizer.zero_grad()
                batch_loss = 0
                
                # Process all pairs in batch before backward pass
                for traj1, traj2, preferred in batch:
                    r1 = self.compute_trajectory_reward(traj1)
                    r2 = self.compute_trajectory_reward(traj2)
                    diff = r1 - r2
                    prob = torch.sigmoid(diff)
                    target = torch.tensor(float(preferred), dtype=torch.float32, device=self.device)
                    loss = self.reward_loss(prob.unsqueeze(0), target.unsqueeze(0))
                    batch_loss += loss / len(batch)
                
                batch_loss.backward()
                self.reward_optimizer.step()
                total_loss += batch_loss.item() * len(batch)
            
            avg_loss = total_loss / len(preference_pairs)
            epoch_losses.append(avg_loss)
            print(f"Reward model training epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return epoch_losses[-1]
    
    @profile
    def update_policy(self, num_rollouts=10, gamma=0.99):
        total_loss = 0
        
        for i in range(num_rollouts):
            state, _ = self.env.reset()
            done = False
            log_probs = []
            states = []
            actions = []

            while not done:
                action, probs = self.select_action(state)
                log_prob = torch.log(probs[0, action])
                next_state, _, done, truncated, _ = self.env.step(action)
                
                log_probs.append(log_prob)
                states.append(state)
                actions.append(action)
                state = next_state

                if truncated or done:
                    break

            states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)

            with torch.no_grad():
                rewards = self.reward_model(states_tensor, actions_tensor)
            discounted_rewards = self.calculate_discounted_rewards(rewards, gamma)
            policy_loss = sum(-log_prob * R for log_prob, R in zip(log_probs, discounted_rewards))
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            loss_value = policy_loss.item()
            total_loss += loss_value
                
            if (i + 1) % 5 == 0:
                print(f"Policy update: Rollout {i+1}/{num_rollouts}, Loss: {loss_value:.4f}")
        
        return total_loss / num_rollouts
        
    def evaluate_preference_accuracy(self, num_test_pairs=50):
        """Evaluate how often the reward model correctly predicts preferences"""
        # Collect test trajectories
        test_trajectories = self.collect_trajectories(num_trajectories=20)
        
        # Create preference pairs without adding them to training
        correct_predictions = 0
        
        for _ in range(num_test_pairs):
            idx1, idx2 = random.sample(range(len(test_trajectories)), 2)
            traj1, reward1 = test_trajectories[idx1]
            traj2, reward2 = test_trajectories[idx2]
            
            # Ground truth preference (using environment rewards)
            true_preferred = 1 if reward1 > reward2 else 0
            
            # Model's prediction
            with torch.no_grad():
                r1 = self.compute_trajectory_reward(traj1)
                r2 = self.compute_trajectory_reward(traj2)
                pred_preferred = 1 if r1 > r2 else 0
            
            if pred_preferred == true_preferred:
                correct_predictions += 1
        
        accuracy = correct_predictions / num_test_pairs
        print(f"Preference prediction accuracy: {accuracy:.2f}")
        return accuracy

    def visualize_reward_evolution(self, iterations, rewards):
        """
        Create visualization showing evolution of rewards over training iterations
        """
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, rewards, 'b-o', linewidth=2)
        plt.title('Reward Evolution During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reward_evolution.png')
        plt.show()
        
    def visualize_reward_model_loss(self, iterations, losses):
        """
        Create visualization showing the reward model loss during training
        """
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, losses, 'r-o', linewidth=2)
        plt.title('Reward Model Loss During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')  # Log scale often helps visualize loss curves
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reward_model_loss.png')
        plt.show()

    def visualize_reward_correlation(self, true_rewards, predicted_rewards):
        """
        Create visualization showing correlation between true rewards and predicted rewards
        
        Args:
            true_rewards: List of true rewards from the environment
            predicted_rewards: List of predicted rewards from the reward model
        """
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(true_rewards, predicted_rewards, alpha=0.6)
        
        # Find the min and max across both axes to set equal limits
        min_val = min(min(true_rewards), min(predicted_rewards))
        max_val = max(max(true_rewards), max(predicted_rewards))
        
        # Add y=x line for reference (perfect correlation)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (perfect correlation)')
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(true_rewards, predicted_rewards)[0, 1]
        plt.title(f'Correlation Between True and Predicted Rewards (r = {correlation:.3f})')
        plt.xlabel('True Rewards (Environment)')
        plt.ylabel('Predicted Rewards (Reward Model)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reward_correlation.png')
        plt.show()

    @profile
    def estimate_uncertainty(self, traj1, traj2, n_samples=10):
        # Extract states and actions once before the loop
        states1 = [step[0] for step in traj1]
        actions1 = [step[1] for step in traj1]
        states2 = [step[0] for step in traj2]
        actions2 = [step[1] for step in traj2]
        
        # Convert to tensors once before the loop
        states1_tensor = torch.tensor(np.array(states1), dtype=torch.float32, device=self.device)
        actions1_tensor = torch.tensor(np.array(actions1), dtype=torch.long, device=self.device)
        states2_tensor = torch.tensor(np.array(states2), dtype=torch.float32, device=self.device)
        actions2_tensor = torch.tensor(np.array(actions2), dtype=torch.long, device=self.device)
        
        # OPTIMIZATION: Batch the MC dropout samples instead of doing them in a loop
        self.reward_model.enable_dropout()
        
        with torch.no_grad():
            # Stack the same inputs n_samples times
            states1_repeated = states1_tensor.repeat(n_samples, 1, 1)
            actions1_repeated = actions1_tensor.repeat(n_samples, 1)
            states2_repeated = states2_tensor.repeat(n_samples, 1, 1)
            actions2_repeated = actions2_tensor.repeat(n_samples, 1)
            
            # Calculate rewards for all samples at once
            r1_samples = self.reward_model(states1_repeated.view(-1, states1_tensor.size(-1)), 
                                        actions1_repeated.view(-1)).view(n_samples, -1).sum(dim=1)
            r2_samples = self.reward_model(states2_repeated.view(-1, states2_tensor.size(-1)), 
                                        actions2_repeated.view(-1)).view(n_samples, -1).sum(dim=1)
            
            # Calculate differences and probabilities
            diff_samples = r1_samples - r2_samples
        
        probs = torch.sigmoid(diff_samples)
        uncertainty = probs.var().item()
        self.reward_model.eval()
        return uncertainty


    @profile
    def train(self, iterations=20, trajectories_per_iter=200, preference_pairs=500, 
              reward_epochs=50, policy_rollouts=20, use_uncertainty=True):
        iteration_numbers = []
        eval_rewards = []
        reward_model_losses = []
        accuracy_history = []

        true_rewards_data = []
        predicted_rewards_data = []

        uncertainty_comparison_data = []
        
        for iter in range(iterations):
            print(f"\nIteration {iter+1}/{iterations}")
            
            # Collect trajectories
            trajectories = self.collect_trajectories(num_trajectories=trajectories_per_iter)
            avg_reward = np.mean([r for _, r in trajectories])
            print(f"Collected {len(trajectories)} trajectories. Average reward: {avg_reward:.2f}")

            # if iter == 0 or iter % 10 == 0 or iter == iterations - 1:
            #     random_pairs, uncertain_pairs = self.compare_sampling_strategies(
            #         trajectories, num_pairs=min(50, preference_pairs))
                
            #     # Store data for later analysis
            #     random_avg = np.mean([self.estimate_uncertainty(t1, t2) for t1, t2, _ in random_pairs])
            #     active_avg = np.mean([self.estimate_uncertainty(t1, t2) for t1, t2, _ in uncertain_pairs])
            #     uncertainty_comparison_data.append((iter+1, random_avg, active_avg))

            if use_uncertainty:
                pairs = self.create_preference_pairs(trajectories, num_pairs=preference_pairs)
            else:
                pairs = []
                for _ in range(preference_pairs):
                    idx1, idx2 = random.sample(range(len(trajectories)), 2)
                    traj1, reward1 = trajectories[idx1]
                    traj2, reward2 = trajectories[idx2]
                    if reward1 > reward2:
                        preferred = 1
                    elif reward2 > reward1:
                        preferred = 0
                    else: 
                        preferred = random.randint(0, 1)
                pairs.append((traj1, traj2, preferred))

            print(f"Created {len(pairs)} preference pairs")

            reward_loss = self.train_reward_model(pairs, epochs=reward_epochs)
            reward_model_losses.append(reward_loss)
            
            # Calculate preference prediction accuracy
            if iter % 2 == 0:  # Calculate every other iteration to save time
                accuracy = self.evaluate_preference_accuracy(num_test_pairs=30)
                accuracy_history.append(accuracy)

            if iter % 4 == 0 or iter == iterations - 1:
                correlation_trajectories = self.collect_trajectories(num_trajectories=10)
                
                for trajectory, true_reward in correlation_trajectories:
                    # Calculate predicted reward
                    with torch.no_grad():
                        predicted_reward = self.compute_trajectory_reward(trajectory).item()
                    
                    true_rewards_data.append(true_reward)
                    predicted_rewards_data.append(predicted_reward)
                
                print(f"Collected {len(correlation_trajectories)} trajectories for reward correlation")

            # Update policy
            self.update_policy(num_rollouts=policy_rollouts)

            # Evaluate current policy
            eval_trajectories = self.collect_trajectories(num_trajectories=5)
            eval_reward = np.mean([r for _, r in eval_trajectories])
            print(f"Evaluation: Average reward = {eval_reward:.2f}")
            
            # Store metrics
            iteration_numbers.append(iter + 1)
            eval_rewards.append(eval_reward)
        
        self.env.close()

            # After training completes, visualize uncertainty evolution
        if uncertainty_comparison_data:
            iterations_compared, random_avgs, active_avgs = zip(*uncertainty_comparison_data)
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations_compared, random_avgs, 'r-o', label='Random Sampling')
            plt.plot(iterations_compared, active_avgs, 'b-o', label='Uncertainty Sampling')
            plt.title('Evolution of Uncertainty During Training')
            plt.xlabel('Iteration')
            plt.ylabel('Average Uncertainty')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('uncertainty_evolution.png')
            plt.show()
        
        # Create reward evolution visualization
        self.visualize_reward_evolution(iteration_numbers, eval_rewards)
        
        # Create reward model loss visualization
        self.visualize_reward_model_loss(iteration_numbers, reward_model_losses)

        # Create reward correlation visualization
        self.visualize_reward_correlation(true_rewards_data, predicted_rewards_data)
        
        # Plot preference prediction accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(accuracy_history)+1), accuracy_history, 'g-o', linewidth=2)
        plt.title('Preference Prediction Accuracy')
        plt.xlabel('Evaluation')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('preference_accuracy.png')
        plt.show()
        
        return self.policy, self.reward_model
    
    def compare_sampling_strategies(self, trajectories, num_pairs=20):
        """Compare random sampling vs uncertainty-based sampling"""
        print("\nComparing sampling strategies...")
        
        # Method 1: Random sampling
        random_pairs = []
        for _ in range(num_pairs):
            idx1, idx2 = random.sample(range(len(trajectories)), 2)
            traj1, reward1 = trajectories[idx1]
            traj2, reward2 = trajectories[idx2]
            if reward1 > reward2:
                preferred = 1
            elif reward2 > reward1:
                preferred = 0
            else: 
                preferred = random.randint(0, 1)
            random_pairs.append((traj1, traj2, preferred))
        
        # Method 2: Uncertainty-based sampling
        uncertain_pairs = self.create_preference_pairs(trajectories, num_pairs=num_pairs)
        
        # Calculate uncertainty for both methods
        random_uncertainty = [self.estimate_uncertainty(traj1, traj2) for traj1, traj2, _ in random_pairs]
        active_uncertainty = [self.estimate_uncertainty(traj1, traj2) for traj1, traj2, _ in uncertain_pairs]
        
        # Print statistics
        print(f"Average uncertainty - Random: {np.mean(random_uncertainty):.4f}")
        print(f"Average uncertainty - Active: {np.mean(active_uncertainty):.4f}")
        print(f"Uncertainty improvement: {np.mean(active_uncertainty)/np.mean(random_uncertainty):.2f}x")
        
        # Visualize the uncertainty distributions
        plt.figure(figsize=(10, 6))
        plt.hist(random_uncertainty, alpha=0.5, label='Random Sampling')
        plt.hist(active_uncertainty, alpha=0.5, label='Uncertainty Sampling')
        plt.xlabel('Uncertainty')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Distribution: Random vs Active Sampling')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('uncertainty_comparison.png')
        plt.show()
        
        return random_pairs, uncertain_pairs
    
    def evaluate_policy(self, num_episodes=10, max_steps=200, render=False):
        """
        Evaluate the current policy by running it for a number of episodes.
        
        Args:
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment.
            
        Returns:
            avg_reward: Average reward across episodes.
            rewards_list: List of rewards per episode.
        """
        rewards_list = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            for _ in range(max_steps):
                if render:
                    self.env.render()
                action, _ = self.select_action(state)
                state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                if done or truncated:
                    break
            
            rewards_list.append(episode_reward)
            print(f"Episode {episode+1}: Reward = {episode_reward}")
        
        avg_reward = np.mean(rewards_list)
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward, rewards_list


if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize and run the RLHF algorithm
    rlhf = OnlineRLHF(env_name='CartPole-v1', device=device)
    policy, reward_model = rlhf.train(
        iterations=70,
        trajectories_per_iter=100,
        preference_pairs=300,
        reward_epochs=5,
        policy_rollouts=100
    )

    avg_reward, rewards_list = rlhf.evaluate_policy(num_episodes=20, max_steps=200, render=False)
    print(f"Final evaluation: Average reward = {avg_reward:.2f}")
    # Plot the rewards from the evaluation
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards_list)+1), rewards_list, 'b-o', linewidth=2)
    plt.title('Rewards from Evaluation Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('evaluation_rewards.png')
    plt.show()

    
    # Save the trained models
    torch.save(policy.state_dict(), 'policy_rlhf.pt')
    torch.save(reward_model.state_dict(), 'reward_model.pt')
    print("Training complete. Models saved.")
