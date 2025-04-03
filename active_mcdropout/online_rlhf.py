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
from visualizations import (
    visualize_reward_evolution,
    visualize_reward_model_loss, 
    visualize_reward_correlation,
    visualize_uncertainty_evolution,
    visualize_uncertainty_comparison,
    visualize_preference_accuracy,
    visualize_evaluation_rewards
)

class OnlineRLHF:

    def __init__(self, env_name='CartPole-v1', device='cpu', buffer_capacity=5000):
        self.env = gym.make(env_name)
        self.device = device
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy = PolicyNetwork(state_dim=self.state_dim).to(device)
        self.reward_model = RewardModel(state_dim=self.state_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.reward_loss = nn.BCELoss()
        
        # Set reward model to eval mode by default to disable dropout
        self.reward_model.eval()

        # Replay buffer for storing trajectories
        self.trajectory_buffer = []
        self.buffer_capacity = buffer_capacity

    @profile
    def compute_trajectory_reward(self, trajectory):
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
    def collect_trajectories(self, num_trajectories=100, max_steps=500):
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
    
    # Add new trajectories to the buffer and remove old ones if necessary
    @profile
    def add_to_buffer(self, trajectory):
        self.trajectory_buffer.extend(trajectory)
        if len(self.trajectory_buffer) > self.buffer_capacity:
            excess = len(self.trajectory_buffer) - self.buffer_capacity
            self.trajectory_buffer = self.trajectory_buffer[excess:]
        print(f"Buffer size: {len(self.trajectory_buffer)} trajectories")

    # Sample trajectories from the buffer
    @profile
    def sample_from_buffer(self, num_samples=100, recent_ratio=0.3):
        if len(self.trajectory_buffer) == 0:
            return []
        num_samples = min(num_samples, len(self.trajectory_buffer))
        num_recent = int(num_samples * recent_ratio)
        num_random = num_samples - num_recent
        recent_samples = self.trajectory_buffer[-num_recent:] if num_recent > 0 else []

        if num_random > 0 and len(self.trajectory_buffer) > num_recent:
            older_trajectories = self.trajectory_buffer[:-num_recent] if num_recent > 0 else self.trajectory_buffer
            random_indices = random.sample(range(len(older_trajectories)), 
                                        min(num_random, len(older_trajectories)))
            random_samples = [older_trajectories[i] for i in random_indices]
        else:
            random_samples = []
        
        return recent_samples + random_samples
    
    # Create preference pairs by selecting the most uncertain pairs based on MC dropout
    @profile
    def create_preference_pairs(self, trajectories, num_pairs=20, candidate_multiplier=2,
                                w_uncertainty=0.7, w_reward_diff=0.3, w_diversity=0.2):
        """
        Create preference pairs by selecting the most informative pairs based on a composite score
        that includes uncertainty, reward difference, and state space diversity.
        
        Args:
            trajectories: List of (trajectory, total_reward) tuples.
            num_pairs: Number of pairs to return.
            candidate_multiplier: Multiplier for candidate pairs.
            w_uncertainty: Weight for uncertainty component.
            w_reward_diff: Weight for reward difference component.
            w_diversity: Weight for diversity component.
        
        Returns:
            List of (traj1, traj2, preferred) tuples.
        """
        candidate_pairs = []
        num_candidates = len(trajectories) * candidate_multiplier

        for _ in range(num_candidates):
            idx1, idx2 = random.sample(range(len(trajectories)), 2)
            traj1, reward1 = trajectories[idx1]
            traj2, reward2 = trajectories[idx2]

            # Ground truth preference (for training the reward model)
            if reward1 > reward2:
                preferred = 1
            elif reward2 > reward1:
                preferred = 0
            else:
                preferred = random.randint(0, 1)

            # Uncertainty measure (using your existing MC dropout approach)
            uncertainty = self.estimate_uncertainty(traj1, traj2)

            # Reward signal difference
            reward_diff = abs(reward1 - reward2)

            # Diversity: compute the mean state for each trajectory and use Euclidean distance.
            # states1 = np.array([step[0] for step in traj1])
            # states2 = np.array([step[0] for step in traj2])
            # mean_state1 = states1.mean(axis=0)
            # mean_state2 = states2.mean(axis=0)
            # diversity = np.linalg.norm(mean_state1 - mean_state2)

            # Composite score: weighted sum of the components
            # composite_score = (w_uncertainty * uncertainty +
            #                 w_reward_diff * reward_diff)
            composite_score = uncertainty

            candidate_pairs.append((traj1, traj2, preferred, composite_score))

        # Sort candidate pairs by composite score in descending order
        candidate_pairs.sort(key=lambda x: x[3], reverse=True)

        # Select the top num_pairs
        selected_pairs = [(traj1, traj2, preferred) for traj1, traj2, preferred, _ in candidate_pairs[:num_pairs]]

        print(f"Selected {num_pairs} pairs based on composite score from {num_candidates} candidates.")
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
        visualize_reward_evolution(iterations, rewards)
        
    def visualize_reward_model_loss(self, iterations, losses):
        """
        Create visualization showing the reward model loss during training
        """
        visualize_reward_model_loss(iterations, losses)

    def visualize_reward_correlation(self, true_rewards, predicted_rewards):
        """
        Create visualization showing correlation between true rewards and predicted rewards
        
        Args:
            true_rewards: List of true rewards from the environment
            predicted_rewards: List of predicted rewards from the reward model
        """
        visualize_reward_correlation(true_rewards, predicted_rewards)

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
        
        # Save the current mode of the model
        was_training = self.reward_model.training
        self.reward_model.enable_dropout()
        
        with torch.no_grad():
            states1_repeated = states1_tensor.repeat(n_samples, 1, 1)
            actions1_repeated = actions1_tensor.repeat(n_samples, 1)
            states2_repeated = states2_tensor.repeat(n_samples, 1, 1)
            actions2_repeated = actions2_tensor.repeat(n_samples, 1)
            
            r1_samples = self.reward_model(states1_repeated.view(-1, states1_tensor.size(-1)), 
                                        actions1_repeated.view(-1)).view(n_samples, -1).sum(dim=1)
            r2_samples = self.reward_model(states2_repeated.view(-1, states2_tensor.size(-1)), 
                                        actions2_repeated.view(-1)).view(n_samples, -1).sum(dim=1)
            # Calculate differences and probabilities
            diff_samples = r1_samples - r2_samples
        
        # probs = torch.sigmoid(diff_samples)
        uncertainty = diff_samples.abs().var().item()
        
        if not was_training:
            self.reward_model.disable_dropout()
        
        return uncertainty


    @profile
    def train(self, iterations=20, trajectories_per_iter=100, preference_pairs=50, 
              reward_epochs=50, policy_rollouts=20, use_uncertainty=True, buffer_sample_ratio=0.5,
              recent_trajectory_ratio=0.3, use_buffer=True, return_metrics=False,
              uncertainty_warmup_iterations=15):
        iteration_numbers = []
        eval_rewards = []
        reward_model_losses = []
        accuracy_history = []
        uncertainty_values = []

        true_rewards_data = []
        predicted_rewards_data = []

        uncertainty_comparison_data = []
        
        for iter in range(iterations):
            print(f"\nIteration {iter+1}/{iterations}")
            
            # Collect trajectories
            new_trajectories = self.collect_trajectories(num_trajectories=trajectories_per_iter)
            avg_reward = np.mean([r for _, r in new_trajectories])
            print(f"Collected {len(new_trajectories)} trajectories. Average reward: {avg_reward:.2f}")


            if use_buffer:
                self.add_to_buffer(new_trajectories)
                buffer_samples_count = int(trajectories_per_iter * buffer_sample_ratio)
                buffer_trajectories = self.sample_from_buffer(buffer_samples_count, recent_ratio=recent_trajectory_ratio) if len(self.trajectory_buffer) > 0 else []
                new_samples_needed = trajectories_per_iter - len(buffer_trajectories)
                new_samples = new_trajectories[:new_samples_needed] if new_samples_needed > 0 else []
                combined_trajectories = buffer_trajectories + new_samples
                # combined_trajectories = self.trajectory_buffer.copy()
                # print(f"Using {len(combined_trajectories)} trajectories: {len(buffer_trajectories)} from buffer, {len(new_samples)} new")
            else:
                # Skip buffer usage - just use new trajectories directly
                combined_trajectories = new_trajectories
                print(f"Using  {len(combined_trajectories)} new trajectories (buffer disabled)")

            # Check if we're still in the warm-up phase
            in_warmup = iter < uncertainty_warmup_iterations
            
            # Use uncertainty-based sampling only after the warm-up period
            if use_uncertainty and not in_warmup:
                print(f"Using uncertainty-based sampling (iteration {iter+1})")
                pairs = self.create_preference_pairs(combined_trajectories, num_pairs=preference_pairs)
                # Capture average uncertainty for plotting
                avg_uncertainty = np.mean([self.estimate_uncertainty(traj1, traj2) for traj1, traj2, _ in pairs[:10]])  # Sample a subset to save time
                uncertainty_values.append(avg_uncertainty)
            else:
                # Use random sampling during warm-up or if uncertainty is disabled
                pairs = []
                for _ in range(preference_pairs):
                    idx1, idx2 = random.sample(range(len(combined_trajectories)), 2)
                    traj1, reward1 = combined_trajectories[idx1]
                    traj2, reward2 = combined_trajectories[idx2]
                    if reward1 > reward2:
                        preferred = 1
                    elif reward2 > reward1:
                        preferred = 0
                    else: 
                        preferred = random.randint(0, 1)
                    pairs.append((traj1, traj2, preferred))
                
                # Add dummy or 0 uncertainty value for consistency
                uncertainty_values.append(0.0)
                
                if in_warmup and use_uncertainty:
                    print(f"Warm-up phase: Using random sampling (iteration {iter+1}/{uncertainty_warmup_iterations})")
                else:
                    print(f"Using random sampling (uncertainty disabled)")

            print(f"Created {len(pairs)} preference pairs")

            reward_loss = self.train_reward_model(pairs, epochs=reward_epochs)
            reward_model_losses.append(reward_loss)
            
            # Calculate preference prediction accuracy
            if iter % 2 == 0:  # Calculate every other iteration to save time
                accuracy = self.evaluate_preference_accuracy(num_test_pairs=30)
                accuracy_history.append(accuracy)
            else:
                # Add previous accuracy or 0 to maintain consistent length
                accuracy_history.append(accuracy_history[-1] if accuracy_history else 0)

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
            visualize_uncertainty_evolution(iterations_compared, random_avgs, active_avgs)
        
        # Create reward evolution visualization
        self.visualize_reward_evolution(iteration_numbers, eval_rewards)
        
        # Create reward model loss visualization
        self.visualize_reward_model_loss(iteration_numbers, reward_model_losses)

        # Create reward correlation visualization
        self.visualize_reward_correlation(true_rewards_data, predicted_rewards_data)
        
        # Plot preference prediction accuracy
        visualize_preference_accuracy(accuracy_history)
        
        if return_metrics:
            training_metrics = {
                'epoch_rewards': eval_rewards,
                'reward_model_losses': reward_model_losses,
                'uncertainty_values': uncertainty_values if use_uncertainty else [],
                'preference_accuracies': accuracy_history,
                'true_rewards': true_rewards_data,
                'predicted_rewards': predicted_rewards_data
            }
            return self.policy, self.reward_model, training_metrics
        else:
            return self.policy, self.reward_model
    
    def evaluate_policy(self, num_episodes=10, max_steps=500, render=False):
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
    rlhf = OnlineRLHF(env_name='CartPole-v1', device=device, buffer_capacity=5000)
    policy, reward_model, _ = rlhf.train(
        iterations=40,
        trajectories_per_iter=100,
        preference_pairs=300,
        reward_epochs=5,
        policy_rollouts=100,
        buffer_sample_ratio=0.8,
        recent_trajectory_ratio=0.2,
        use_uncertainty=True,
        use_buffer=True,
        return_metrics=True
    )

    avg_reward, rewards_list = rlhf.evaluate_policy(num_episodes=20, max_steps=500, render=False)
    print(f"Final evaluation: Average reward = {avg_reward:.2f}")
    # Plot the rewards from the evaluation
    visualize_evaluation_rewards(rewards_list)

    
    # Save the trained models
    # torch.save(policy.state_dict(), 'policy_rlhf.pt')
    # torch.save(reward_model.state_dict(), 'reward_model.pt')
    print("Training complete. Models saved.")
