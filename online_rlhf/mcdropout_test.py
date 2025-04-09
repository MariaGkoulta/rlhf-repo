import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from policy_network import PolicyNetwork
from reward_model import RewardModel
import random
import matplotlib.pyplot as plt
from save_experiments import save_experiment_results, save_evaluation_results
import copy

CONFIG = {
    'env_name': 'CartPole-v1',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_iterations': 50,
    'trajectories_per_iter': 1000,
    'trajectories_to_collect': 20,
    'preference_pairs': 2,
    'num_candidate_pairs': 50,
    'reward_epochs': 3,
    'policy_rollouts': 100,
    'use_uncertainty': True,
    'max_steps': 500,
    'dropout_rate': 0.9,
    'hidden_dim': 16,
    'reward_model_lr': 1e-3,
    'policy_lr': 1e-4,
    'reward_loss_fn': nn.BCELoss(),
    'gamma': 0.99,
    'warmup_iterations': 0,
    'history_prefere'
    'history_pairs_multiplier': 3,
}

class OnlineRLHF:

    def __init__(self, env_name=CONFIG['env_name'], device=CONFIG['device']):
        self.env = gym.make(env_name)
        self.device = device
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy = PolicyNetwork(state_dim=self.state_dim).to(device)
        self.reward_model = RewardModel(state_dim=self.state_dim, dropout_rate=CONFIG['dropout_rate'], hidden_dim=CONFIG['hidden_dim']).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=CONFIG['policy_lr'])
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=CONFIG['reward_model_lr'])
        self.reward_loss = CONFIG['reward_loss_fn']
        self.trajectories_history = []
        self.preferences_history = []

    def compute_trajectory_reward(self, trajectory):
        states = [step[0] for step in trajectory]
        actions = [step[1] for step in trajectory]

        states_array = np.array(states)
        actions_array = np.array(actions)
    
        states_tensor = torch.tensor(states_array, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions_array, dtype=torch.long, device=self.device)

        rewards = self.reward_model(states_tensor, actions_tensor)
        return rewards.sum()
    
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action, probs

    def collect_trajectories(self, num_trajectories=100, max_steps=CONFIG['max_steps']):
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
    
    def create_preference_pairs(self, trajectories, num_pairs=20):
        pairs = []
        for _ in range(num_pairs):
            idx1, idx2 = random.sample(range(len(trajectories)), 2)
            traj1, reward1 = trajectories[idx1]
            traj2, reward2 = trajectories[idx2]
            if reward1 > reward2:
                preferred = 1
            elif reward2 > reward1:
                preferred = 0
            else: 
                preferred = random.choice([0, 1])
            pairs.append((traj1, traj2, preferred))
            self.trajectories_history.append((traj1, reward1))
            self.trajectories_history.append((traj2, reward2))
        return pairs
    
    def calculate_discounted_rewards(self, rewards, gamma):
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards.tolist()):
            R = reward + gamma * R
            discounted_rewards.insert(0, R)
            
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        return discounted_rewards
    
    def calculate_model_uncertainty(self, trajectories, n_samples=10):
        was_training = self.reward_model.training
        self.reward_model.enable_dropout()
        variances = []
        total_uncertainty = 0
        n_trajectories = 0
        trajectories_uncertainty_list = []

        with torch.no_grad():
            for traj1, reward in trajectories:
                states1 = [step[0] for step in traj1]
                actions1 = [step[1] for step in traj1]

                states1_tensor = torch.tensor(np.array(states1), dtype=torch.float32, device=self.device)
                actions1_tensor = torch.tensor(np.array(actions1), dtype=torch.long, device=self.device)

                r1_samples = []

                for _ in range(n_samples):
                    r1 = self.reward_model(states1_tensor, actions1_tensor).sum()
                    r1_samples.append(r1)

                # calculate the variance of the samples
                var1 = np.var(r1_samples)
                variances.append(var1)
                trajectories_uncertainty_list.append((traj1, reward, var1))

                total_uncertainty += var1
                n_trajectories += 1

        if not was_training:
            self.reward_model.disable_dropout()

        sorted_ids = np.argsort(variances)
        sorted_ids = sorted_ids[::-1]
        trajectories_uncertainty_list = [trajectories_uncertainty_list[i] for i in sorted_ids]

        avg_uncertainty = total_uncertainty / n_trajectories if n_trajectories > 0 else 0
    
        return avg_uncertainty, trajectories_uncertainty_list
    
    def train_temp_model(self, trajectory_pair, preferred, epochs=1):
        temp_model = copy.deepcopy(self.reward_model)
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=CONFIG['reward_model_lr'])

        traj1, traj2, _ = trajectory_pair

        states1 = torch.tensor([step[0] for step in traj1], dtype=torch.float32, device=self.device)
        actions1 = torch.tensor([step[1] for step in traj1], dtype=torch.long, device=self.device)
        states2 = torch.tensor([step[0] for step in traj2], dtype=torch.float32, device=self.device)
        actions2 = torch.tensor([step[1] for step in traj2], dtype=torch.long, device=self.device)

        for _ in range(epochs):
            temp_optimizer.zero_grad()
            r1 = temp_model(states1, actions1).sum()
            r2 = temp_model(states2, actions2).sum()
            diff = r1 - r2
            prob = torch.sigmoid(diff)
            target = torch.tensor(float(preferred), dtype=torch.float32, device=self.device)
            loss = self.reward_loss(prob.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            temp_optimizer.step()

        return temp_model
    
    def calculate_information_gain(self, traj_pair, trajectories, current_uncertainty, n_samples=10, train_epochs=5):
        original_model_state = copy.deepcopy(self.reward_model.state_dict())

        model_pref0 = self.train_temp_model(traj_pair, 0, epochs=train_epochs)
        self.reward_model.load_state_dict(model_pref0.state_dict())
        uncertainty_pref0, _ = self.calculate_model_uncertainty(trajectories, n_samples=n_samples)

        self.reward_model.load_state_dict(original_model_state)
        model_pref1 = self.train_temp_model(traj_pair, 1, epochs=train_epochs)
        self.reward_model.load_state_dict(model_pref1.state_dict())
        uncertainty_pref1, _ = self.calculate_model_uncertainty(trajectories, n_samples=n_samples)

        self.reward_model.load_state_dict(original_model_state)
        information_gain = current_uncertainty - (uncertainty_pref0 + uncertainty_pref1) / 2

        return information_gain
    
    def sort_pairs_by_information_gain(self, pairs, trajectories, current_uncertainty, n_samples=10, train_epochs=5):
        information_gains = []
        for traj_pair in pairs:
            info_gain = self.calculate_information_gain(traj_pair, trajectories, current_uncertainty, n_samples=n_samples, train_epochs=train_epochs)
            information_gains.append(info_gain)
        sorted_pairs = [x for _, x in sorted(zip(information_gains, pairs), reverse=True)]
        return sorted_pairs

    def train_reward_model(self, preference_pairs, epochs=5):
        epoch_losses = []
    
        for epoch in range(epochs):
            total_loss = 0

            for traj1, traj2, preferred in preference_pairs:
                self.reward_optimizer.zero_grad()
                r1 = self.compute_trajectory_reward(traj1)
                r2 = self.compute_trajectory_reward(traj2)

                diff = (r1 - r2)
                prob = torch.sigmoid(diff)
                target = torch.tensor(float(preferred), dtype=torch.float32, device=self.device)
                loss = self.reward_loss(prob.unsqueeze(0), target.unsqueeze(0))
                loss.backward()
                self.reward_optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(preference_pairs)
            epoch_losses.append(avg_loss)
            print(f"Reward model training epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return epoch_losses[-1]
    
    def update_policy(self, num_rollouts=10, gamma=CONFIG['gamma'], use_true_rewards=False):
        total_loss = 0
        
        for i in range(num_rollouts):
            state, _ = self.env.reset()
            done = False
            log_probs = []
            states = []
            actions = []
            rewards_list = []
            while not done:
                action, probs = self.select_action(state)
                log_prob = torch.log(probs[0, action])
                next_state, reward, done, truncated, _ = self.env.step(action)
                rewards_list.append(reward)

                log_probs.append(log_prob)
                states.append(state)
                actions.append(action)
                state = next_state

                if truncated or done:
                    break

            states_array = np.array(states)
            actions_array = np.array(actions)

            states_tensor = torch.tensor(states_array, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions_array, dtype=torch.long, device=self.device)

            if use_true_rewards:
                rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
            else: 
                with torch.no_grad():
                    rewards = self.reward_model(states_tensor, actions_tensor)

            discounted_rewards = self.calculate_discounted_rewards(rewards, gamma)
            policy_loss = sum(-log_prob * R for log_prob, R in zip(log_probs, discounted_rewards))
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            loss_value = policy_loss.item()
            total_loss += loss_value
                
            # if (i + 1) % 5 == 0:
            #     print(f"Policy update: Rollout {i+1}/{num_rollouts}, Loss: {loss_value:.4f}")
        
        return total_loss / num_rollouts
        
    def evaluate_preference_accuracy(self, num_test_pairs=20):
        """Evaluate how often the reward model correctly predicts preferences"""
        # Collect test trajectories
        test_trajectories = self.collect_trajectories(num_trajectories=200)
        
        # Create preference pairs without adding them to training
        correct_predictions = 0
        reward_diff = []
        for _ in range(num_test_pairs):
            idx1, idx2 = random.sample(range(len(test_trajectories)), 2)
            traj1, reward1 = test_trajectories[idx1]
            traj2, reward2 = test_trajectories[idx2]

            reward_diff.append(np.abs(reward1-reward2))
            
            # Ground truth preference (using environment rewards)
            true_preferred = 1 if reward1 > reward2 else 0 if reward1 < reward2 else random.choice([0, 1])
            
            # Model's prediction
            with torch.no_grad():
                r1 = self.compute_trajectory_reward(traj1)
                r2 = self.compute_trajectory_reward(traj2)
                pred_preferred = 1 if r1 > r2 else 0
            
            if pred_preferred == true_preferred:
                correct_predictions += 1
        
        avg_reward_diff = np.mean(reward_diff, axis=0)
        print(f'Average reward value: {np.mean([x[1] for x in test_trajectories]):.2f}')
        print(f'Average reward difference: {avg_reward_diff:.2f}')
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
        # plt.show()
        
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
        # plt.show()

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
        # plt.show()

    def train(self, iterations=20, trajectories_per_iter=200, trajectories_to_collect=10, preference_pairs=50, num_candidate_pairs=200,
              reward_epochs=3, policy_rollouts=20, use_uncertainty=True, warmup_iterations=10, history_pairs_multiplier=3):
        iteration_numbers = []
        eval_rewards = []
        reward_model_losses = []
        accuracy_history = []
        reward_mse_losses = []
        correlation_coeff_list = []
        reward_model_uncertainties = []
        random_variances = []
        uncertainty_variances = []

        true_rewards_data = []
        predicted_rewards_data = []
        
        for iter in range(iterations):
            print(f"\nIteration {iter+1}/{iterations}")
            # print whether model is in train mode
            print(f"Reward model training mode: {self.reward_model.training}")
            
            # Collect trajectories
            trajectories = self.collect_trajectories(num_trajectories=trajectories_per_iter)
            avg_reward = np.mean([r for _, r in trajectories])
            print(f"Collected {len(trajectories)} trajectories. Average true reward: {avg_reward:.2f}")

            # Calculate model uncertainty
            uncertainty, trajectories_with_uncertainty = self.calculate_model_uncertainty(trajectories, n_samples=20)
            print(f"Model uncertainty: {uncertainty:.4f}")
            reward_model_uncertainties.append(uncertainty)

            if use_uncertainty:
                trajectories = trajectories_with_uncertainty[:trajectories_to_collect]
            else:
                trajectories = random.sample(trajectories, trajectories_to_collect)

            
            random_trajectories = random.sample(trajectories_with_uncertainty, min(len(trajectories), trajectories_to_collect))
            unc_trajectories = trajectories_with_uncertainty[:trajectories_to_collect]
            random_trajectories_variance = [traj[2] for traj in random_trajectories]
            unc_trajectories_variance = [traj[2] for traj in unc_trajectories]
            print(f"Random trajectories variance: mean = {np.mean(random_trajectories_variance):.4f}, std = {np.std(random_trajectories_variance):.4f}")
            print(f"Uncertainty trajectories variance: mean = {np.mean(unc_trajectories_variance):.4f}, std = {np.std(unc_trajectories_variance):.4f}")
            random_variances.append(np.mean(random_trajectories_variance))
            uncertainty_variances.append(np.mean(unc_trajectories_variance))

            # remove third column from trajectories
            trajectories = [(traj[0], traj[1]) for traj in trajectories]

            # pairs = self.create_preference_pairs(trajectories, num_pairs=preference_pairs)
            # print(f"Created {len(pairs)}  preference pairs")
            candidate_pairs = self.create_preference_pairs(trajectories, num_pairs=num_candidate_pairs)
            print(f"Created {len(candidate_pairs)} candidate preference pairs")
            

            # Sort pairs by information gain
            if use_uncertainty:
                sorted_pairs = self.sort_pairs_by_information_gain(candidate_pairs, trajectories, uncertainty)
                pairs = sorted_pairs[:preference_pairs]
            else:
                pairs = random.sample(candidate_pairs, min(len(candidate_pairs), preference_pairs))
            print(f"Selected {len(pairs)} preference pairs for training")


            history_pairs = []
            if len(self.preferences_history) > 0:
                history_pairs = random.sample(self.preferences_history, min(len(self.preferences_history), history_pairs_multiplier * preference_pairs))

            reward_loss = self.train_reward_model(pairs + history_pairs, epochs=reward_epochs)
            reward_model_losses.append(reward_loss)
            self.preferences_history.extend(pairs)
            

            if iter % 4 == 0 or iter == iterations - 1:
                accuracy = self.evaluate_preference_accuracy(num_test_pairs=20)
                accuracy_history.append(accuracy)

                correlation_trajectories = self.collect_trajectories(num_trajectories=20)
                
                for trajectory, true_reward in correlation_trajectories:
                    # Calculate predicted reward
                    with torch.no_grad():
                        predicted_reward = self.compute_trajectory_reward(trajectory).item()
                    
                    true_rewards_data.append(true_reward)
                    predicted_rewards_data.append(predicted_reward)
                
                print(f"Collected {len(correlation_trajectories)} trajectories for reward correlation")


                episode_losses = []
                true_rewards_history = []
                predicted_rewards_history = []
                # for traj, true_reward in random.sample(self.trajectories_history, np.minimum(len(self.trajectories_history), 200)):
                for traj, true_reward in self.trajectories_history[-min(len(self.trajectories_history), 200):]:
                    with torch.no_grad():
                        predicted_reward = self.compute_trajectory_reward(traj).item()
                        mse_loss = np.mean((true_reward - predicted_reward) ** 2)
                        episode_losses.append(mse_loss)
                        true_rewards_history.append(true_reward)
                        predicted_rewards_history.append(predicted_reward)
                mse_loss_avg = np.mean(episode_losses)
                reward_mse_losses.append(mse_loss_avg.item())
                correlation = np.corrcoef(true_rewards_history, predicted_rewards_history)[0, 1]
                print(f"Average predicted reward: {np.mean(predicted_rewards_history):.2f}")
                print(f"Average true reward: {np.mean(true_rewards_history):.2f}")
                correlation_coeff_list.append(correlation)
                print(f"Correlation coefficient: {correlation:.3f}")
                print(f"Reward model MSE loss: {mse_loss_avg:.4f}")
            
            # Update policy
            if iter > warmup_iterations:
                self.update_policy(num_rollouts=policy_rollouts, use_true_rewards=False)
            else:
                self.update_policy(num_rollouts=policy_rollouts, use_true_rewards=True)

            # Evaluate current policy
            eval_trajectories = self.collect_trajectories(num_trajectories=10)
            eval_reward = np.mean([r for _, r in eval_trajectories])
            print(f"Evaluation: Average reward = {eval_reward:.2f}")
            
            # Store metrics
            iteration_numbers.append(iter + 1)
            eval_rewards.append(eval_reward)
        
        self.env.close()
        
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
        # plt.show()

        # plot reward MSE losses with log
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(reward_mse_losses)+1), reward_mse_losses, 'm-o', linewidth=2)
        plt.title('Reward Model MSE Losses')
        plt.xlabel('Evaluation')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reward_mse_losses.png')
        # plt.show()

        # plot correlation coefficients with log scale
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(correlation_coeff_list)+1), correlation_coeff_list, 'c-o', linewidth=2)
        plt.title('Correlation Coefficients')
        plt.xlabel('Evaluation')
        plt.ylabel('Correlation Coefficient')
        plt.ylim(np.min(correlation_coeff_list), np.max(correlation_coeff_list))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('correlation_coefficients.png')
        # plt.show()

        # plot model uncertainties
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(reward_model_uncertainties)+1), reward_model_uncertainties, 'y-o', linewidth=2)
        plt.title('Model Uncertainties')
        plt.xlabel('Evaluation')
        plt.ylabel('Uncertainty')
        plt.ylim(np.min(reward_model_uncertainties), np.max(reward_model_uncertainties))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('model_uncertainties.png')
        # plt.show()

        #plot random and uncertainty variances
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(random_variances)+1), random_variances, 'b-o', linewidth=2, label='Random Variance')
        plt.plot(range(1, len(uncertainty_variances)+1), uncertainty_variances, 'r-o', linewidth=2, label='Uncertainty Variance')
        plt.title('Random and Uncertainty Variances')
        plt.xlabel('Evaluation')
        plt.ylabel('Variance')
        plt.ylim(np.min(random_variances), np.max(uncertainty_variances))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('random_uncertainty_variances.png')
        # plt.show()

        metrics_data = {
            'iteration_numbers': iteration_numbers,
            'eval_rewards': eval_rewards,
            'reward_model_losses': reward_model_losses,
            'true_rewards_data': true_rewards_data,
            'predicted_rewards_data': predicted_rewards_data,
            'accuracy_history': accuracy_history,
            'reward_mse_losses': reward_mse_losses,
            'correlation_coeff_list': correlation_coeff_list,
            'reward_model_uncertainties': reward_model_uncertainties,
            'random_variances': random_variances,
            'uncertainty_variances': uncertainty_variances,
        }
        results_folder = save_experiment_results(CONFIG, metrics_data)

        return self.policy, self.reward_model, results_folder
    
    def evaluate_policy(self, num_episodes=10, max_steps=CONFIG['max_steps'], render=False):
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

    rlhf.policy.load_state_dict(torch.load('baseline_policy.pt'))
    rlhf.reward_model.load_state_dict(torch.load('baseline_reward_model.pt'))
    print("Pretrained baseline models loaded.")

    policy, reward_model, results_folder = rlhf.train(
        iterations=CONFIG['num_iterations'],
        trajectories_per_iter=CONFIG['trajectories_per_iter'],
        trajectories_to_collect=CONFIG['trajectories_to_collect'],
        preference_pairs=CONFIG['preference_pairs'],
        num_candidate_pairs=CONFIG['num_candidate_pairs'],
        reward_epochs=CONFIG['reward_epochs'],
        policy_rollouts=CONFIG['policy_rollouts'],
        use_uncertainty=CONFIG['use_uncertainty'],
        warmup_iterations=CONFIG['warmup_iterations'],
        history_pairs_multiplier=CONFIG['history_pairs_multiplier']
    )

    avg_reward, rewards_list = rlhf.evaluate_policy(num_episodes=20, max_steps=CONFIG['max_steps'], render=False)
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
    # plt.show()

    save_evaluation_results(results_folder, rewards_list, avg_reward)
    
    # Save the trained models
    # torch.save(policy.state_dict(), 'policy_rlhf.pt')
    # torch.save(reward_model.state_dict(), 'reward_model.pt')
    print("Training complete. Models saved.")
    