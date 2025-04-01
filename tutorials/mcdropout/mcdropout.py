import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Transform: convert PIL image to tensor, then normalize [0,1] range to mean=0.5, std=0.5
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download + load training data
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MCDropoutNet(nn.Module):
    def __init__(self, dropout_p=0.2):
        super(MCDropoutNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No ReLU on final layer
        return x

def enable_dropout(model):
    """Enable dropout during inference"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def train_mc_dropout(model, train_loader, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return train_losses

def mc_dropout_predict(model, x, T=20):
    """Perform T forward passes with dropout enabled"""
    model.eval()  # Set model to evaluation mode but dropout will still be active
    enable_dropout(model)  # Ensure dropout is enabled

    with torch.no_grad():
        predictions = torch.stack([model(x) for _ in range(T)])
    
    return predictions

def predict_with_uncertainty(model, x, T=20):
    """Get predictions with uncertainty estimates"""
    device = x.device
    predictions = mc_dropout_predict(model, x, T)
    
    # Compute mean and variance of softmax probabilities
    probs = F.softmax(predictions, dim=-1)
    mean_probs = probs.mean(dim=0)
    var_probs = probs.var(dim=0)
    
    # Compute predictive entropy (a measure of uncertainty)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
    
    return {
        'logits': predictions,
        'mean_probs': mean_probs,
        'var_probs': var_probs,
        'entropy': entropy
    }

def evaluate_mc_dropout(model, test_loader, T=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    correct = 0
    total = 0
    all_entropies = []
    all_variances = []
    all_correct = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Get predictions and uncertainty
        results = predict_with_uncertainty(model, images, T)
        
        # Get predicted class
        pred_labels = results['mean_probs'].argmax(dim=1)
        
        # Track accuracy
        is_correct = (pred_labels == labels)
        correct += is_correct.sum().item()
        total += labels.size(0)
        
        # Store entropy and max variance for each prediction
        all_entropies.extend(results['entropy'].cpu().numpy())
        all_variances.extend(results['var_probs'].max(dim=1)[0].cpu().numpy())
        all_correct.extend(is_correct.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    print(f"MC Dropout Test Accuracy: {accuracy:.2f}%")
    
    return {
        'accuracy': accuracy,
        'entropies': all_entropies,
        'variances': all_variances,
        'correct': all_correct
    }

def visualize_uncertainty(evaluation_results):
    """Visualize the relationship between uncertainty and accuracy"""
    entropies = np.array(evaluation_results['entropies'])
    variances = np.array(evaluation_results['variances'])
    correct = np.array(evaluation_results['correct'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Predictive entropy histogram for correct vs incorrect predictions
    ax1.hist(entropies[correct], bins=30, alpha=0.5, label='Correct predictions')
    ax1.hist(entropies[~correct], bins=30, alpha=0.5, label='Incorrect predictions')
    ax1.set_xlabel('Predictive Entropy')
    ax1.set_ylabel('Count')
    ax1.set_title('Predictive Entropy for Correct vs Incorrect Predictions')
    ax1.legend()
    
    # Plot 2: Max variance histogram for correct vs incorrect predictions
    ax2.hist(variances[correct], bins=30, alpha=0.5, label='Correct predictions')
    ax2.hist(variances[~correct], bins=30, alpha=0.5, label='Incorrect predictions')
    ax2.set_xlabel('Max Predictive Variance')
    ax2.set_ylabel('Count')
    ax2.set_title('Max Variance for Correct vs Incorrect Predictions')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('uncertainty_metrics.png')
    plt.show()
    
    # Plot accuracy vs uncertainty
    n_bins = 10
    entropy_bins = np.linspace(np.min(entropies), np.max(entropies), n_bins+1)
    
    bin_accuracies = []
    bin_counts = []
    bin_centers = []
    
    for i in range(n_bins):
        mask = (entropies >= entropy_bins[i]) & (entropies < entropy_bins[i+1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(correct[mask]))
            bin_counts.append(np.sum(mask))
            bin_centers.append((entropy_bins[i] + entropy_bins[i+1]) / 2)
    
    plt.figure(figsize=(10, 5))
    plt.bar(bin_centers, bin_accuracies, width=(entropy_bins[1]-entropy_bins[0])*0.8)
    
    # Overlay the sample counts
    for i, (center, acc, count) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
        plt.text(center, acc + 0.02, f'n={count}', ha='center')
    
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Predictive Entropy')
    plt.ylim(0, 1.1)
    plt.savefig('accuracy_vs_entropy.png')
    plt.show()

def identify_uncertain_samples(model, dataloader, T=20, n_samples=10):
    """Identify the most uncertain samples"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_images = []
    all_labels = []
    all_entropies = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        results = predict_with_uncertainty(model, images, T)
        
        all_images.append(images.cpu())
        all_labels.append(labels.cpu())
        all_entropies.append(results['entropy'].cpu())
        
        if len(all_entropies) * batch_size >= n_samples * 10:
            break
    
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_entropies = torch.cat(all_entropies)
    
    # Get indices of samples with highest entropy
    highest_entropy_indices = torch.argsort(all_entropies, descending=True)[:n_samples]
    
    # Display these samples
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(highest_entropy_indices):
        img = all_images[idx].squeeze().numpy()
        label = all_labels[idx].item()
        entropy = all_entropies[idx].item()
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}\nEntropy: {entropy:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('most_uncertain_samples.png')
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    model = MCDropoutNet(dropout_p=0.5)
    
    # Train the model
    print("Training the model...")
    train_losses = train_mc_dropout(model, train_loader, epochs=5)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    # Evaluate with MC Dropout
    print("\nEvaluating with MC Dropout...")
    results = evaluate_mc_dropout(model, test_loader, T=20)
    
    # Compare with standard evaluation (no MC Dropout)
    print("\nEvaluating without MC Dropout (standard)...")
    model.eval()  # This will disable dropout
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            device = next(model.parameters()).device
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    standard_accuracy = 100.0 * correct / total
    print(f"Standard Evaluation Accuracy: {standard_accuracy:.2f}%")
    
    # Visualize uncertainty metrics
    print("\nVisualizing uncertainty metrics...")
    visualize_uncertainty(results)
    
    # Identify most uncertain samples
    print("\nIdentifying most uncertain samples...")
    identify_uncertain_samples(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'mc_dropout_model.pt')
    print("\nModel saved as 'mc_dropout_model.pt'")