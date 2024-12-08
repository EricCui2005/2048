import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def grad_log_pi(theta, action, state):
    """
    Computes the gradient of the log likelihood for a given state-action pair
    
    Args:
        theta: Policy parameters
        action: Action taken
        state: State observation
        
    Returns:
        Gradient of log probability of the action given the state
    """
    # Compute action probabilities using softmax
    scores = np.dot(theta, state)
    probs = np.exp(scores) / np.sum(np.exp(scores))
    
    # Compute gradient
    grad = state * (action - probs)
    
    return grad

class BehavioralCloning:
    def __init__(self, alpha, k_max, grad_log_pi):
        self.alpha = alpha          # step size
        self.k_max = k_max         # number of iterations
        self.grad_log_pi = grad_log_pi  # log likelihood gradient

def optimize(M: BehavioralCloning, D, theta):
    alpha, k_max, grad_log_pi = M.alpha, M.k_max, M.grad_log_pi
    
    for k in range(1, k_max + 1):
        # Calculate mean gradient over the dataset
        grad = sum(grad_log_pi(theta, a, s) for s, a in D) / len(D)
        theta += alpha * grad
    
    return theta


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# class PolicyNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128):
#         super(PolicyNetwork, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim)
#         )
    
#     def forward(self, state):
#         return self.network(state)

# def grad_log_pi(policy_net, state, action):
#     """
#     Computes the gradient of the log likelihood for a given state-action pair
#     using the policy network
#     """
#     scores = policy_net(state)
#     probs = torch.softmax(scores, dim=-1)
    
#     # Convert action to one-hot
#     action_one_hot = torch.zeros_like(probs)
#     action_one_hot[action] = 1
    
#     # Compute gradient
#     grad = action_one_hot - probs
#     return grad

# class DeepBehavioralCloning:
#     def __init__(self, state_dim, action_dim, learning_rate=0.001, k_max=1000):
#         self.policy_net = PolicyNetwork(state_dim, action_dim)
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
#         self.k_max = k_max
    
#     def optimize(self, D):
#         """
#         D: list of (state, action) tuples
#         """
#         for k in range(1, self.k_max + 1):
#             self.optimizer.zero_grad()
            
#             # Calculate mean gradient over the dataset
#             total_loss = 0
#             for state, action in D:
#                 state = torch.FloatTensor(state)
#                 scores = self.policy_net(state)
#                 probs = torch.softmax(scores, dim=-1)
#                 loss = -torch.log(probs[action])  # Negative log likelihood
#                 total_loss += loss
            
#             # Backpropagate and update
#             mean_loss = total_loss / len(D)
#             mean_loss.backward()
#             self.optimizer.step()
        
#         return self.policy_net





# Neural Network for 2048
class Policy2048(nn.Module):
    def __init__(self):
        super(Policy2048, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 256),  # 4x4 board flattened = 16 inputs
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),    # 4 possible actions (up, down, left, right)
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

# Dataset for storing expert demonstrations
class GameDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# Collect expert demonstrations
def collect_demonstrations(expert_policy, env, num_episodes=1000):
    states, actions = [], []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Normalize state (log2 of tiles, 0 for empty)
            norm_state = np.log2(state.flatten() + 1) / 11.0  # max tile is 2048 (2^11)
            action = expert_policy(state)  # Get expert action
            states.append(norm_state)
            actions.append(action)
            state, _, done, _ = env.step(action)
    
    return np.array(states), np.array(actions)

# Training function
def train_behavioral_cloning(expert_policy, env, device='cuda'):
    # Initialize policy network
    policy_net = Policy2048().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Collect expert demonstrations
    states, actions = collect_demonstrations(expert_policy, env)
    dataset = GameDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            # Forward pass
            predictions = policy_net(batch_states)
            loss = criterion(predictions, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return policy_net

# Inference function
def get_action(policy_net, state, device='cuda'):
    with torch.no_grad():
        norm_state = torch.FloatTensor(np.log2(state.flatten() + 1) / 11.0).to(device)
        probs = policy_net(norm_state)
        return torch.argmax(probs).item()


# Initialize environment and expert policy
env = Game2048Env()  # 2048 environment
expert = ExpertPolicy()  # expert policy

# Train the behavioral cloning policy
policy_net = train_behavioral_cloning(expert, env)

# Use the trained policy
state = env.reset()
while not done:
    action = get_action(policy_net, state)
    state, reward, done, _ = env.step(action)
