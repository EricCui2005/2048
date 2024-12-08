import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Dataset class
class ImitationDataset(Dataset):
    def __init__(self, states, actions):
        
        """
        states: numpy array of shape (num_samples, 16) representing game board states.
        actions: numpy array of shape (num_samples,) representing the expert actions (0-3).
        """
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# Neural network class
class ImitationPolicyNet(nn.Module):
    
    def __init__(self):
        super(ImitationPolicyNet, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

# Example random data implementation
def load_data():
    
    data = pd.read_csv('')
    
    # """NEED TO UNRANDOMIZE"""
    # num_samples = 10000  # Example size
    # states = np.random.rand(num_samples, 16)  # Random states as an example
    # actions = np.random.randint(0, 4, size=num_samples)  # Random actions (0, 1, 2, 3)
    # return states, actions

# Training function
def train_model():
    
    # Loading train data data and creating dataset
    states, actions = load_data()
    dataset = ImitationDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Neural network
    model = ImitationPolicyNet()
    
    # Initializing log-likelihood optimizer and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 20
    for _ in range(num_epochs):
        epoch_loss = 0.0
        for states_batch, actions_batch in dataloader:
            
            # Zeroing gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(states_batch)
            
            # Compute loss
            loss = criterion(outputs, actions_batch)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
    
    return model

# Evaluate the model
def evaluate_model(model, states, actions):
    model.eval()
    with torch.no_grad():
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        outputs = model(states_tensor)
        predicted_actions = torch.argmax(outputs, dim=1)
        accuracy = (predicted_actions == actions_tensor).float().mean().item()
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Main script
if __name__ == "__main__":
    # Train the model
    trained_model = train_model()
    
    # Load validation data
    val_states, val_actions = load_data()  # Replace with real validation data
    
    # Evaluate the model
    evaluate_model(trained_model, val_states, val_actions)
