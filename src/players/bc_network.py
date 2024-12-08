import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import ast
import tqdm

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

# return states, actions

# Example random data implementation
def load_train_data():
    data = pd.read_csv('train.csv')


    states_data = data['state']
    states_processed = [ast.literal_eval(state) for state in states_data]
    states = []

    for matrix in states_processed:
        state_list = []
        for row in matrix:
            state_list += row
        # Normalize state (log2 of tiles, 0 for empty)
        new = [x + 1 for x in state_list]
        final_list = np.log2(new) / 11.0
        states.append(final_list)

    # Processing actions
    actions_data = data['action']
    actions = []
    for a in actions_data:
        match a:
            case 'up':
                actions.append(0)
            case 'left':
                actions.append(1)
            case 'down':
                actions.append(2)
            case 'right':
                actions.append(3)

    states = np.array(states)
    actions = np.array(actions)
    
    print(states[0])
    # print(states[1])
    # print(states[2])
    return states, actions

# Training function
def train_model(device='cuda'):
    
    # Loading train data data and creating dataset
    states, actions = load_train_data()
    dataset = ImitationDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Neural network
    model = ImitationPolicyNet().to(device)
    
    # Initializing log-likelihood optimizer and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 500
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for states_batch, actions_batch in dataloader:
            
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)
            
            # Forward pass
            outputs = model(states_batch)
            
            # Compute loss
            loss = criterion(outputs, actions_batch)
            
            # Zeroing gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    return model

# Evaluate the model
def evaluate_model(model, states, actions, device='cuda'):
    model.eval()
    with torch.no_grad():
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        outputs = model(states_tensor)
        predicted_actions = torch.argmax(outputs, dim=1)
        accuracy = (predicted_actions == actions_tensor).float().mean().item()
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Main script
if __name__ == "__main__":

    # Train the model
    trained_model = train_model('cuda')
    
    torch.save(trained_model.state_dict(), 'bc_model_weights.pth')

    # Load validation data
    val_states, val_actions = load_train_data()  # Replace with real validation data
    
    # Evaluate the model
    evaluate_model(trained_model, val_states, val_actions, 'cuda')
