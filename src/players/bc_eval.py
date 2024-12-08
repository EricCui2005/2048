import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import ast
import tqdm


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

def load_train_data():
    data = pd.read_csv('validationSet.csv')


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

def evaluate_model(model, states, actions, device='cuda'):
    model.eval()
    with torch.no_grad():
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        outputs = model(states_tensor)
        predicted_actions = torch.argmax(outputs, dim=1)
        accuracy = (predicted_actions == actions_tensor).float().mean().item()
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":

    model = ImitationPolicyNet().to('cuda')  # Recreate the model structure
    model.load_state_dict(torch.load('bc_model_weights.pth', map_location=torch.device('cuda')))

    # Load validation data
    val_states, val_actions = load_train_data()  # Replace with real validation data
    
    # Evaluate the model
    evaluate_model(model, val_states, val_actions, 'cuda')
