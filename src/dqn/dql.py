import random
from collections import deque

import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self) -> None:
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 128), # Convert 4x4 grid to linear indices
            nn.ReLu(),
            nn.Linear(128, 128), 
            nn.ReLu(),
            nn.Linear(128, 4) # 4 actions (up, down, left, right)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool) -> None:
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple:
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self) -> int:
        return len(self.buffer)