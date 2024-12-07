import random
from collections import deque
import torch
import torch.nn as nn
import numpy as np
from game.game_logic import Game

class DQN(nn.Module):
    """
    Deep Q-Network for learning 2048 game strategies. 

    Architecture consists of three fully connected layers with ReLu activations,
    transformtion a 4x4 game board state into Q-values for four possible actions.
    """

    def __init__(self) -> None:
        """Initialize the DQN architecture with three linear layers."""
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 128), # Convert 4x4 grid to linear indices
            nn.ReLu(),
            nn.Linear(128, 128), 
            nn.ReLu(),
            nn.Linear(128, 4) # 4 actions (up, down, left, right)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the game state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        return self.network(x)
    

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Stores and samples game state transitions for training the DQN. 
    Uses a deque with maximum length for efficient memory management. 
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize the replay buffer with fixed capacity.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: str, reward: int, next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the replay buffer.

        Args:
            state (np.ndarray): Current game state.
            action (str): Action taken ('up', 'down', 'left', 'right').
            reward (int): Reward received for the action.
            next_state (np.ndarray): Resulting game state.
            done (bool): Whether the episode ended.
        """
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.
        
        Returns:
            tuple: Batch of states, actions, rewards, next states, and done flags.
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self) -> int:
        """
        Return current buffer size.
        
        Returns:
            int: Number of transitions stored in the buffer.
        """
        return len(self.buffer)


class DQLearning:
    def __init__(self, game: Game):
        pass