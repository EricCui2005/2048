import random
from collections import deque
import torch
import torch.nn as nn
import numpy as np
from game.game_logic import Game

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


class DQLearningAgent:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)

        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.action_map = {
            0: 'up',
            1: 'left',
            2: 'down',
            3: 'right'
        }

    def get_state(self) -> np.ndarray:
        state = np.array(self.game.matrix).astype(np.float32)
        return np.log2(state + 1) / 11.0 # max tile is 2048 = 2^11
    
    def select_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()


def train_agent(episodes=10000, save_path='src/policies/2048_model.pth'):
    pass


class DQLearningPlayer:
    def __init__(self, game: Game, model_path: str) -> None:
        self.game = game

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN().to(self.device)

        # Load trained model
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.action_map = {
            0: 'up',
            1: 'left',
            2: 'down',
            3: 'right'
        }
    
    def get_state(self):
        state = np.array(self.game.matrix).astype(np.float32)
        return np.log2(state + 1) / 11.0  # max tile is 2048 = 2^11
    
    def move(self):
        state = self.get_state()
        state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = q_values.argmax().item()
        
        return self.action_map[action]


def main():
    train_agent(episodes=10000)

    game = Game()
    player = DQLearningPlayer(game, 'src/policies/2048_model.pth')
    game.game_init()

    while True:
        move = player.move()

        match move:
            case 'up':
                game.up()
                game.game_log()
            case 'left':
                game.left()
                game.game_log()
            case 'down':
                game.down()
                game.game_log()
            case 'right':
                game.right()
                game.game_log()
        
        if game.game_over() == 1:
            print("You Win!")
            break

        if game.game_over() == -1:
            print("Game Over!")
            break


if __name__ == "__main__":
    main()