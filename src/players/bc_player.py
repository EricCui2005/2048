import torch
from players.bc_network import ImitationPolicyNet
import random

# Testing
from game.game_logic import Game

class BCPlayer:
    
    def __init__(self):
        self._model = ImitationPolicyNet()  # Recreate the model structure
        self._model.load_state_dict(torch.load('src/players/bc_model_weights.pth', map_location=torch.device('cpu')))
    
    def move(self, game):
        game_list = []
        for row in game.matrix:
            game_list += row
            
        result_tensor = self._model(torch.tensor(game_list).float())
        probs = []
        for element in result_tensor:
            probs.append(float(element))
        
        moves = ['up', 'down', 'left', 'right']
        print(probs)
        
        return random.choices(moves, weights=probs, k=1)[0]

    
    