import torch
from players.bc_network import ImitationPolicyNet
import random
import numpy as np

# Testing
from game.game_logic import Game

class BCPlayer:
    
    def __init__(self):
        self._model = ImitationPolicyNet().to('cpu')  # Recreate the model structure
        self._model.load_state_dict(torch.load('players/bc_model_weights_12.pth', map_location=torch.device('cpu')))
    
    def move(self, game, device='cpu'):
        game_list = []
        for row in game.matrix:
            game_list += row
        
        # Normalize state (log2 of tiles, 0 for empty)
        new = [x + 1 for x in game_list]
        final_list = np.log2(new) / 11.0
            
        probs = self._model(torch.FloatTensor(final_list).to(device))
        # probs = []
        # for element in result_tensor:
        #     probs.append(float(element))
        
        # moves = ['up', 'down', 'left', 'right']
        # print(probs)
        
        # return random.choices(moves, weights=probs, k=1)[0]
        return probs

    
    