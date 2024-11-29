from game.game_logic import Game
from math import sqrt, log
        

class MCTS:
    
    def __init__(self, simulations, rollouts, c):
        self._move_dict = {
            'left': {'N': 0, 'Q': 0},
            'down': {'N': 0, 'Q': 0},
            'right': {'N': 0, 'Q': 0},
            'up': {'N': 0, 'Q': 0}
        }
        self._total_visits = 0
        
        self._simulations = simulations
        self._rollouts = rollouts
        self._c = c
    
    
    def div(dividend, divisor):
        try:
            return dividend / divisor
        except ZeroDivisionError:
            if dividend == 0:
                raise ValueError("0/0 is undefined")
            return float('inf') if dividend > 0 else float('-inf')
    
    
    def UCB_choice(self):
        
        m_dict = self._move_dict
        ucbs = []
        for move in m_dict:
            ucb = m_dict[move]['Q'] + self._c * sqrt(self.div(log(self._total_visits), m_dict[move]['N']))
            ucbs.append((move, ucb))
        best = max(ucbs, key=lambda x: x[1])
        return best[0]
    
    
    def MCTS_move(self, state, simulations, rollouts):
        
        for _ in range(simulations):
            
            move = self.UCB_choice(c=100)
            