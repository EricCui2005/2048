from game.game_logic import Game
from math import sqrt, log
import random
import copy
        

class MCTS:
    
    def __init__(self, simulations: int, rollouts: int, c: int, discount: int) -> None:
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
        self._discount = discount
    
    
    def div(dividend: int, divisor: int) -> float:
        try:
            return dividend / divisor
        except ZeroDivisionError:
            if dividend == 0:
                raise ValueError("0/0 is undefined")
            return float('inf') if dividend > 0 else float('-inf')
    
    
    def UCB_choice(self) -> str:
        
        m_dict = self._move_dict
        ucbs = []
        for move in m_dict:
            ucb = m_dict[move]['Q'] + self._c * sqrt(self.div(log(self._total_visits), m_dict[move]['N']))
            ucbs.append((move, ucb))
        best = max(ucbs, key=lambda x: x[1])
        return best[0]
    
    
    def MCTS_move(self, game_state: Game, simulations: int, rollouts: int) -> str:
        
        m_dict = self._move_dict
        
        for _ in range(simulations):
            
            # Selection
            move = self.UCB_choice(c=100)
            
            game_copy = copy.deepcopy(game_state)
            
            # Expansion
            match move:
                case 'left':
                    game_copy.left()
                case 'down':
                    game_copy.down()
                case 'right':
                    game_copy.right()
                case 'up':
                    game_copy.up()
            
            utility_estimate = 0
            
            # Simulation with random moving policy
            for i in range(rollouts):
                utility_estimate += (self._discount ** i) * game_copy.score
                sim_move = random.choice(['left', 'down', 'right', 'up'])
                match sim_move:
                    case 'left':
                        game_copy.left()
                    case 'down':
                        game_copy.down()
                    case 'right':
                        game_copy.right()
                    case 'up':
                        game_copy.up()
            
            # Update
            m_dict[move]['N'] += 1
            N = m_dict[move]['N']
            Q = m_dict[move]['Q']
            m_dict[move]['Q'] = ((N - 1) * Q + utility_estimate) / N
        
        # Returning move with the maximum updated Q-value
        return max(self._move_dict, key=lambda move: self._move_dict[move]['Q'])
            
            
                
            
            