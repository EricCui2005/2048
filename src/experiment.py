from game.game_logic import Game
from players.mcts_player import MCTSPlayer
from players.random_player import Random
from players.random_corner_player import RandomCorner
 
def trials(player, num_trials):
    
    game = Game()
    res = {'Wins': 0, 'Losses': 0}
    
    for i in range(num_trials):
        
        game.game_init()
        
        while True:
            
            move = player.move(game)
            
            match move:
                case 'up':
                    game.up()
                case 'left':
                    game.left()
                case 'down':
                    game.down()
                case 'right':
                    game.right()
                
            if game.game_over() == 1:
                # print(f"Trial {i + 1}: Win")
                res['Wins'] += 1
                break
            if game.game_over() == -1:
                # print(f"Trial {i + 1}: Loss")
                res['Losses'] += 1
                break
    
    return res

def main():
    simulations = [20, 40, 60, 80, 100]
    rollouts = [20, 40, 60, 80, 100]
    
    for sim in simulations:
        for rollout in rollouts:
            print(f"Simulations: {sim} | Rollouts {rollout} | {trials(MCTSPlayer(simulations=sim, rollouts=rollout, c=100, discount=0.9), 100)}")
        
if __name__ == "__main__":
    main()