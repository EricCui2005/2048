from game.game_logic import Game
from players.mcts_player import MCTSPlayer
from players.random_player import Random
from players.random_corner_player import RandomCorner
 
def trials(num_trials):
    
    game = Game()
    player = RandomCorner()
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
                print(f"Trial {i}: Win")
                res['Win'] += 1
                break
            if game.game_over() == -1:
                print(f"Trial {i}: Loss")
                res['Losses'] += 1
                break
    
    return res

def main():
    print(trials(100))
        
if __name__ == "__main__":
    main()