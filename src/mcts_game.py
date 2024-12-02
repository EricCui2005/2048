from game.game_logic import Game
from players.mcts_player import MCTSPlayer

def main():
    game = Game()
    player = MCTSPlayer(simulations=100, rollouts=100, c=100, discount=0.9)
    game.game_init()
    game.game_log()
    
    while True:
        
        # Test
        move = player.move(game)
        
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