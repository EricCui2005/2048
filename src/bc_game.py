from game.game_logic import Game
from players.bc_player import BCPlayer
import torch

def main():
    game = Game()
    player = BCPlayer()
    game.game_init()
    game.game_log()
    
    while True:
        
        move_priority = player.move(game)
        # move = torch.argmax(move_priority).item()
        values, indices = torch.topk(move_priority, k=4)
        for i in range(4):
            best_move = indices[i].item()
            before = game.matrix
            match best_move:
                case 0:
                    game.up()
                    game.game_log()
                case 1:
                    game.left()
                    game.game_log()
                case 2:
                    game.down()
                    game.game_log()
                case 3:
                    game.right()
                    game.game_log()
            after = game.matrix
            if before != after:
                break
            
        if game.game_over() == 1:
            print("You Win!")
            break
        if game.game_over() == -1:
            print("Game Over!")
            break
        
if __name__ == "__main__":
    main()