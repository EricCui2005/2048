import csv
import sys
from game.game_logic import Game
from players.mcts_player import MCTSPlayer
from players.random_player import Random
from players.random_corner_player import RandomCorner
 
def trials(filename, player, num_trials):
    
    game = Game()
    res = {'Wins': 0, 'Losses': 0}
    
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for i in range(num_trials):
            
            game.game_init()

            
            data = []
            
            while True:
                current_state = game.matrix

                move = player.move(game)
                
                # UWU - eric
                match move:
                    case 'up':
                        game.up()
                    case 'left':
                        game.left()
                    case 'down':
                        game.down()
                    case 'right':
                        game.right()

                data.append((move, current_state))
                    
                if game.game_over() == 1:
                    print(f"Trial {i + 1}: Win")
                    res['Wins'] += 1
                    for row in data:
                        processed_row = [str(element) if isinstance(element, list) else element for element in row]
                        writer.writerow(processed_row)
                    break
                if game.game_over() == -1:
                    print(f"Trial {i + 1}: Loss")
                    res['Losses'] += 1
                    break
    
    return res


def main(arg):
    filepath = 'data/run' + str(arg) + '.csv'
    print(trials(filepath, player=MCTSPlayer(simulations=100, rollouts=10, c=100, discount=0.9), num_trials=1000))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a run number")
        sys.exit(1)