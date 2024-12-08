import csv
from game.game_logic import Game
from players.mcts_player import MCTSPlayer
from players.random_player import Random
from players.random_corner_player import RandomCorner
 
def trials(player, num_trials):
    
    game = Game()
    res = {'Wins': 0, 'Losses': 0}
    
    with open('data/run1.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for i in range(num_trials):
            
            game.game_init()

            
            data = []
            
            while True:
                current_state = game.matrix
                current_score = game.score

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

                next_state = game.matrix
                next_score = game.score

                reward = next_score - current_score

                data.append((i, current_state, current_score, move, reward, next_state, next_score))
                    
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


def main():
    print(trials(player=MCTSPlayer(simulations=100, rollouts=10, c=100, discount=0.9), num_trials=20))


if __name__ == "__main__":
    main()