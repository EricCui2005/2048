import csv
import sys
from game.game_logic import Game
from players.mcts_player import MCTSPlayer
from players.random_player import Random
from players.random_corner_player import RandomCorner
from players.bc_player import BCPlayer
import torch
import tqdm

def BCReach1024(filename, player, num_trials):
    
    game = Game()
    res = {'Wins': 0, 'Losses': 0, 'Win Ratio' : 0}

    for j in tqdm.tqdm(range(num_trials)):
        
        game.game_init()

        
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
                    case 1:
                        game.left()
                    case 2:
                        game.down()
                    case 3:
                        game.right()
                after = game.matrix
                if before != after:
                    break

            if game.game_over1024() == 1:
                res['Wins'] += 1
                break
            if game.game_over1024() == -1:
                res['Losses'] += 1
                break
    res['Win Ratio'] = (res['Wins'] / (res['Losses'] + res['Wins']))
    return res

def BCReach512(filename, player, num_trials):
    
    game = Game()
    res = {'Wins': 0, 'Losses': 0, 'Win Ratio' : 0}

    for j in tqdm.tqdm(range(num_trials)):
        
        game.game_init()

        
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
                    case 1:
                        game.left()
                    case 2:
                        game.down()
                    case 3:
                        game.right()
                after = game.matrix
                if before != after:
                    break

            if game.game_over512() == 1:
                res['Wins'] += 1
                break
            if game.game_over512() == -1:
                res['Losses'] += 1
                break
    res['Win Ratio'] = (res['Wins'] / (res['Losses'] + res['Wins']))
    return res

def BCReach256(filename, player, num_trials):
    
    game = Game()
    res = {'Wins': 0, 'Losses': 0, 'Win Ratio' : 0}

    for j in tqdm.tqdm(range(num_trials)):
        
        game.game_init()

        
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
                    case 1:
                        game.left()
                    case 2:
                        game.down()
                    case 3:
                        game.right()
                after = game.matrix
                if before != after:
                    break

            if game.game_over256() == 1:
                res['Wins'] += 1
                break
            if game.game_over256() == -1:
                res['Losses'] += 1
                break
    res['Win Ratio'] = (res['Wins'] / (res['Losses'] + res['Wins']))
    return res

def BCReach128(filename, player, num_trials):
    
    game = Game()
    res = {'Wins': 0, 'Losses': 0, 'Win Ratio' : 0}

    for j in tqdm.tqdm(range(num_trials)):
        
        game.game_init()

        
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
                    case 1:
                        game.left()
                    case 2:
                        game.down()
                    case 3:
                        game.right()
                after = game.matrix
                if before != after:
                    break

            if game.game_over128() == 1:
                res['Wins'] += 1
                break
            if game.game_over128() == -1:
                res['Losses'] += 1
                break
    res['Win Ratio'] = (res['Wins'] / (res['Losses'] + res['Wins']))
    return res

def trialsReach1024(filename, player, num_trials):
    
    res = {'Wins': 0, 'Losses': 0, 'Win Ratio' : 0}
    game = Game()
    res = {'Wins': 0, 'Losses': 0}

    for i in range(num_trials):
        
        game.game_init()
        
        while True:
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
                
            if game.game_over1024() == 1:
                res['Wins'] += 1
                break
            if game.game_over1024() == -1:
                res['Losses'] += 1
                break
    
    res['Win Ratio'] = (res['Wins'] / (res['Losses'] + res['Wins']))
    return res

def trialsReach512(filename, player, num_trials):
    
    res = {'Wins': 0, 'Losses': 0, 'Win Ratio' : 0}
    game = Game()
    res = {'Wins': 0, 'Losses': 0}

    for i in range(num_trials):
        
        game.game_init()
        
        while True:
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
                
            if game.game_over512() == 1:
                res['Wins'] += 1
                break
            if game.game_over512() == -1:
                res['Losses'] += 1
                break
    
    res['Win Ratio'] = (res['Wins'] / (res['Losses'] + res['Wins']))
    return res

def trialsReach256(filename, player, num_trials):
    
    res = {'Wins': 0, 'Losses': 0, 'Win Ratio' : 0}
    game = Game()
    res = {'Wins': 0, 'Losses': 0}

    for i in range(num_trials):
        
        game.game_init()
        
        while True:
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
                
            if game.game_over256() == 1:
                res['Wins'] += 1
                break
            if game.game_over256() == -1:
                res['Losses'] += 1
                break
    
    res['Win Ratio'] = (res['Wins'] / (res['Losses'] + res['Wins']))
    return res

def trialsReach128(filename, player, num_trials):
    
    res = {'Wins': 0, 'Losses': 0, 'Win Ratio' : 0}
    game = Game()
    res = {'Wins': 0, 'Losses': 0}

    for i in range(num_trials):
        
        game.game_init()
        
        while True:
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
                
            if game.game_over128() == 1:
                res['Wins'] += 1
                break
            if game.game_over128() == -1:
                res['Losses'] += 1
                break
    
    res['Win Ratio'] = (res['Wins'] / (res['Losses'] + res['Wins']))
    return res

def append_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(str(text) + '\n')


def main(arg, detail):
    filepath = 'data/GameStageBechmarkResults' + str(arg) + '.txt'
    #print(trials(filepath, player=MCTSPlayer(simulations=100, rollouts=10, c=100, discount=0.9), num_trials=1000))

    append_to_file(filepath, detail)
    append_to_file(filepath, '\n' + '\n')

    # benchmarking early -> end game performance
    res = BCReach1024(filepath, player=BCPlayer(), num_trials=1000)
    append_to_file(filepath, "bc player - win condition: reach 1024 - 1000 trials:")
    append_to_file(filepath, res)
    append_to_file(filepath, '\n' + '\n')

    res = trialsReach1024(filepath, player=Random(), num_trials=1000)
    append_to_file(filepath, "random player - win condition: reach 1024 - 1000 trials:")
    append_to_file(filepath, res)
    append_to_file(filepath, '\n' + '\n')

    # also compare to optimal parameter MCTS

    res = BCReach512(filepath, player=BCPlayer(), num_trials=1000)
    append_to_file(filepath, "bc player - win condition: reach 512 - 1000 trials:")
    append_to_file(filepath, res)
    append_to_file(filepath, '\n' + '\n')

    res = trialsReach512(filepath, player=Random(), num_trials=1000)
    append_to_file(filepath, "random player - win condition: reach 512 - 1000 trials:")
    append_to_file(filepath, res)
    append_to_file(filepath, '\n' + '\n')

    res = BCReach256(filepath, player=BCPlayer(), num_trials=1000)
    append_to_file(filepath, "bc player - win condition: reach 256 - 1000 trials:")
    append_to_file(filepath, res)
    append_to_file(filepath, '\n' + '\n')
    
    res = trialsReach256(filepath, player=Random(), num_trials=1000)
    append_to_file(filepath, "random player - win condition: reach 256 - 1000 trials:")
    append_to_file(filepath, res)
    append_to_file(filepath, '\n' + '\n')

    res = BCReach128(filepath, player=BCPlayer(), num_trials=1000)
    append_to_file(filepath, "bc player - win condition: reach 128 - 1000 trials:")
    append_to_file(filepath, res)
    append_to_file(filepath, '\n' + '\n')

    res = trialsReach128(filepath, player=Random(), num_trials=1000)
    append_to_file(filepath, "random player - win condition: reach 128 - 1000 trials:")
    append_to_file(filepath, res)
    append_to_file(filepath, '\n' + '\n')

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide a run number and model detail (e.g. \" 1 mil training data, 300 epochs, 0.01 learning rate, structure: 16 -> ReLU -> 256 -> ReLu -> 4 \")")
        sys.exit(1)