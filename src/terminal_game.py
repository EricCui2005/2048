from game.game_logic import Game

def main():
    game = Game()
    game.game_init()
    game.game_log()
    
    while True:
        move = input("Move: ")

        match move:
            case 'w':
                game.up()
                game.game_log()
            case 'a':
                game.left()
                game.game_log()
            case 's':
                game.down()
                game.game_log()
            case 'd':
                game.right()
                game.game_log()
            case '_':
                continue
            
        if game.game_over() == 1:
            print("You Win!")
            break

        if game.game_over() == -1:
            print("Game Over!")
            break
        
if __name__ == "__main__":

    main()