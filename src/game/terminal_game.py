from game_logic import Game

def main():
    game = Game()
    game.game_init()
    
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
        

if __name__ == "__main__":
    main()