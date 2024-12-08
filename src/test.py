from players.bc_player import BCPlayer
from game.game_logic import Game

print("Hello world")
player = BCPlayer()

game = Game()
game.game_init()
print(player.move(game))


