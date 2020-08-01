from main import *
def print_game(game):
    print(" ")
    for ligne in game.grid:
        print(*ligne)


game = Game()
print_game(game)
game.play(compatibility_create_move(0, 0, 1, 1))
print_game(game)