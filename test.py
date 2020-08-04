from main import *
from time import time
def print_game(game):
    print(" ")
    for ligne in game.grid:
        print(*ligne)


game = Game()
def play_print(a,b,c,d):
    game.play(compatibility_create_move(a,b,c,d))
    print_game(game)
    
play_print(1, 1, 2, 3)
play_print(2, 2, 1, 2)
play_print(1, 2, 2, 3)
play_print(4, 0, 1, 4)
play_print(1, 4, 1, 4)
play_print(2, 4, 1, 1)

if True:
    best, root = monte_carlo_tree_search(game, 1)
    t = time()
    depth = 2
    best, score = min_max(game, depth = depth)
    print("Temps ecoule pour la profondeur", depth,'=',time()-t)

    print(best, score)
    game.play(best)
    print_game(game)
else:
    play_print(2, 4, 1, 1)
    print(game.fini)
    print(game.winner)

print()
