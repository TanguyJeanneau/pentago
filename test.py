from main import *
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

best, root = monte_carlo_tree_search(game, 1)

print(best, root)