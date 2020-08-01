from random import shuffle
from time import time
import numpy as  np 
#from libcpp cimport bool
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t


cdef enum Rotation:
    top_right_C = 1
    top_right_A = 2
    top_left_C = 3
    top_left_A = 4
    bottom_right_C = 5
    bottom_right_A = 6
    bottom_left_C = 7
    bottom_left_A = 8

cdef struct Coup:
    int x
    int y 
    Rotation rot

cdef class Game:

    cdef public np.int_t[:,:] grid
    
    cdef public int nb_moves
    
    cdef public int winner           

    cdef public int fini
    
    def __init__(self):
        self.grid = np.zeros((6,6), dtype = DTYPE)
        self.fini = 0
        self.winner = 0
        self.nb_moves = 0
        cdef int i
            
    cpdef Game copy(self):
        cdef Game other = Game()
        other.grid = self.grid.copy()
        other.fini = self.fini
        other.winner = self.winner
        other.nb_moves  = self.nb_moves
        other.hauteurs = self.hauteurs.copy()
        other.moves = self.moves.copy()
        return other
    
    cpdef int turn(self):
        return 1 + (self.nb_moves%2)
    

    cpdef int check_win(self):
        winners = []
        for i in range (6-3):
            for j in range(6-3):
                origin_tok_1 = self.grid[i, j]
                horizontal_match = all([self.grid[i, j + idx] == origin_tok_1 for idx in range(1, 4)]) if origin_tok_1 != 0 else False
                vertical_match = all([self.grid[i + idx, j] == origin_tok_1 for idx in range(1, 4)]) if origin_tok_1 != 0 else False
                diag_desc_match = all([self.grid[i + idx, j + idx] == origin_tok_1 for idx in range(1, 4)]) if origin_tok_1 != 0 else False
                origin_tok_2 = self.grid[i, j + 3]
                diag_asc_match = all([self.grid[i + idx, j - idx] == origin_tok_2 for idx in range(1, 4)]) if origin_tok_2 != 0 else False
                if (horizontal_match or vertical_match or diag_desc_match):
                    self.fini = True
                    winners.append(origin_tok_1)
                if diag_asc_match:
                    self.fini = True
                    winners.append(origin_tok_2)
        if len(winners) == 1:
            self.winner = winners[0]
        return self.fini

    cpdef int is_move_possible(self, int i, int j):
        return self.grid[i, j] != 0

    cpdef play(self, Coup coup):
        "assuming the move is possible"
        if self.fini:
            return
        cdef int turn = self.turn()
        self.grid[coup.x][coup.y] = turn
        self.apply_rotation(coup.rot)
        self.check_win()
        self.nb_moves += 1

    cdef apply_rotation(self, rot):
        pass

def from_grid(grid, turn = 1):
    game = Game()
    game.grid = grid
    if turn ==2:
        game.nb_moves = 1
    for i in range(7):
        for j in range(5,-1,-1):
            if grid[j, i] == 0:
                game.hauteurs[i]=j
                break
    return game
    
cdef Coup create_move(int x, int y, Rotation rot):
    cdef Coup coup
    coup.x = x
    coup.y = y
    coup.rot = rot
    return coup

def possible_moves(Game game):
    coups = []
    for i in range(6):
        for j in range(6):
            if game.grid[i][j] == 0:
                for rot in Rotation:
                    coups.append( create_move(i,j,rot))

cdef class Node:
    cdef public int expended
    cdef public int parent_move
    cdef public int nb_children
    cdef public Node[:] children
    cdef public Node parent
    cdef public Game game
    cdef public int win 
    cdef public int visited 
    
    def __init__(self, Game game, int parent_move = -1,):
        self.expended = False
        self.game = game
        self.parent_move = parent_move
        self.win = 0
        self.visited = 0
        self.children = np.zeros(len(possible_moves(self.game)), dtype = np.object)
        
    cpdef expend(self):
        cdef int i 
        cdef int index = 0
        cdef Game otherGame
        coups_possibles = possible_moves(self.game)
        for i in range(len(coups_possibles)):
            otherGame = self.game.copy()
            otherGame.play(coups_possibles[i])
            self.children[index] = Node(otherGame, i)
            self.children[index].parent = self
            index += 1
        self.nb_children = index
        self.expended = True
        
    cpdef Node random_child(self):
        return self.children[np.random.randint(self.nb_children)]
    
    cpdef update_stats(self, winner):
        self.visited += 1
        if winner != self.game.turn():
            self.win +=1
            
cdef float eval_score(Node self):
    cdef int visits  
    if self.visited == 0:
        visits = 1
    else:
        visits = self.visited 
    cdef float res = self.win / visits
    res += np.sqrt(2*np.log(self.parent.visited) / visits) #assert self.parent.visited >= 1, sinon c'est que l'algo marche pas comme prévu
    return res

def win_rate(self):
    if self.visited == 0:
        return 0
    else:
        return self.win / self.visited 

cdef class Evaluation:
    cpdef int evaluate(self, Game jeu) except *:
        return 0

cdef class Evaluation_simple(Evaluation):
    cpdef int evaluate(self, Game jeu) except *:
        if jeu.winner == 1:
            return 1
        elif jeu.winner == 2:
            return -1
        else:
            return 0

  
def monte_carlo_tree_search(game, time_allocated): 
    d = time()
    current_time = 0
    root = Node(game.copy())
    cdef int iterations = 0
    while current_time < time_allocated: 
        leaf = traverse(root)  
        simulation_result = rollout(leaf)
        backpropagate(leaf, simulation_result)
        iterations += 1
        current_time = time() - d
    cdef float best_score =-1
    cdef float score
    cdef int best = -1
    for i in range(root.nb_children):
        score = win_rate(root.children[i])
        if score > best_score:
            best_score = score
            best = root.children[i].parent_move
    return best 
  
cdef Node traverse(node): 
    while node.expended: 
        if node.nb_children == 0:
            return node
        node = best_child(node)
    node.expend()
    if node.nb_children == 0:
        return node
    else:
        return node.random_child()
    
# function for the result of the simulation 
cdef int rollout(Node node):
    cdef Game game = node.game.copy()
    cdef coups_possibles
    cdef coup_choisi
    while not game.fini: 
        coups_possibles = possible_moves(game)
        coup_choisi = coups_possibles[1 + np.random.randint(coups_possibles[0])]
        game.play(coup_choisi)
    return game.winner
  
# function for backpropagation 
cdef backpropagate(Node node, int result): 
    node.update_stats(result)  
    if not node.parent:
        return
    backpropagate(node.parent, result) 

# function for selecting the best child 
# node with highest number of visits 
cdef Node best_child(Node node): 
    cdef Node best = node
    cdef float best_score = -1
    cdef float score
    cdef int i = 0 
    for i in range(node.nb_children):
        score = eval_score(node.children[i])
        if score > best_score:
            best_score = score
            best = node.children[i]
    return best