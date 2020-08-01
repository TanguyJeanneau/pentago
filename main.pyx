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

cdef class Game:
    cdef public int moves[6*7]
    
    cdef public np.int_t[:,:] grid
    
    cdef public int nb_moves
    
    cdef public int winner           

    cdef public int fini 
    cdef public int [7] hauteurs 
    
    def __init__(self):
        self.grid = np.zeros((6,7), dtype = DTYPE)
        self.fini = 0
        self.winner = 0
        self.nb_moves = 0
        cdef int i
        for i in range(7):
            self.hauteurs[i]= 5
            
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

    cpdef undo(self):
        colonne = self.moves[self.nb_moves -1]
        row = self.row(colonne) + 1
        self.fini = False
        self.winner = 0
        self.grid[row, colonne] = 0
        self.hauteurs[colonne] +=1
        self.nb_moves -= 1
    

    cpdef int check_win(self, int i, int j ):
        cdef DTYPE_t joueur = self.grid[i, j]
        cdef int xi, xj, count, offset
        cdef int idx
        for idx in range(4):
            if idx == 0:
                xi, xj = 0, 1
            elif idx == 1:
                xi, xj = 1, 1
            elif idx == 2:
                xi, xj = 1, 0
            else:
                xi, xj =-1, 1
            count = 1
            offset = 1
            while are_valid(i + offset * xi, j + offset * xj) and self.grid[i + offset * xi, j + offset * xj] == joueur:
                count +=1
                offset +=1
            offset = -1
            while are_valid(i + offset * xi, j + offset * xj) and self.grid[i + offset * xi, j + offset * xj] == joueur:
                count +=1
                offset -=1
            if count >=4:
                self.fini = True
                self.winner = joueur
                return True
        self.fini = True
        for idx in range(7):
            if self.grid[0,idx] == 0:
                self.fini = False
                return False
        return True

    cpdef int is_move_possible(self, int colonne):
        return self.grid[0,colonne] == 0

    cpdef int row(self, int colonne):
        return self.hauteurs[colonne]

    cpdef play(self, int colonne):
        "assuming the move is possible"
        if self.fini:
            return
        cdef int i = self.row(colonne)
        cdef int turn = self.turn()
        self.grid[i][colonne] = turn
        self.moves[self.nb_moves] = colonne
        self.check_win(i,colonne)
        self.hauteurs[colonne] -=1
        self.nb_moves += 1

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
    
    
cdef int* possible_moves(Game game):
    cdef int[8] moves
    cdef int index = 0, i
    for i in range(7):
        if game.is_move_possible(i):
            index +=1
            moves[index] = i 
    moves[0] = index
    return moves

cdef int are_valid(int i, int j):
    return 0 <= i <6 and  0<= j <7 

cdef double alpha_beta(Game jeu, Evaluation evaluation, int depth, float alpha, float beta):
    cdef int joueur = jeu.turn()
    cdef float best_score = float('inf') * (-1 if joueur ==1 else 1)
    if depth == 0 or jeu.fini:
        return evaluation.evaluate(jeu)
    cdef int colonne
    cdef float score
    for colonne in range(7):
        if jeu.is_move_possible(colonne):
            jeu.play(colonne)
            score = alpha_beta(jeu, evaluation, depth - 1 , alpha, beta)
            jeu.undo()
            if joueur == 1 and score > best_score or joueur == 2 and score < best_score:
                best_score = score
            if joueur ==1:
                if score > beta :
                    return score 
                alpha = max(alpha, score)
            if joueur ==2:
                if score < alpha:
                    return score 
                beta = min(beta, score)
    return best_score

cpdef int min_max(jeu, Evaluation evaluation, depth = 3):
    cdef float alpha = float('-inf')
    cdef float beta = float('inf')
    cdef int joueur = jeu.turn()
    cdef int best = -1
    cdef float best_score = float('inf') * (-1 if joueur ==1 else 1)
    cdef int colonne
    cdef float score
    for colonne in np.random.permutation(7):
        if jeu.is_move_possible(colonne):
            jeu.play(colonne)
            score = alpha_beta(jeu, evaluation, depth - 1 , alpha, beta)
            jeu.undo()
            if joueur == 1 and score > best_score or joueur == 2 and score < best_score:
                best = colonne
                best_score = score
            if joueur ==1:
                alpha = max(alpha, score)
            if joueur ==2:
                beta = min(beta, score)
    return best



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
        self.children = np.zeros(7, dtype = np.object)
        
    cpdef expend(self):
        cdef int i 
        cdef int index = 0
        cdef Game otherGame
        for i in range(7):
            if self.game.is_move_possible(i):
                otherGame = self.game.copy()
                otherGame.play(i)
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
    cdef int* coups_possibles
    cdef int coup_choisi
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