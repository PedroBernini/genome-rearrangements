# coding: utf-8

from scipy.ndimage.interpolation import shift
from sklearn.utils import shuffle


# Fortemente baseado no capitulo 1 do livro do Mitchell (Machine
# Learning).  No livro tem a descrição de um jogo de damas, estou
# traduzindo para Genome Rearrangement.

import numpy as np

print_progress=False

class Player :
    def __init__(self, model, operations_selection_kind) :
        self.model = model
        self.kind  = operations_selection_kind ## Whether we break strips or not

class Game :
    def __init__(self, permutation, genome_problem) :
        self.size        = len(permutation)        
        self.permutation = [0]+permutation+[self.size+1]

        self.genome_problem = genome_problem
        self.set_upper_bound()
        
    def set_upper_bound(self, upper = None ) :
        if upper :
            self.upper_bound = upper
        else :
            self.upper_bound = self.genome_problem.upper_bound(self.permutation)
            
    def start_game(self) :
        permutation1 = list(self.permutation)
        permutation2 = list(self.permutation)
        self.permutations   = [
            permutation1,
            permutation2
        ]
        self.distance = 0

    def game_status(self) :
        p0 = self.genome_problem.is_sorted(self.permutations[0])
        p1 = self.genome_problem.is_sorted(self.permutations[1])

        if self.distance >  self.upper_bound  :
            return "loss"

        if p0 == p1 == False :
            return "In Progress"
        elif p0 == p1 == True :
            return "draw"
        elif p0 == True :
            return "p0"
        elif p1 == True :
            return "p1"


    def move_selector(self, player, permutation) :
        model = player.model
        
        best_move   = [None, -2]
        legal_moves = self.genome_problem.legal_move_generator(permutation, kind = player.kind)
        for legal_move in legal_moves :
            sigma    = self.genome_problem.perform_operation(permutation, legal_move)
            np_sigma = np.array(sigma[1:-1])    
            np_sigma = np.expand_dims(np_sigma, axis=0)
            score    = model.predict(np_sigma)[0,0]
            if score > best_move[1] :
                best_move[0] = sigma
                best_move[1] = score
        return best_move
    

    def game_loop(self, player0, player1) :
        if print_progress :
            print("___________________________________________________________________")
            print("Starting a new game")
    
        self.start_game()
        
        score_list0 = []
        score_list1 = []
    
        perm_list0 = []
        perm_list1 = []

        while self.game_status() == "In Progress" :
    
            if print_progress :
                print("___________________________________________________________________")
                print("NEW ROUND")
    
            self.permutations[0], score = self.move_selector(player0, self.permutations[0])
            score_list0.append(score)
            perm_list0.append (self.permutations[0][1:-1])
    
            self.permutations[1], score = self.move_selector(player1, self.permutations[1])
            score_list1.append(score)
            perm_list1.append (self.permutations[1][1:-1])

            self.distance += 1

        return score_list0, score_list1, perm_list0, perm_list1



class Reinforcement : 
    
    def shift_score_list(self, score_list0, score_list1, game) :
        corrected_scores_list0 = []
        corrected_scores_list1 = []
        
        score_list0 = np.array(score_list0)
        score_list1 = np.array(score_list1)
    
        status = game.game_status()
        if  status == "draw" :
            ## Not clear how should we recompensate draws. I decided to test several
            corrected_scores_list0 = shift(score_list0,-1,cval=0.8)
            corrected_scores_list1 = shift(score_list1,-1,cval=0.8)
    
        elif status == "p0" :
            corrected_scores_list0 = shift(score_list0,-1,cval= 1.0)
            corrected_scores_list1 = shift(score_list1,-1,cval=-1.0)
    
        elif status == "p1" :
            corrected_scores_list0 = shift(score_list0,-1,cval=-1.0)
            corrected_scores_list1 = shift(score_list1,-1,cval= 1.0)
    
        elif status == "loss": 
            corrected_scores_list0 = shift(score_list0,-1,cval=-1.0)
            corrected_scores_list1 = shift(score_list1,-1,cval=-1.0)
        else :
            raise ValueError("Unknow Option: " + status)     
   
        return corrected_scores_list0, corrected_scores_list1

    def train(self, player0, player1, game):
        score_list0, score_list1, perm_list0, perm_list1 = game.game_loop(player0, player1)

        corrected_scores_list0, corrected_scores_list1 = self.shift_score_list(score_list0, score_list1, game)
        
    
        if print_progress==True :
            print("Program has ",game.game_status() )
            print("\n Correcting the Scores and Updating the model weights:")
            print("___________________________________________________________________\n")
            
            print(corrected_scores_list0, perm_list0)
            print(corrected_scores_list1, perm_list1)
    
        corrected_scores_list0, perm_list0 = shuffle(corrected_scores_list0, perm_list0)
        corrected_scores_list1, perm_list1 = shuffle(corrected_scores_list1, perm_list1)
    
        corrected_scores_list0 = np.array(corrected_scores_list0)
        corrected_scores_list1 = np.array(corrected_scores_list1)        
        perm_list0             = np.array(perm_list0)
        perm_list1             = np.array(perm_list1)


        if len(perm_list0) != 0 :
            player0.model.fit(perm_list0, corrected_scores_list0, epochs=1, batch_size=1, verbose = 0)
            player1.model.fit(perm_list1, corrected_scores_list1, epochs=1, batch_size=1, verbose = 0)
