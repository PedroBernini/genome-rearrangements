# coding: utf-8

import re
import time
import sys
import numpy as np 
from sklearn.utils import shuffle
import datetime
import os
import random
import rearrangements
import models

from reinforcement import Game, Reinforcement

##################################################
## Main and Auxiliary Functions. 
##################################################

def get_database(filename) :
    data = []
    with open(filename) as f :
        for line in f : 
            match = re.match("([0-9,]*).*", line)
            if match : 
                data.append(eval("[%s]" % match.group(1)))
    return data

def instance_generator_easy(size, problem, operation_kind, num_instances) :
    instances_per_block = num_instances / 10 ## Quatro distancias
    
    pi = list(range(1, size+1))    
    legal_moves = problem.legal_move_generator(pi, "FREE")

    i = 0
    while i < num_instances :
        i = i + 1
        sigma = problem.perform_operation(pi, random.choice(legal_moves))

        extra_moves = None
        if operation_kind == "FREE" : 
            extra_moves  = int(i / instances_per_block)
        elif operation_kind == "BREAKPOINTS" :
            extra_moves  = int(i / instances_per_block) + 1
            
        for j in range(extra_moves) :
            sigma = problem.perform_operation(sigma, random.choice(legal_moves))
        yield sigma


def instance_generator_any(size, num_instances) :
    pi = list(range(1, size+1))
    i  = 0
    while i < num_instances :
        i = i + 1
        random.shuffle(pi)
        yield pi

def iterate_over_database(perm_generator, rein, problem, player0, player1) :
    results = {
        "draw"     : 0,
        "p0"       : 0, 
        "p1"       : 0,
        "loss"     : 0
    }
    soma          = 0
    game_counter  = 0

    for perm in perm_generator :
        game = Game(perm, problem)
        rein.train(player0, player1, game)    

        results[game.game_status()] += 1
        soma   += game.distance

        game_counter+=1
        if game_counter % 1000 == 0:
            print("Partial %s: %s --> %.3f" % (str(game_counter), str(results), float(soma)/game_counter ))
            

def save_models(genome_problem, player0, player1, m0, m1, epoch, batch_count, championship_label, operation_selection_kind, permutation_size) :
    directory = "saved_models/%s" % genome_problem.__class__.__name__.lower()
    if not os.path.isdir("saved_models") :
        os.makedirs(mydir)

    if not os.path.isdir(directory) :
        os.makedirs(directory)
        
    player0.model.save_weights("%s/player_%i_%i_%i_%s_%s_%s.h5" % (
        directory, m0, epoch, batch_count, championship_label, operation_selection_kind, permutation_size
    ))
    
    player1.model.save_weights("%s/player_%i_%i_%i_%s_%s_%s.h5" % (
        directory, m1, epoch, batch_count, championship_label, operation_selection_kind, permutation_size
    ))
    

if __name__ == "__main__" :
    ##################################################
    ## Command line arguments and Configurations
    ##################################################
    championship_label = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    permutation_size     = int(sys.argv[1])

    operation_selection_kinds = ["BREAKPOINTS", "FREE"]
    operation_selection_kind  = operation_selection_kinds[int(sys.argv[2])]

    genome_problems = [
        rearrangements.Unsigned_Reversal,
        rearrangements.Transposition,
        rearrangements.Unsigned_RevTrans,
        rearrangements.Prefix_Unsigned_Reversal,
        rearrangements.Prefix_Transposition,
        rearrangements.Prefix_Unsigned_RevTrans        
    ]
    genome_problem = genome_problems[int(sys.argv[3])]()

    easy_epoch   = int(sys.argv[4])
    normal_epoch = int(sys.argv[5])

    reinforcement = Reinforcement() ## Very importante Object

    m0 = int(sys.argv[6])
    m1 = int(sys.argv[7])

    player0 = models.select_player(m0, operation_selection_kind, permutation_size)
    player1 = models.select_player(m1, operation_selection_kind, permutation_size)

    if len(sys.argv) == 10 :
        player0.model.load_weights(sys.argv[8])
        player1.model.load_weights(sys.argv[9])


    epoch  = 0
    while epoch < easy_epoch :        
        perm_generator = instance_generator_easy(permutation_size, genome_problem, operation_selection_kind, 1000000000)
        
        iterate_over_database(perm_generator, reinforcement, genome_problem, player0, player1)
        epoch += 1
        save_models(genome_problem, player0, player1, m0, m1, -1, -1,
                    championship_label, operation_selection_kind,
                    permutation_size)
    
    epoch  = 0
    while epoch < normal_epoch :
        perm_generator = instance_generator_any(permutation_size, 1000000000)
        iterate_over_database(perm_generator, reinforcement, genome_problem, player0, player1)
        epoch += 1
        save_models(genome_problem, player0, player1, m0, m1,
                    epoch, -1, championship_label,
                    operation_selection_kind, permutation_size)



