# coding: utf-8

import re
import sys
import numpy as np 
import models
import rearrangements


def sort(player, genome_problem, permutation, operation_selection_kind):
    permutation    = [0] + permutation + [len(permutation)+1]
    upper_bound    = genome_problem.upper_bound(permutation)
    operations     = []

    while (not genome_problem.is_sorted(permutation) and len(operations) < upper_bound) :
        best_move   = [None, -2, None]
        legal_moves = genome_problem.legal_move_generator(permutation, kind = player.kind)
        #print(legal_moves)

        for legal_move in legal_moves :
            sigma = genome_problem.perform_operation(permutation, legal_move)
            np_sigma = np.array(sigma[1:-1])    
            np_sigma = np.expand_dims(np_sigma, axis=0)
            score    = player.model.predict(np_sigma)[0,0]
            if score > best_move[1] :
                best_move[0] = sigma
                best_move[1] = score
                best_move[2] = legal_move
        operations.append(best_move[2])
        permutation = best_move[0]
    return operations
    

if __name__ == "__main__" :
    genome_problems = [
        rearrangements.Unsigned_Reversal,
        rearrangements.Transposition,
        rearrangements.Unsigned_RevTrans,
        rearrangements.Prefix_Unsigned_Reversal,
        rearrangements.Prefix_Transposition,
        rearrangements.Prefix_Unsigned_RevTrans
    ]
    genome_problem = genome_problems[int(sys.argv[1])]()

    database_file = sys.argv[2]

    
    
    players         = []
    pnames          = []

    for model_file in sys.argv[3:] :
        match = re.match(".*player_([0-9])_.*_.*_.*_(FREE|BREAKPOINTS)_([0-9]+).h5", model_file)
        if match :
            PLAYER                   = int(match.group(1))
            OPERATION_SELECTION_KIND = match.group(2)
            PERMUTATION_SIZE         = int(match.group(3))

            ## Loading trained model
            player = models.select_player(PLAYER, OPERATION_SELECTION_KIND, PERMUTATION_SIZE)

            player.model.load_weights(model_file)
            players.append(player)
            pnames.append(model_file)


    soma  = [0 for i in range(len(players))]
    count = 0
    with open(database_file) as f :
        for line in f :
            match = re.match("([0-9,]*)[ \t\n\r\f\v].*", line)
            if match : 
                for i in range(len(players)) :
                    model = players[i]
                    perm  = eval("[%s]" %match.group(1))
                    l_operations = sort(model, genome_problem, perm, OPERATION_SELECTION_KIND)
                    soma[i] += len(l_operations)
                        
                count += 1

    
    for i in range(len(pnames)) :
        print("%s,%s" % (pnames[i], float(soma[i])/count))
