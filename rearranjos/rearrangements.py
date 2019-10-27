# encoding: utf-8

##################################################
## Base Class: do not instantiate it. It is here just to document the
## methods I need to implement for all sub-problems.
##################################################
class Rearrangement :
    
    ## Return a list of valid moves.
    def legal_move_generator(self) :
        return []
    
    ## Give an upper bound for the training. Do not use a very tight
    ## upper bound.
    def upper_bound(self, permutation) :
        return int(2 * len(permutation))    

    ## Check if a permutation is sorted.
    def is_sorted(self, p) :        
        i = 0
        for j in p:
            if j != i:
                return False
            i += 1
        return True

    ## Apply the operation in the input permutation and return the
    ## result.
    def perform_operation(self, permutation, operation) :
        return list(permutation)


##################################################
## Class for unsigned reversals. It is the base class for all the
## other classes that use reversals
##################################################
class Unsigned_Reversal(Rearrangement) :
    def __init__(self) :
        self.reversals_generator_FREE = []

    def _breakpoints(self, permutation):
        b = []
        n = len(permutation)
        for i in range(1,n):
            if abs(permutation[i] - permutation[i-1]) != 1:
                b.append(i)
        return b

    def upper_bound(self, permutation) :
        return len(self._breakpoints(permutation))

    
    def _legal_reversals_BREAKPOINTS(self, permutation) :
        valid_moves = []
        breakpoints = self._breakpoints(permutation)
        num_breaks  = len(breakpoints)

        for i in range(num_breaks) :
            for j in range(i+1, num_breaks) :
                if breakpoints[i] != breakpoints[j]-1 :
                    valid_moves.append( 
                        (
                            breakpoints[i], 
                            breakpoints[j]-1) 
                    )
        return valid_moves

    def _legal_reversals_FREE(self, permutation) :
        if self.reversals_generator_FREE :
            return self.reversals_generator_FREE
        
        size = len(permutation)
        for i in range(1, size) :
            for j in range(i+1, size) :
                self.reversals_generator_FREE.append( (i,j)  )
        return self.reversals_generator_FREE


    ## Essa função vai devolver todas as operações que fazem
    ## sentido. No caso aqui apresentado, qualquer reversão que não
    ## gera breakpoints é dita com sentido.
    def legal_move_generator(self, permutation, kind) :
        if kind   == "BREAKPOINTS" :
            return self._legal_reversals_BREAKPOINTS(permutation) 
        elif kind == "FREE" :
            return self._legal_reversals_FREE(permutation) 
        raise ValueError("Invalid constrain for legal_move_generator")   


    def _perform_reversal(self, permutation, operation) :
        i,j = operation
        n   = len(permutation)
        new = []

        for x in range(0,i):
            new.append(permutation[x])
        for x in range(i,j+1):
            new.append(permutation[j-x+i])
        for x in range(j+1,n):
            new.append(permutation[x])
        return new        


    def perform_operation(self, permutation, operation) :
        return self._perform_reversal(permutation, operation)



##################################################
## Class for transpositions. It is the base class for all the
## other classes that use transpositions
##################################################
class Transposition(Rearrangement) :
    def __init__(self) :
        self.transposition_generator_FREE = []

    def _breakpoints(self, permutation):
        b = []
        n = len(permutation)
        for i in range(1,n):
            if permutation[i] - permutation[i-1] != 1:
                b.append(i)
        return b

    def _legal_transpositions_BREAKPOINTS(self, permutation) :
        valid_moves = []
        breakpoints = self._breakpoints(permutation)
        num_breaks  = len(breakpoints)

        for i in range(num_breaks) :
            for j in range(i+1, num_breaks) :
                for k in range(j+1, num_breaks) :
                    valid_moves.append( 
                        (
                            breakpoints[i], 
                            breakpoints[j],
                            breakpoints[k]
                        ) 
                    )
        return valid_moves

    def _legal_transpositions_FREE(self, permutation) :
        if self.transposition_generator_FREE :
            return self.transposition_generator_FREE
        
        size = len(permutation)
        for i in range(1, size) :
            for j in range(i+1, size) :
                for k in range(j+1, size) :
                    self.transposition_generator_FREE.append( (i,j,k) )
        return self.transposition_generator_FREE


    def legal_move_generator(self, permutation, kind) :
        if kind   == "BREAKPOINTS" :
            return self._legal_transpositions_BREAKPOINTS(permutation) 
        elif kind == "FREE" :
            return self._legal_transpositions_FREE(permutation) 
        raise ValueError("Invalid constrain for legal_move_generator")   


    def _perform_transposition(self, permutation, operation):
        i,j,k = operation
        n = len(permutation)
        new = []      
        for x in range(0,i):
          new.append(permutation[x])
        for x in range(j,k):
          new.append(permutation[x])
        for x in range(i,j):
          new.append(permutation[x])
        for x in range(k,n):
          new.append(permutation[x])
        return new

    def perform_operation(self, permutation, operation) :
            return self._perform_transposition(permutation, operation)


##################################################
class Unsigned_RevTrans(Unsigned_Reversal, Transposition) :
    def __init__(self) :
        self.reversals_generator_FREE     = []
        self.transposition_generator_FREE = []

    def _legal_RevTrans_BREAKPOINTS(self, permutation) :
        reversals      = self._legal_reversals_BREAKPOINTS(permutation) 
        transpositions = self._legal_transpositions_BREAKPOINTS(permutation)
        return reversals + transpositions

    def _legal_RevTrans_FREE(self, permutation) :
        reversals      = self._legal_reversals_FREE(permutation)
        transpositions = self._legal_transpositions_FREE(permutation) 
        return reversals + transpositions

    def legal_move_generator(self, permutation, kind) :
        if kind   == "BREAKPOINTS" :
            return self._legal_RevTrans_BREAKPOINTS(permutation) 
        elif kind == "FREE" :
            return self._legal_RevTrans_FREE(permutation) 
        raise ValueError("Invalid constrain for legal_move_generator")   

    def perform_operation(self, permutation, operation) :
        if len(operation) == 2 :
            return self._perform_reversal(permutation, operation)
        elif len(operation) == 3 :
            return self._perform_transposition(permutation, operation)


##################################################
class Prefix_Unsigned_Reversal(Unsigned_Reversal) :    
    def _breakpoints(self, permutation) :
        b = Unsigned_Reversal._breakpoints(self, permutation)
        if not "1" in b :
            b.insert(0, 1)
        return b

    def _legal_reversals_BREAKPOINTS(self, permutation) :
        valid_moves = []
        breakpoints = self._breakpoints(permutation)
        num_breaks  = len(breakpoints)

        for j in range(num_breaks) :
            if breakpoints[j] != 1 and breakpoints[j]-1 != 1 :
                valid_moves.append( 
                    (
                        1,
                        breakpoints[j]-1) 
                )
        return valid_moves

    def _legal_reversals_FREE(self, permutation) :
        if self.reversals_generator_FREE :
            return self.reversals_generator_FREE
        
        size = len(permutation)
        i = 1
        for j in range(i+1, size) :
            self.reversals_generator_FREE.append( (i,j)  )
        return self.reversals_generator_FREE


##################################################
class Prefix_Transposition(Transposition) :
    def _breakpoints(self, permutation) :
        b = Transposition._breakpoints(self, permutation)
        if not "1" in b :
            b.insert(0, 1)
        return b

    def _legal_transpositions_BREAKPOINTS(self, permutation) :
        valid_moves = []
        breakpoints = self._breakpoints(permutation)
        num_breaks  = len(breakpoints)

        for j in range(num_breaks) :
            for k in range(j+1, num_breaks) :
                if breakpoints[j] != 1 :
                    valid_moves.append( 
                        (
                            1, 
                            breakpoints[j],
                            breakpoints[k]
                        ) 
                )
        return valid_moves
    

    def _legal_transpositions_FREE(self, permutation) :
        if self.transposition_generator_FREE :
            return self.transposition_generator_FREE
        
        size = len(permutation)
        i = 1
        for j in range(i+1, size) :
            for k in range(j+1, size) :
                self.transposition_generator_FREE.append( (i,j,k) )
        return self.transposition_generator_FREE


##################################################
class Prefix_Unsigned_RevTrans(Prefix_Unsigned_Reversal, Prefix_Transposition, Unsigned_RevTrans) :

    def __init__(self) :
        Unsigned_RevTrans.__init__(self)

    def legal_move_generator(self, permutation, kind) :
        if kind   == "BREAKPOINTS" :
            return self._legal_RevTrans_BREAKPOINTS(permutation) 
        elif kind == "FREE" :
            return self._legal_RevTrans_FREE(permutation) 
        raise ValueError("Invalid constrain for legal_move_generator")   


    def perform_operation(self, permutation, operation) :
        if len(operation) == 2 :
            return self._perform_reversal(permutation, operation)
        elif len(operation) == 3 :
            return self._perform_transposition(permutation, operation)


##################################################
# Some code just to verify if the main methods of the above classes
# make sense.
##################################################
if __name__ == "__main__" :
    print("Unsigned Reversal")
    rev = Unsigned_Reversal()
    print(rev.is_sorted([0,1,2,3,4,5]))
    print(rev.legal_move_generator([0,3,2,1,5,4,6,7], "BREAKPOINTS"))
    print(rev.legal_move_generator([0,3,2,1,5,4,6,7], "FREE"))
    print(rev.perform_operation([0,3,2,1,5,4,6,7],(1,3)))
    
    print("\n Transposition")
    trans = Transposition()
    print(trans.is_sorted([0,1,2,3,4,5]))
    print(trans.legal_move_generator([0,3,2,1,5,4,6,7], "BREAKPOINTS"))
    print(trans.legal_move_generator([0,3,2,1,5,4,6,7], "FREE"))
    print(trans.perform_operation([0,3,2,1,5,4,6,7],(3,5,7)))

    print("\n Unsigned RevTrans")
    rt = Unsigned_RevTrans()
    print(rt.is_sorted([0,1,2,3,4,5]))
    print(rt.legal_move_generator([0,3,2,1,5,4,6,7], "BREAKPOINTS"))
    print(rt.legal_move_generator([0,3,2,1,5,4,6,7], "FREE"))
    print(rt.perform_operation([0,3,2,1,5,4,6,7],(3,5,7)))


    print("\n Unsigned Prefix Reversal")
    rev = Prefix_Unsigned_Reversal()
    print(rev.is_sorted([0,1,2,3,4,5]))
    print(rev.legal_move_generator([0,3,2,1,5,4,6,7], "BREAKPOINTS"))
    print(rev.legal_move_generator([0,3,2,1,5,4,6,7], "FREE"))
    print(rev.perform_operation([0,3,2,1,5,4,6,7],(1,3)))
    
    print("\n Prefix Transposition")
    trans = Prefix_Transposition()
    print(trans.is_sorted([0,1,2,3,4,5]))
    print(trans.legal_move_generator([0,3,2,1,5,4,6,7], "BREAKPOINTS"))
    print(trans.legal_move_generator([0,3,2,1,5,4,6,7], "FREE"))
    print(trans.perform_operation([0,3,2,1,5,4,6,7],(3,5,7)))

    print("\n Prefix Unsigned RevTrans")
    rt = Prefix_Unsigned_RevTrans()
    print(rt.is_sorted([0,1,2,3,4,5]))
    print(rt.legal_move_generator([0,3,2,1,5,4,6,7], "BREAKPOINTS"))
    print(rt.legal_move_generator([0,3,2,1,5,4,6,7], "FREE"))
    print(rt.perform_operation([0,3,2,1,5,4,6,7],(3,5,7)))
    print(rt.__class__.__name__)
