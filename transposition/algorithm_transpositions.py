# Algorithm - Transpositions
# Autor: Pedro Henrique Bernini Silva

import sys
import os

# ----- PERMUTAÇÃO (LINHA DE COMANDO) ----- #
permutation = []
n = len(sys.argv)
last = 0

permutation.append(0)
for el in sys.argv[1].split(",") :
    identifier = int(el)
    permutation.append(identifier)
    if identifier > last :
        last = identifier
permutation.append(last + 1)

# ----- FUNÇÕES ----- #
def isAdjacent(a, b) :
    if b + 1 == a :
        return True
    else :
        return False

def breakPoints(permutation) :
    bkpMap.clear()
    breakPoints = 0
    for i in range(0, len(permutation)-1) :
        if (isAdjacent(permutation[i+1], permutation[i]) is False):
            breakPoints += 1
            bkpMap.append((i,i+1))
    return breakPoints

def sequenceAdjacents(index) :
    adjacents = 1
    j = 0
    prox = index + j + 1
    while prox <= len(permutation) - 1 and isAdjacent(permutation[index + j + 1], permutation[index + j]) == True :
        j += 1
        prox = index+j+1
        adjacents += 1
    return adjacents

def transposition(i, j, k) :
    resultTransposition = []
    strip = []
    for l in range(i, j) :
        strip.append(permutation[l])   
    if j < k :
        for l in range(0, i) :
            resultTransposition.append(permutation[l])
        for l in range(j, k) :
            resultTransposition.append(permutation[l])
        for el in strip :
            resultTransposition.append(el)
        for l in range(k, len(permutation)) :
            resultTransposition.append(permutation[l])
    else :
        for l in range(0, k) :
            resultTransposition.append(permutation[l])
        for el in strip :
            resultTransposition.append(el)
        for l in range(k, i) :
            resultTransposition.append(permutation[l])
        for l in range(j, len(permutation)) :
            resultTransposition.append(permutation[l])
    return resultTransposition 
   
# ----- ALGORITMO ----- #
bkpMap = []
Permutations = [list(permutation)]
transpositions = 0

print("\nPermutação:", permutation)
print("Quantidade de breakPoints:", breakPoints(permutation))
print("Mapa de breakPoints:", bkpMap);

# SelectionSort Algorithm
while(breakPoints(permutation) > 0) :
    for i in range(0, len(permutation)) :
        for el in permutation :
            if i == el and i != permutation[i] :
                index = permutation.index(el)
                adjacents = sequenceAdjacents(index)
                resultTranspositions = transposition(index, index+adjacents, el)
                permutation = resultTranspositions
                Permutations.append(list(permutation))
                transpositions += 1
                break

print("\nSequência de Permutações:")
for el in Permutations :
    print(el)
    
print("\nTotal de transposições até a identidade:", transpositions)

drawString = ''
for permutation in Permutations :
    drawString += '"'
    for i in range(1, len(permutation) - 1) :
        drawString += str(permutation[i])
        if i != len(permutation) - 2 :
            drawString += ","
    drawString += '" '
    
print("\nDraw_Canvas( ", drawString, ")")
os.system('python canvas.py ' + drawString)
