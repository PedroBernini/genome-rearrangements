# The Natural Algorithm
# The Greedy Algorithm - Kececioglu
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
    if (abs(a - b) == 1) :
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

def hasDecreasingStrip(permutation, bkpMap) :
    stripDecreasing = False
    for i in range(0, len(bkpMap) - 1) :
        start = bkpMap[i][1]
        end = bkpMap[i+1][0]
        if start == end :
            stripDecreasing = True
            break
        elif (permutation[start] - permutation[start + 1] == 1):
            stripDecreasing = True
            break
    return stripDecreasing

def takeSmallestElement(permutation, bkpMap) :
    smaller = len(permutation)
    for i in range(0, len(bkpMap) - 1) :
        start = bkpMap[i][1]
        end = bkpMap[i+1][0]
        if start == end and permutation[start] < smaller :
            smaller = permutation[start]
        elif (permutation[start] - permutation[start + 1] == 1) :
            while start <= end :
                if permutation[start] < smaller :
                    smaller = permutation[start]
                start += 1
    return smaller

def takeBiggestElement(permutation, bkpMap) :
    bigger = 0
    for i in range(0, len(bkpMap) - 1) :
        start = bkpMap[i][1]
        end = bkpMap[i+1][0]
        if start == end and permutation[start] > bigger :
            bigger = permutation[start]
        elif (permutation[start] - permutation[start + 1] == 1) :
            while start <= end :
                if permutation[start] > bigger :
                    bigger = permutation[start]
                start += 1
    return bigger

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
        if(permutation[i] != i) :
            resultTranspositions = transposition(i, i+1, permutation[i]+1)
            permutation = resultTranspositions
            Permutations.append(list(permutation))
            transpositions += 1
    
print("\nTotal de transposições até a identidade:", transpositions)
print("\nSequência de Permutações:")
for el in Permutations :
    print(el)
    
#drawString = ''
#for permutation in Permutations :
#    drawString += '"'
#    for i in range(1, len(permutation) - 1) :
#        drawString += str(permutation[i])
#        if i != len(permutation) - 2 :
#            drawString += ","
#    drawString += '" '
#    
#print("\nDraw_Canvas( ", drawString, ")")
#os.system('python canvas.py ' + drawString)