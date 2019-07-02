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
        if (permutation[start] - permutation[start + 1] == 1):
            stripDecreasing = True
            break
    return stripDecreasing

def takeSmallestElement(permutation, bkpMap) :
    smaller = len(permutation)
    for i in range(0, len(bkpMap) - 1) :
        start = bkpMap[i][1]
        end = bkpMap[i+1][0]
        if (permutation[start] - permutation[start + 1] == 1) :
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
        if (permutation[start] - permutation[start + 1] == 1) :
            while start <= end :
                if permutation[start] > bigger :
                    bigger = permutation[start]
                start += 1
    return bigger

def reWrite(permutation) :
    newPermutation = list(permutation)
    sequence = []
    for i in range(0, len(newPermutation)) :
        if i == len(newPermutation) - 1 :
            sequence.append(newPermutation[i])
        elif isAdjacent(newPermutation[i+1], newPermutation[i]) == False :
            sequence.append(newPermutation[i])
    k = 1
    while (k != len(sequence)) :
        if k in sequence :
            k += 1
        else :
            for i in range(0, len(sequence)) :
                if sequence[i] > k :
                    sequence[i] -= 1
    newPermutation = sequence
    return newPermutation

def mapDecreasingStrip(permutation, bkpMap) :
    stripMap = []
    for i in range(0, len(bkpMap) - 1) :
        start = bkpMap[i][1]
        end = bkpMap[i+1][0]
        if (permutation[start] - permutation[start + 1] == 1) :
            stripMap.append([start, end])
    return stripMap

def blockInterchange(i, j, k, l) :
    resultTransposition = [] 
    if j > k :
        i,j,k,l = k,l,i,j
    for m in range(0, i) :
        resultTransposition.append(permutation[m])
    for m in range(k, l) :
        resultTransposition.append(permutation[m])
    for m in range(j, k) :
        resultTransposition.append(permutation[m])
    for m in range(i, j) :
        resultTransposition.append(permutation[m])
    for m in range(l, len(permutation)) :
        resultTransposition.append(permutation[m])
    return resultTransposition


   
# ----- ALGORITMO ----- #
bkpMap = []
Permutations = [(list(permutation), "Original")]
blocksInterchanges = 0

print("\nPermutação:", permutation)
print("Quantidade de breakPoints:", breakPoints(permutation))
print("Mapa de breakPoints:", bkpMap);

# Algorithm
while(breakPoints(permutation) > 0) :
    if hasDecreasingStrip(permutation,bkpMap) == True :
        #    Para cada strip decrescente, faça virar crescente com block-interchange
        mapStrip = mapDecreasingStrip(permutation, bkpMap)
        for strip in mapStrip :
            while (strip[1] > strip[0]) :
                permutation = blockInterchange(strip[0], strip[0] + 1, strip[1], strip[1] + 1)
                Permutations.append(permutation)
                blocksInterchanges += 1
                strip[0] += 1
                strip[1] -= 1
    else :
        #    Reescrever permutação
        permutation = reWrite(permutation)
        Permutations.append((permutation,"ReWrite"))
        #    Mapear BkpMap
        breakPoints(permutation)
        if hasDecreasingStrip(permutation,bkpMap) == False :
            #    Se não tem strip decrescente, então faça um block-interchange que remova um breakpoint
            for i in range(0, len(permutation)) :
                if permutation[i] != len(permutation) - 1 :
                    permutation = blockInterchange(i+1, i+2, permutation.index(permutation[i]+1), permutation.index(permutation[i]+1)+1)
                    Permutations.append(permutation)
                    blocksInterchanges += 1
                    break
    
print("\nSequência de Permutações:")
for el in Permutations :
    print(el)
    
print("\nTotal de troca de blocos até a identidade:", blocksInterchanges)
    
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
