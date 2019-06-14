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

def reversal(i, j) :
    resultReversal = list(permutation)
    strip = []
    if(i>j) :
        temp = i
        i = j
        j = temp
    for k in range(i,j+1) :
        strip.append(resultReversal[k]) 
    strip.reverse();
    for k in range(i,j+1) :
        resultReversal[k] = strip[k-i]
    return resultReversal
   
# ----- ALGORITMO ----- #
bkpMap = []
Permutations = [list(permutation)]
reversoes = 0

print("\nPermutação:", permutation)
print("Quantidade de breakPoints:", breakPoints(permutation))
print("Mapa de breakPoints:", bkpMap);

# The Natural Algorithm
#while(breakPoints(permutation) > 0) :
#    for i in range(0, len(permutation)) :
#        if(permutation[i] != i + 1) :
#            permutation = reversal(i, permutation[i]-1)
#            Permutations.append(list(permutation))
#            reversoes += 1

# The Greedy Algorithm - Kececioglu
while(breakPoints(permutation) > 0) :
    
    resultReversal = None
    if hasDecreasingStrip(permutation, bkpMap) :
        # Encontrar o menor elemento de strip decrescente
        k = takeSmallestElement(permutation, bkpMap)   
        if permutation.index(k) < permutation.index(k - 1) :
            resultReversal = reversal(permutation.index(k)+1,permutation.index(k-1))
        else :
            resultReversal = reversal(permutation.index(k),permutation.index(k-1)+1)
        
        # Encontrar o maior elemento de strip decrescente
        breakPoints(resultReversal)
        if hasDecreasingStrip(resultReversal, bkpMap) == False :
            breakPoints(permutation)
            l = takeBiggestElement(permutation, bkpMap)
            if permutation.index(l) > permutation.index(l + 1) :
                resultReversal = reversal(permutation.index(l)-1,permutation.index(l+1))
            else :
                resultReversal = reversal(permutation.index(l),permutation.index(l+1)-1)
    else :
        resultReversal = reversal(bkpMap[0][1],bkpMap[1][0])
    
    permutation = resultReversal
    Permutations.append(list(permutation))
    reversoes += 1
    

print("\nTotal de reversões até a identidade:", reversoes)
print("\nSequência de Permutações:")
for el in Permutations :
    print(el)
    
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
