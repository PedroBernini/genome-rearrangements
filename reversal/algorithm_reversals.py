# The Natural Algorithm - Watterson
# The Greedy Algorithm - Kececioglu
# Autor: Pedro Henrique Bernini Silva

import sys
import os

# ----- PERMUTAÇÃO (LINHA DE COMANDO) ----- #
permutation = []
n = len(sys.argv)

for el in sys.argv[1].split(",") :
        permutation.append(int(el))

# ----- FUNÇÕES ----- #
def isAdjacent(a, b) :
    if (abs(a - b) == 1) :
        return True
    else :
        return False

def breakPoints(permutation) :
    bkpMap.clear()
    breakpoints = 0
    for i in range(0,len(permutation)-1) :
        if (isAdjacent(permutation[i+1], permutation[i]) is False):
            breakpoints += 1
            bkpMap.append((i,i+1))
    return breakpoints

def reversal(i, j) :
    strip = []
    if(i>j) :
        temp = i
        i = j
        j = temp
    for k in range(i,j+1) :
        strip.append(permutation[k]) 
    strip.reverse();
    for k in range(i,j+1) :
        permutation[k] = strip[k-i]
    return permutation
   
# ----- ALGORITMO ----- #
bkpMap = []
Permutations = [list(permutation)]
reversoes = 0

print("\nPermutação:", permutation)
print("Quantidade de Breakpoints:", breakPoints(permutation))
print("Mapa de BreakPoints:", bkpMap);

# The Natural Algorithm - Watterson
while(breakPoints(permutation) > 0) :
    for    i in range(0, len(permutation)) :
        if(permutation[i] != i + 1) :
            Permutations.append(list(reversal(i, permutation[i]-1)))
            reversoes += 1

# The Greedy Algorithm - Kececioglu
#while(breakPoints(permutation) > 0) :
    
#    Encontrar uma reversão que diminua 2 breakpoints
#    for first in bkpMap :
#        for second in bkpMap :
#            if isAdjacent(permutation[first[0]], permutation[second[0]]) and isAdjacent(permutation[first[1]], permutation[second[1]]) :
#                Permutations.append(list(reversal(first[1], second[0])))
#                reversoes += 1
            
#    Encontrar uma reversão que diminua 1 breakpoints e deixe uma split decrescente
            
#    Encontrar uma reversão que diminua 1 breakpoints
            
#    Encontrar uma reversão que deixe uma split decrescente


print("Total de reversões até a identidade:", reversoes)
print("\nSequência de Permutações:")
for el in Permutations :
    print(el)
    
drawString = ''
for permutation in Permutations :
    drawString += '"'
    for i in range(0, len(permutation)) :
        drawString += str(permutation[i])
        if i != len(permutation) - 1 :
            drawString += ","
    drawString += '" '
    
print("\nDraw_Canvas( ", drawString, ")")
os.system('python canvas.py ' + drawString)
