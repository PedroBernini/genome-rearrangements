# The Natural Algorithm - Watterson
# The Greedy Algorithm - Kececioglu
# Autor: Pedro Henrique Bernini Silva

import sys

# ----- PERMUTAÇÃO (LINHA DE COMANDO) ----- #
permutation = []
n = len(sys.argv)

for el in sys.argv[1].split(",") :
        permutation.append(int(el))

# ----- FUNÇÕES ----- #
def breakPoints(permutation) :
    breakpoints = 0
    for i in range(0,len(permutation)-1) :
        if (abs(permutation[i+1] - permutation[i]) != 1) :
            breakpoints += 1
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
print("\nPermutação:", permutation)
print("Quantidade de Breakpoints:", breakPoints(permutation))

Permutations = [list(permutation)]
reversoes = 0
while(breakPoints(permutation) > 0) :
    for i in range(0, len(permutation)) :
        if(permutation[i] != i + 1) :
            Permutations.append(list(reversal(i, permutation[i]-1)))
            reversoes += 1

print("\nPermutação Final:", permutation)
print("Quantidade de Breakpoints:", breakPoints(permutation))

print("Total de reversões até a identidade:", reversoes)
print("\nSequência de Permutações:")
for el in Permutations :
    print(el)