import sys

# ----- PERMUTAÇÃO (LINHA DE COMANDO) ----- #
permutacaoInicial = [0]
permutacaoFinal = []
n = len(sys.argv) - 1
ultimo = None

for el in range(1, n) :
    permutacaoInicial.append(eval(sys.argv[el]))
    if abs(eval(sys.argv[el])) == n - 1 :
        if eval(sys.argv[el]) < 0 :
            ultimo = (eval(sys.argv[el]) - 1) * -1
        else :
            ultimo = (eval(sys.argv[el]) + 1) * -1
permutacaoInicial.append(ultimo)

for el in range(0, n + 1) :
    permutacaoFinal.append(el)

print("\nPermutacao Inicial", permutacaoInicial)
print("\nPermutacao Final  ", permutacaoFinal)



