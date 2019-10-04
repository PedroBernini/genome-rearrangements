# Objetivo: Este programa deve apontar quantos ciclos uma permutação possui e dizer quais são bons e quais são ruins.
# Autor: Pedro Henrique Bernini Silva.

import sys

# ----- PERMUTAÇÃO (LINHA DE COMANDO) ----- #
permutacao = []
n = len(sys.argv)
ultimo = None

for el in range(1, n) :
    permutacao.append(eval(sys.argv[el]))
    if abs(eval(sys.argv[el])) == n - 1 :
        ultimo = (abs(eval(sys.argv[el])) + 1) * -1

# ----- CRIAÇÃO DAS LINHAS PRETAS (TUPLAS) ----- #
linhasPretas = []
for i in range(0, len(permutacao)) :
    if i == 0 :
        linhasPretas.append((0, permutacao[i] * -1))
        linhasPretas.append((permutacao[i], permutacao[i + 1] * -1))
    elif i == len(permutacao) - 1 :
        linhasPretas.append((permutacao[i], ultimo))
    else :
        linhasPretas.append((permutacao[i], permutacao[i + 1] * -1))

# ----- CRIAÇÃO DOS VÉRTICES ----- #
listaVertices = []
for tupla in linhasPretas :
    listaVertices.append(tupla[0])
    listaVertices.append(tupla[1])

# ----- CRIAÇÃO DAS ARESTAS (CONEXÕES) ----- #
listaArestas = []
for tupla in linhasPretas :
    listaArestas.append((tupla[0], tupla[1]))
for i in range(0, len(linhasPretas)) :
        listaArestas.append((i,(i+1) * -1))

# ----- CRIAÇÃO DAS LIGAÇÕES ----- #
posicoes = []
for i in range(0,len(linhasPretas)) :
    tupla = linhasPretas[i]
    posicoes.append((i, 0, tupla[0]))
    posicoes.append((i, 1, tupla[1]))
    
ligacoes = []
for aresta in listaArestas :
    primeiro = aresta[0]
    segundo = aresta[1]
    posicaoPrimeiro = None
    posicaoSegundo = None
    for el in posicoes :
        if el[2] == primeiro :
            posicaoPrimeiro = (el[0], el[1])
        elif el[2] == segundo :
            posicaoSegundo = (el[0], el[1])
    ligacoes.append([posicaoPrimeiro, posicaoSegundo])
    
#print("\nLigações:")
#for ligacao in ligacoes :
#    if ligacao[0][0] == ligacao[1][0] :
#        print(ligacao[0], "<-->", ligacao[1], "- Linha preta")
#    else :
#        print(ligacao[0], "<-->", ligacao[1], "- Linha cinza")

# ----- CRIAÇÃO DOS CICLOS ----- #
ciclos = []
visitados = []
for ligacao in ligacoes :
    direcoes = []
    ondeComecei = ligacao[0]
    ondeEstou = ligacao[0]
    ondeVou = ligacao[1]

    while ondeEstou not in visitados :
        visitados.append(ondeEstou)
        if ondeEstou[0] == ondeVou[0] : #Linha preta
            if ondeEstou[1] < ondeVou[1] : #Para a direita
                direcoes.append((linhasPretas[ondeEstou[0]][ondeEstou[1]], linhasPretas[ondeVou[0]][ondeVou[1]], "Direita"))
                
            else : #Para a esquerda
                direcoes.append((linhasPretas[ondeEstou[0]][ondeEstou[1]], linhasPretas[ondeVou[0]][ondeVou[1]],"Esquerda"))
                
        for el in ligacoes :
            if el[0] == ondeVou :
                if el[1] != ondeEstou :
                    ondeEstou = ondeVou
                    ondeVou = el[1]
                    break
            elif el[1] == ondeVou :
                if el[0] != ondeEstou :
                    ondeEstou = ondeVou
                    ondeVou = el[0]
                    break
            
    if direcoes != [] :
        ciclos.append(direcoes)

# ----- PRINTS ----- #
print("\nPermutacao:", permutacao)
print("\nExtensão + Linhas Pretas (tuplas):", linhasPretas)
print("\nConexões existentes: ", listaArestas)

print("\nA permutação possui", len(ciclos), "ciclo(s).")

for i in range(0,len(ciclos)) :
    if len(ciclos[i]) == 1 :
        print("\nCiclo", i + 1, "-", ciclos[i], "- Ciclo Bom")
    else :
        cicloBom = False
        primeiraDirecao = None
        for el in ciclos[i] :
            if primeiraDirecao == None :
                primeiraDirecao = el[2]
            elif primeiraDirecao != el[2] :
                cicloBom = True
        if cicloBom is True :
            print("\nCiclo", i + 1, "-", ciclos[i], "- Ciclo Bom")
        else :
            print("\nCiclo", i + 1, "-", ciclos[i], "- Ciclo Ruim")