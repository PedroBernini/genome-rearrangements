import sys

#def criar_grafo(listaVertices, listaArestas) :
#    grafo = {}
#    for vertice in listaVertices :
#        grafo[vertice] = []
#    for aresta in listaArestas :
#        grafo[aresta[0]].append(aresta[1])
#    return grafo


#grafo = criar_grafo(listaVertices, listaArestas)
#print("\nGrafo: ", grafo)

# ----- PERMUTAÇÃO (LINHA DE COMANDO) ----- #
permutacaoOrg = []
n = len(sys.argv)
ultimo = None

for el in range(1, n) :
    permutacaoOrg.append(eval(sys.argv[el]))
    if abs(eval(sys.argv[el])) == n - 1 :
        ultimo = (abs(eval(sys.argv[el])) + 1) * -1

print("\nPermutacao Original:", permutacaoOrg)

# ----- CRIAÇÃO DAS LINHAS PRETAS (TUPLAS) ----- #
linhasPretas = []
for i in range(0, len(permutacaoOrg)) :
    if i == 0 :
        linhasPretas.append((0, permutacaoOrg[i] * -1))
        linhasPretas.append((permutacaoOrg[i], permutacaoOrg[i + 1] * -1))
    elif i == len(permutacaoOrg) - 1 :
        linhasPretas.append((permutacaoOrg[i], ultimo))
    else :
        linhasPretas.append((permutacaoOrg[i], permutacaoOrg[i + 1] * -1))
print("\nLinhasPretas:", linhasPretas)

## ----- CRIAÇÃO DOS VÉRTICES DO GRAFO ----- #
#listaVertices = []
#for tupla in linhasPretas :
#    listaVertices.append(tupla[0])
#    listaVertices.append(tupla[1])
##print("\nLista de Vertices: ", listaVertices)

# ----- CRIAÇÃO DAS ARESTAS ----- #
listaArestas = []
for tupla in linhasPretas :
    listaArestas.append((tupla[0], tupla[1]))
    listaArestas.append((tupla[1], tupla[0]))
for i in range(0, len(linhasPretas)) :
        listaArestas.append((i,(i+1) * -1))
        listaArestas.append(((i+1) * -1, i))
#print("Lista de Arestas: ", listaArestas)

posicoes = []
for i in range(0,len(linhasPretas)) :
    tupla = linhasPretas[i]
    posicoes.append((i, 0, tupla[0]))
    posicoes.append((i, 1, tupla[1]))
    
print("\nPosicoes:", posicoes)
    
print("\nArestas: ", listaArestas)

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

print("\nLigações:")
for ligacao in ligacoes :
    if ligacao[0][0] == ligacao[1][0] :
        print(ligacao[0], "<-->", ligacao[1], "- Linha preta")
    else :
        print(ligacao[0], "<-->", ligacao[1], "- Linha cinza")

direcoes = []
for ligacao in ligacoes :
    ondeComecei = ligacao[0]
    ondeEstou = ligacao[0]
    ondeVou = ligacao[1]

    while ondeEstou != ondeComecei :
        if ondeEstou[0] == ondeVou[0] : #Linha preta
            if ondeEstou[1] < ondeVou[1] : #Para a direita
                direcoes.append("Direita")
                
            else : #Para a esquerda
                direcoes.append("Esquerda")
                
        for el in ligacoes :
            if el[0] == ondeVou and el[1] != ondeEstou :
                ondeEstou = ondeVou
                ondeVou = el[1]
                break

















