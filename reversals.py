import sys

def criar_grafo(listaVertices, listaArestas) :
    grafo = {}
    for vertice in listaVertices :
        grafo[vertice] = []
    for aresta in listaArestas :
        grafo[aresta[0]].append(aresta[1])
    return grafo

# ----- PERMUTAÇÃO (LINHA DE COMANDO) ----- #
permutacaoOrg = []
n = len(sys.argv) - 1
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

# ----- CRIAÇÃO DOS VÉRTICES DO GRAFO ----- #
listaVertices = []
for tupla in linhasPretas :
    listaVertices.append(tupla[0])
    listaVertices.append(tupla[1])
print("\nLista de Vertices: ", listaVertices)

# ----- CRIAÇÃO DAS ARESTAS DO GRAFO ----- #
listaArestas = []
for tupla in linhasPretas :
    listaArestas.append((tupla[0], tupla[1]))
    listaArestas.append((tupla[1], tupla[0]))
for i in range(0, len(linhasPretas)) :
        listaArestas.append((i,(i+1) * -1))
        listaArestas.append(((i+1) * -1, i))
#print("Lista de Arestas: ", listaArestas)

grafo = criar_grafo(listaVertices, listaArestas)
print("\nGrafo: ", grafo)
