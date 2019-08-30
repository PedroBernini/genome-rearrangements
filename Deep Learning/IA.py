import sys
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

def saveNetwork(model):
    torch.save({'state_dict': model.state_dict()}, 'network.pth')
    if os.path.isfile('network.pth'):
        print("Rede salva com sucesso!")
    else:
        print("Erro ao salvar a rede!")
    
def loadNetwork():
    global model
    if os.path.isfile('network.pth'):
        checkpoint = torch.load('network.pth')
        model.load_state_dict(checkpoint['state_dict'])
        print("Rede carregada com sucesso!")
    else:
        print("Erro ao carregar a rede!")

def isIdentity(pi):
    for i in range(0, len(pi)) :
        if i+1 != pi[i] :
            return False
    return True

def numberReversals(n):
    reversals = []
    for i in range(0,n):
        for j in range(i+1, n):
            reversals.append((i,j))
    return reversals

def getSigmas(length, pi):
    reversals = numberReversals(length)
    sigmas = []
    for rev in reversals:
        sigmas.append(reversal(rev[0], rev[1], pi))
    return sigmas

def reversal(i, j, pi):
    resultReversal = list(pi)
    strip = []
    if(i>j):
        temp = i
        i = j
        j = temp
    for k in range(i,j+1):
        strip.append(resultReversal[k]) 
    strip.reverse();
    for k in range(i,j+1):
        resultReversal[k] = strip[k-i]
    return resultReversal

def join(pi, sigma):
    state = []
    for el in pi:
        state.append(el)
    for el in sigma:
        state.append(el)
    return state

# Estrutura da arquitetura da rede neural
class Network(nn.Module):
    def __init__(self, input_size, hidden_size, nb_action): # (quantidade de entradas, número de ações)
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.hidden_size = hidden_size
        
        # Neurônios na camada de entrada -> input_size
        # Neurônios na camada oculta -> hidden_size
        # Neurônios na camada de saída -> nb_action
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, nb_action)
        
    def forward(self, state): # O estado representa as entradas da rede neural
        # Funcão de ativação -> Relu
        x = F.relu(self.fc1(state)) # Aplicação da função de ativação (camada de entrada -> camada oculta)
        q_values = torch.tanh (self.fc2(x)) # Resultado da aplicação da função de ativação
        return q_values # Retorna os valores finais da rede neural
    
startState = []
for el in sys.argv[1].split(",") :
    element = int(el)
    startState.append(element)

input_size = len(startState) * 2
hidden_size = 10
output_size = 1
model = Network(input_size, hidden_size, output_size)

for epoca in range(1):
    pi = startState
    arrayMap = []
    while (isIdentity(pi) == False):
        results = []
        choices = []
        sigmas = getSigmas(len(pi), pi)
        print("\nEstado Atual -> ", pi)
        for sigma in sigmas:
            state = join(pi, sigma)
            state = torch.Tensor(state).float().unsqueeze(0)
            valueExit = model(Variable(state)).item()
            results.append(valueExit)
            choices.append(sigma)
            print("Para:", sigma, "\t", "Saída:", '{:.4f}'.format(valueExit))
        
        nextState = sigmas[results.index(max(results))]
        biggerScore = results[results.index(max(results))]
        choices.append(nextState)
        escolha = random.choice(choices)
        arrayMap.append((pi, escolha, biggerScore))
        pi = escolha
        print("Estado com melhor pontuação ->", nextState, "\tScore:", '{:.4f}'.format(biggerScore))
        print("Estado escolhido: ", escolha)
        
        print("---------------------------------------------------------------------")


    print("\nEstado inicial", startState)
    print("Mapa Percorrido: ")
    for el in arrayMap:
        print("-->", el[1], "\tScore:", '{:.4f}'.format(el[2]))




















