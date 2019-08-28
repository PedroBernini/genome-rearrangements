import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

def isIdentity(permutation) :
    for i in range(0, len(permutation)) :
        if i+1 != permutation[i] :
            return False
    return True

def numberReversals(n) :
    reversals = []
    for i in range(0,n) :
        for j in range(i+1, n) :
            reversals.append((i,j))
    return reversals

def getSigmas(length, permutation) :
    reversals = numberReversals(length)
    sigmas = []
    for rev in reversals:
        sigmas.append(reversal(rev[0], rev[1], permutation))
    return sigmas

def reversal(i, j, permutation) :
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

def join(pi, sigma) :
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
        
        # Quantidade de neurônios na camada de entrada: 3 (input_size)
        # Quantidade de neurônios na camada oculta: 8 (hidden_size)
        # Quantidade de neurônios na camada de saída: 3 (nb_action)
        self.fc1 = nn.Linear(input_size, hidden_size) # Ligação da camada de entrada até a camada oculta
        self.fc2 = nn.Linear(hidden_size, nb_action)
        
    def forward(self, state): # O estado representa as entradas da rede neural
        # Funcão de ativação -> Relu
        x = F.relu(self.fc1(state)) # Aplicação da função de ativação (camada de entrada -> camada oculta)
        q_values = torch.sigmoid(self.fc2(x)) # Resultado da aplicação da função de ativação
        return q_values # Retorna os valores finais da rede neural
    
permutation = [2,3,1]

input_size = len(permutation) * 2
hidden_size = 8
output_size = 1
model = Network(input_size, hidden_size, output_size)

arrayMap = []

while (isIdentity(permutation) == False):
    results = []
    sigmas = getSigmas(len(permutation), permutation)
    print("\nEstado Atual -> ", permutation)
    for sigma in sigmas:
        state = join(permutation, sigma)
        state = torch.Tensor(state).float().unsqueeze(0)
        valueExit = model(Variable(state)).item()
        results.append(valueExit)
        print("Estado Adjacente:", sigma, "\t", "Valor da Saída:", valueExit, "\t[", sigmas.index(sigma), "]" )
    
    nextState = sigmas[results.index(max(results))]
    biggerScore = results[results.index(max(results))]
    arrayMap.append((permutation, nextState, biggerScore))
    print("Estado com melhor pontuação ->", nextState, "\tScore:", biggerScore)
    permutation = nextState
    print("---------------------------------------------------------------------")
    
print("\nMapa Percorrido: ")
for el in arrayMap:
    print(el[0], "-->", el[1], "\tScore:", el[2])
























