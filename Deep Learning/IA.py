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

def getSigmas(length, pi):
    reversals = numberReversals(length)
    sigmas = []
    for rev in reversals:
        sigmas.append(reversal(rev[0], rev[1], pi))
    return sigmas

def join(pi, sigma):
    state = []
    for el in pi:
        state.append(el)
    for el in sigma:
        state.append(el)
    return state

def markovDecision(choices, intention, length, temperature):
    lower = (100 / length)
    if temperature < lower:
        temperature = lower
    elif temperature > 100:
        temperature = 100
    temperature = temperature / 100
    choices.remove(intention)
    escolha = random.choice(choices)
    result = np.random.choice(a = [1, 2], size = 1, replace = True,
                               p = [temperature, (1 - temperature)])
    if result == 1:
        return intention
    else:
        return escolha

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, nb_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        values = torch.tanh(self.fc2(x))
        return values


# ----------------------------------------------------------------
        
    
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
        
        biggerScore = results[results.index(max(results))]
        intention = sigmas[results.index(max(results))]
        nextState = markovDecision(choices, intention, len(choices), 50)
        arrayMap.append((pi, nextState, biggerScore))
        pi = nextState
        print("Estado com melhor pontuação ->", intention, "\tScore:", '{:.4f}'.format(biggerScore))
        print("Estado escolhido: ", nextState)
        print("---------------------------------------------------------------------")

    print("\nEstado inicial", startState)
    print("Mapa Percorrido: ")
    for el in arrayMap:
        print("-->", el[1], "\tScore:", '{:.4f}'.format(el[2]))




















