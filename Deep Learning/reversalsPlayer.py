import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class Jogador(object):
    def __init__(self, permutation_size, gamma, temperature, movesLimit, memoryCapacity):
        self.permutation_size = permutation_size
        input_size = self.permutation_size * 2
        hidden_size = 10
        output_size = 1
        self.gamma = gamma
        self.startState = self.randomState()
        self.temperature = temperature
        self.memoryCapacity = memoryCapacity
        self.movesLimit = movesLimit
        self.model = Network(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def setTemperature(self, temperature):
        self.temperature = temperature
        
    def setStartState(self, startState):
        self.startState = startState
        
    def setMemoryCapacity(self, memoryCapacity):
        self.memoryCapacity = memoryCapacity
    
    def setMovesLimit(self, movesLimit) :
        self.movesLimit = movesLimit
    
    def identity(self):
        identity = []
        for i in range(1, self.permutation_size + 1):
            identity.append(i)
        return identity
    
    def isIdentity(self, pi):
        if pi == self.identity():
            return True
        return False
    
    def randomState(self):
        permutation = self.identity()
        state = []
        while(len(permutation) > 0):
            x = random.choice(permutation)
            permutation.remove(x)
            state.append(x)
        return state
    
    def numberReversals(self, n):
        reversals = []
        for i in range(0,n):
            for j in range(i+1, n):
                reversals.append((i,j))
        return reversals
    
    def reversal(self, i, j, pi):
        resultReversal = list(pi)
        strip = []
        if(i>j):
            i, j = j, i
        for k in range(i,j+1):
            strip.append(resultReversal[k]) 
        strip.reverse();
        for k in range(i,j+1):
            resultReversal[k] = strip[k-i]
        return resultReversal

    def getSigmas(self, pi):
        reversals = self.numberReversals(len(pi))
        sigmas = []
        for rev in reversals:
            sigmas.append(self.reversal(rev[0], rev[1], pi))
        return sigmas

    def join(self, pi, sigma):
        state = pi + sigma
        return state
    
    def markovDecision(self, choices, intention, temperature):
        lower = (100 / len(choices))
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
        
    def learn(self, inputs, targets):
        for i in range(0, len(inputs)):
            output = self.model(inputs[i])
            target = targets[i]
            loss = F.smooth_l1_loss(output, target)
#            print("Erro:", loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
    def saveNetwork(self, name):
        path = name + '.pth'
        torch.save({'state_dict': self.model.state_dict()}, path)
        if os.path.isfile(path):
            print("Rede salva com sucesso!")
        else:
            print("Erro ao salvar a rede!")
            
    def loadNetwork(self, name):
        path = name + '.pth'
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("Rede carregada com sucesso!")
        else:
            print("Erro ao carregar a rede!")
    
    def runEpocas(self, length):
        for epoca in range(length):
            print("Época:", epoca + 1, "/", length)
            pi = self.randomState()
            tableScore = []
            moves = 0
            while (moves < self.movesLimit) and (self.isIdentity(pi) == False):
                moves += 1
                results = []
                choices = []
                sigmas = self.getSigmas(pi)
                for sigma in sigmas:
                    state = self.join(pi, sigma)
                    state = torch.Tensor(state).unsqueeze(0)
                    valueExit = self.model(state).item()
                    results.append(valueExit)
                    choices.append(sigma)
                
                biggerScore = max(results)
                intention = sigmas[results.index(biggerScore)]
                nextState = self.markovDecision(choices, intention, self.temperature)
                tableScore.append((pi, nextState, biggerScore))
                if len(tableScore) > self.memoryCapacity:
                    del tableScore[0]
                pi = nextState
            if (self.isIdentity(pi) == True):
                inputs = []
                targets = []
                for i in range(0, len(tableScore)):
                    state = self.join(tableScore[i][0], tableScore[i][1])
                    inputs.append(torch.Tensor(state).float().unsqueeze(0))
                    if i == len(tableScore) - 1:
                        score = 1
                    else:
                        score = (float)(self.gamma * tableScore[i+1][2])
                    targets.append(torch.Tensor([score]).float().unsqueeze(0))
                self.learn(inputs, targets)
                print("Chegou na Identidade =)")
            else:
                print("Falhou em chegar na Identidade =(")
    
    def easyTrain(self, distance, repetitions):
        for repetition in range(repetitions):
            print("Repetição:", repetition + 1, "/", repetitions)
            current = []
            previous = []
            tableScore = []
            for i in range (1, self.permutation_size + 1):
                current.append(i)
            visited = [current]
            score = 1
            move = 0
            while (move < distance):
                if move > 0:
                    score = score * self.gamma
                move += 1
                nexts = []
                sigmas = self.getSigmas(current)
                for el in sigmas:
                    if el not in visited:
                        nexts.append(el)
                if nexts != []:
                    previous = current
                    current = random.choice(nexts)
                    for el in nexts:
                        visited.append(el)
                    tableScore.append((current, previous, score))
            tableScore.reverse()
            inputs = []
            targets = []
            for i in range(0, len(tableScore)):
                state = self.join(tableScore[i][0], tableScore[i][1])
                inputs.append(torch.Tensor(state).float().unsqueeze(0))
                score = tableScore[i][2]
                targets.append(torch.Tensor([score]).float().unsqueeze(0))
                self.learn(inputs, targets)
        
    def goIdentity(self, start):
        pi = start
        tableScore = []
        while (self.isIdentity(pi) == False):
            print("Caminhando...")
            results = []
            choices = []
            sigmas = self.getSigmas(pi)
            for sigma in sigmas:
                state = self.join(pi, sigma)
                state = torch.Tensor(state).unsqueeze(0)
                valueExit = self.model(state).item()
                results.append(valueExit)
                choices.append(sigma)
#                print("Para:", sigma, "\t", "Saída:", '{:.4f}'.format(valueExit))
            biggerScore = max(results)
            intention = sigmas[results.index(biggerScore)]
            nextState = intention
            tableScore.append((pi, nextState, biggerScore))
            pi = nextState
#            print("Estado com melhor pontuação ->", intention, "\tScore:", '{:.4f}'.format(biggerScore))
#            print("Estado escolhido: ", nextState)
#            print("---------------------------------------------------------------------")
        print("\nEstado inicial", start)
        print("Caminho Percorrido: ")
        for el in tableScore:
            print("-->", el[1], "\tScore:", '{:.4f}'.format(el[2]))
            
   
# ----------------------------------------------------------------


# -- Jogador (Estado Inicial, Fator de desconto, Temperatura, Limite de Movimentos, Memoria para Backpropagation) --
#idiota = Jogador(6, 0.9, 50, 100, 100)
#idiota.runEpocas(2000)
#
#idiota.easyTrain(4,2000)

#idiota.goIdentity([2,4,1,3,6,5])


#idiota2 = Jogador(7, 0.9, 50, 700, 200)
#idiota2.loadNetwork("NetworkTest7")
#idiota2.goIdentity([2,4,1,7,6,5,3])

























