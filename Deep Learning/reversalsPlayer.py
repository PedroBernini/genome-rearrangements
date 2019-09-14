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
    def __init__(self, startState, gamma, temperature, movesLimit, memoryCapacity):
        self.permutation_size = len(startState)
        input_size = self.permutation_size * 2
        hidden_size = 10
        output_size = 1
        self.startState = startState
        self.gamma = gamma
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
        self.memoryCapacity = movesLimit
    
    def isIdentity(self, pi):
        for i in range(0, len(pi)) :
            if i+1 != pi[i] :
                return False
        return True
    
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
            print("Época:", epoca, "/", length)
            pi = self.startState
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
#            print("TableScore: ")
#            for el in tableScore:
#                print(el)
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
                print("Chegou na Identidade!")
            else:
                print("Falhou em chegar na Identidade!")
    
#    def easyTrain(self, distance, repetitions, epocas):
#        for repetition in range(repetitions):
#            start = []
#            for i in range (1, self.permutation_size + 1):
#                start.append(i)
#            visited = [start]
#            move = 0
#            while (move < distance):
#                move += 1
#                nexts = []
#                sigmas = self.getSigmas(start)
#                for el in sigmas:
#                    if el not in visited:
#                        nexts.append(el)
#                if nexts != []:
#                    start = random.choice(nexts)
#                for el in nexts:
#                    visited.append(el)
#            self.setStartState(start)
#            self.runEpocas(epocas)
                
    def easyTrain(self, repetitions):
        for repetition in range(repetitions):
            print("Repetição:", repetition, "/", repetitions)
            identity = []
            for i in range (1, self.permutation_size + 1):
                identity.append(i)
            nexts = []
            sigmas = self.getSigmas(identity)
            for sigma in sigmas:
                nexts.append(sigma)
            start = random.choice(nexts)
            self.setStartState(start)
#            print("Começei aki", start)
            pi = self.startState
            tableScore = []
            state = self.join(start, identity)
            state = torch.Tensor(state).unsqueeze(0)
            valueExit = self.model(state).item()
            tableScore.append((start, identity, valueExit))
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
#                    print("Estou indo para:", nextState)
                
                state = self.join(nextState, pi)
#                    print("Como coloco:", state)
                state = torch.Tensor(state).unsqueeze(0)
                valueExit = self.model(state).item()
#                    print("Meu valor de ida:", biggerScore)
#                    print("Meu valor de volta:", valueExit)
#                    print("\n")
                tableScore.append((nextState, pi, valueExit))
                if len(tableScore) > self.memoryCapacity:
                    del tableScore[0]
                pi = nextState
#                print("TableScore: ")
#                for el in tableScore:
#                    print(el)
            tableScore.reverse()
#                print("TableScore (REVERSE): ")
#                for el in tableScore:
#                    print(el)
            
            if (self.isIdentity(pi) == False):
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
#                print("Inputs:")
#                for el in inputs:
#                    print(el)
#                print("Targets:")
#                for el in targets:
#                    print(el)
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
idiota = Jogador([5,3,1,7,6,2,4], 0.9, 50, 200, 100)
idiota.easyTrain(100)
idiota.runEpocas(100)
#idiota.goIdentity([2,4,1,7,6,5,3])

#idiota.saveNetwork("idiotaNetwork")

#idiota.goIdentity([3,1,2])

#idiota2 = Jogador([7,3,5,1,6,2,4], 0.9, 50, 1000, 100)
#idiota2.loadNetwork("network7")
#idiota2.goIdentity([3,1,5,2,4,7,6])
