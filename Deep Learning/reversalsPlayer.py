import sys
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
    def __init__(self, startState, gamma, temperature, memoryCapacity):
        input_size = len(startState) * 2
        hidden_size = 10
        output_size = 1
        self.startState = startState
        self.gamma = gamma
        self.temperature = temperature
        self.memoryCapacity = memoryCapacity
        self.model = Network(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def setTemperature(self, temperature):
        self.temperature = temperature
    
    def setStartState(self, startState):
        self.startState = startState
    
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
            temp = i
            i = j
            j = temp
        for k in range(i,j+1):
            strip.append(resultReversal[k]) 
        strip.reverse();
        for k in range(i,j+1):
            resultReversal[k] = strip[k-i]
        return resultReversal

    def getSigmas(self, length, pi):
        reversals = self.numberReversals(length)
        sigmas = []
        for rev in reversals:
            sigmas.append(self.reversal(rev[0], rev[1], pi))
        return sigmas

    def join(self, pi, sigma):
        state = []
        for el in pi:
            state.append(el)
        for el in sigma:
            state.append(el)
        return state
    
    def markovDecision(self, choices, intention, length, temperature):
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
        
    def learn(self, inputs, targets):
        for i in range(0, len(inputs)):
            output = self.model(inputs[i])
            target = targets[i]
            loss = F.smooth_l1_loss(output, target)
            print("Erro:", loss.item())
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
            pi = self.startState
            tableScore = []
            while (self.isIdentity(pi) == False):
                results = []
                choices = []
                sigmas = self.getSigmas(len(pi), pi)
#                print("\nEstado Atual -> ", pi)
                for sigma in sigmas:
                    state = self.join(pi, sigma)
                    state = torch.Tensor(state).unsqueeze(0)
                    valueExit = self.model(state).item()
                    results.append(valueExit)
                    choices.append(sigma)
#                    print("Para:", sigma, "\t", "Saída:", '{:.4f}'.format(valueExit))
                
                biggerScore = results[results.index(max(results))]
                intention = sigmas[results.index(max(results))]
                nextState = self.markovDecision(choices, intention, len(choices), self.temperature)
                tableScore.append((pi, nextState, biggerScore))
                if len(tableScore) > self.memoryCapacity:
                    del tableScore[0]
                pi = nextState
#                print("Estado com melhor pontuação ->", intention, "\tScore:", '{:.4f}'.format(biggerScore))
#                print("Estado escolhido: ", nextState)
#                print("---------------------------------------------------------------------")
        
            print("\nEstado inicial", self.startState)
            print("Caminho Percorrido: ")
            for el in tableScore:
                print("-->", el[1], "\tScore:", '{:.4f}'.format(el[2]))
                
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
            
    def goIdentity(self, start):
        pi = start
        tableScore = []
        while (self.isIdentity(pi) == False):
            print("Caminhando...")
            results = []
            choices = []
            sigmas = self.getSigmas(len(pi), pi)
            for sigma in sigmas:
                state = self.join(pi, sigma)
                state = torch.Tensor(state).unsqueeze(0)
                valueExit = self.model(state).item()
                results.append(valueExit)
                choices.append(sigma)
            biggerScore = results[results.index(max(results))]
            intention = sigmas[results.index(max(results))]
            nextState = intention
            tableScore.append((pi, nextState, biggerScore))
            pi = nextState
        print("\nEstado inicial", start)
        print("Caminho Percorrido: ")
        for el in tableScore:
            print("-->", el[1], "\tScore:", '{:.4f}'.format(el[2]))
            
# ----------------------------------------------------------------
        
startState = [3,1,2]

# -- Jogador (Estado Inicial, Fator de desconto, Temperatura, Memoria para Backpropagation) --
idiota = Jogador(startState, 0.9, 50, 100)
idiota.runEpocas(200)

idiota.setStartState([2,3,1])
idiota.runEpocas(200)

idiota.setStartState([3,1,2])
idiota.setTemperature(80)
idiota.runEpocas(200)

idiota.saveNetwork("idiotaNetwork")

idiota.goIdentity([2,1,3])


#idiota2 = Jogador(startState, 0.9, 50, 100)
#idiota2.loadNetwork("idiotaNetwork")

