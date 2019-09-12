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
#        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.permutation_size = len(startState)
        input_size = self.permutation_size * 2
        hidden_size = 10
        output_size = 1
        self.startState = startState
        self.gamma = gamma
        self.temperature = temperature
        self.memoryCapacity = memoryCapacity
        self.model = Network(input_size, hidden_size, output_size)
#        self.model.to(device)
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
        cont = 0
        for epoca in range(length):
            cont += 1
            print("Época:", cont, "/", length)
            pi = self.startState
            tableScore = []
            while (self.isIdentity(pi) == False):
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
    
#    def easyTrain(self):
#        start = []
#        for i in range (1, self.permutation_size + 1):
#            start.append(i)
#        sigmas = self.getSigmas(start)
            
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
            biggerScore = max(results)
            intention = sigmas[results.index(biggerScore)]
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
idiota.runEpocas(400)

idiota.setStartState([3,1,2])
idiota.setTemperature(80)
idiota.runEpocas(600)

idiota.saveNetwork("idiotaNetwork")

idiota.goIdentity([3,2,1])
#
#
#idiota2 = Jogador(startState, 0.9, 50, 100)
#idiota2.loadNetwork("idiotaNetwork")

