import numpy as np
import random
from sklearn.neural_network import MLPRegressor

class Jogador(object):
    def __init__(self, startState, gamma, temperature, memoryCapacity):
        hidden_size = 10
        self.startState = startState
        self.gamma = gamma
        self.temperature = temperature
        self.memoryCapacity = memoryCapacity
        self.model = MLPRegressor(hidden_layer_sizes=(hidden_size,),
                                  activation='tanh',
                                  solver='adam',
                                  learning_rate='constant',
                                  max_iter=1000,
                                  learning_rate_init=0.001)
        self.model.fit([[0, 0, 0, 0, 0, 0]], [0])
        
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
        self.model.fit(inputs, targets)
        
    def saveNetwork(self, name):
        pass
            
    def loadNetwork(self, name):
        pass
    
    def runEpocas(self, length):
        cont = 0
        for epoca in range(length):
            cont += 1
            print("Epoca:", cont, "/", length)
            pi = self.startState
            tableScore = []
            while (self.isIdentity(pi) == False):
                results = []
                choices = []
                sigmas = self.getSigmas(len(pi), pi)
                for sigma in sigmas:
                    state = self.join(pi, sigma)
                    valueExit = self.model.predict([state])
                    results.append(valueExit[0])
                    choices.append(sigma)
                biggerScore = results[results.index(max(results))]
                intention = sigmas[results.index(max(results))]
                nextState = self.markovDecision(choices, intention, len(choices), self.temperature)
                tableScore.append((pi, nextState, biggerScore))
                if len(tableScore) > self.memoryCapacity:
                    del tableScore[0]
                pi = nextState
            inputs = []
            targets = []
            for i in range(0, len(tableScore)):
                state = self.join(tableScore[i][0], tableScore[i][1])
                inputs.append(state)
                if i == len(tableScore) - 1:
                    score = 1
                else:
                    score = (float)(self.gamma * tableScore[i+1][2])
                targets.append(score)
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
                valueExit = self.model.predict([state])
                results.append(valueExit[0])
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

idiota.goIdentity([2,1,3])





