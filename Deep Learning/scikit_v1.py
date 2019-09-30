import numpy as np
import random
import os
from sklearn.neural_network import MLPRegressor
from joblib import dump, load

class Jogador(object):
    def __init__(self, permutation_size, hidden_size = 100, gamma = 0.9, temperature = 50, movesLimit = 10, memoryCapacity = 30):
        self.permutation_size = permutation_size
        self.startState = self.randomState()
        self.gamma = gamma
        self.temperature = temperature
        self.memoryCapacity = memoryCapacity
        self.movesLimit = movesLimit
        self.model = MLPRegressor(hidden_layer_sizes = (hidden_size,),
                                  activation ='relu',
                                  solver ='adam',
                                  warm_start = True,
                                  learning_rate_init = 0.001)
        self.initialFit()

    def initialFit(self):
        entry = list(range(0, self.permutation_size * 2))
        self.model.fit([entry], [0])
        
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
        for i in range(0, len(pi)) :
            if i+1 != pi[i] :
                return False
        return True
    
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
            temp = i
            i = j
            j = temp
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
        
    def saveNetwork(self, path):
        dump(self.model, path) 
        if os.path.isfile(path):
            print("Rede salva com sucesso!")
        else:
            print("Erro ao salvar a rede!")
            
    def loadNetwork(self, path):
        if os.path.isfile(path):
            self.model = load(path)
            print("Rede carregada com sucesso!")
        else:
            print("Erro ao carregar a rede!")
    
    def runEpocas(self, length):
        for epoca in range(length):
            print("Epoca:", epoca + 1, "/", length)
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
                    valueExit = self.model.predict([state])
                    results.append(valueExit[0])
                    choices.append(sigma)
                biggerScore = max(results)
                intention = sigmas[results.index(max(results))]
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
                    inputs.append(state)
                    if i == len(tableScore) - 1:
                        score = 1
                    else:
                        score = (float)(self.gamma * tableScore[i+1][2])
                    targets.append(score)
                try:
                    self.model.fit(inputs, targets)
                except:
                    pass
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
                inputs.append(state)
                score = tableScore[i][2]
                targets.append(score)
                self.model.fit(inputs, targets)
            
    def goIdentity(self, start):
        pi = start
        tableScore = []
        qtdReversoes = 0
        while (self.isIdentity(pi) == False):
            print("Caminhando...")
            qtdReversoes += 1
            results = []
            choices = []
            sigmas = self.getSigmas(pi)
            for sigma in sigmas:
                state = self.join(pi, sigma)
                valueExit = self.model.predict([state])
                results.append(valueExit[0])
                choices.append(sigma)
            biggerScore = max(results)
            intention = sigmas[results.index(max(results))]
            nextState = intention
            tableScore.append((pi, nextState, biggerScore))
            pi = nextState
        print("\nEstado inicial", start)
        print("Caminho Percorrido: ")
        for el in tableScore:
            print("-->", el[1], "\tScore:", '{:.4f}'.format(el[2]))
        print("Total de reversões:", qtdReversoes)
# ----------------------------------------------------------------

idiota = Jogador(10)

idiota.easyTrain(1,1000000)
idiota.saveNetwork('network_v1_level1.joblib')

idiota.easyTrain(2,500000)
idiota.saveNetwork('network_v1_level2.joblib')

idiota.easyTrain(3,300000)
idiota.saveNetwork('network_v1_level3.joblib')

idiota.easyTrain(4,250000)
idiota.saveNetwork('network_v1_level4.joblib')

idiota.easyTrain(5,180000)
idiota.saveNetwork('network_v1_level5.joblib')

idiota.easyTrain(6,140000)
idiota.saveNetwork('network_v1_level6.joblib')

idiota.easyTrain(7,100000)
idiota.saveNetwork('network_v1_level7.joblib')

idiota.easyTrain(8,80000)
idiota.saveNetwork('network_v1_level8.joblib')

idiota.easyTrain(9,50000)
idiota.saveNetwork('network_v1_level9.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level10.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level11.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level12.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level13.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level14.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level15.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level16.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level17.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level18.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level19.joblib')

idiota.runEpocas(50000)
idiota.saveNetwork('network_v1_level20.joblib')




