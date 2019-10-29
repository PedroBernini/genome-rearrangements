import numpy as np
import random
import os
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
import pdb

class Jogador(object):
    def __init__(self, permutation_size, hidden_size = 400, gamma = 0.9, temperature = 50, movesLimit = 20, memoryCapacity = 30):
        self.permutation_size = permutation_size
        self.startState = self.randomState()
        self.gamma = gamma
        self.temperature = temperature
        self.memoryCapacity = memoryCapacity
        self.movesLimit = movesLimit
        self.bkpReversals = False
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
        
    def setBkpReversals(self, bkpReversals) :
        self.bkpReversals = bkpReversals

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
    
    def isAdjacent(self, a, b) :
        if (abs(a - b) == 1) :
            return True
        else :
            return False
    
    def getBreakPoints(self, permutation):
        bkpMap = []
        if(permutation[0]!= 1):
            bkpMap.append(0)
            
        if(permutation[len(permutation) - 1] != len(permutation)):
            bkpMap.append(len(permutation))
            
        for i in range(0, len(permutation)-1) :
            if (self.isAdjacent(permutation[i], permutation[i+1]) is False):
                bkpMap.append(i+1)
                
        return bkpMap
    
    def getNumBkp(self, pi) :
        numBkp = 0
        if(pi[0]!= 1):
            numBkp += 1
        if(pi[len(pi) - 1] != len(pi)):
            numBkp += 1
        for i in range(0, len(pi)-1) :
            if (self.isAdjacent(pi[i], pi[i+1]) is False):
                numBkp += 1
        return numBkp
    
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
        if self.bkpReversals == False:
            reversals = self.numberReversals(len(pi))
            sigmas = []
            for rev in reversals:
                sigmas.append(self.reversal(rev[0], rev[1], pi))
            return sigmas
        else:
            reversals = self.numberReversals(len(pi))
            sigmas = []
            breakpoints = self.getBreakPoints(pi)
            for rev in reversals:
                if rev[0] in breakpoints and (rev[1] + 1) in breakpoints:
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
        
        if choices == []:
            return intention
        else:
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
                    
    def runEpocas(self, length, oneFit = False):
        if not oneFit:
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
        else:
            fits = []
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
                    score = 1
                    tableScore.reverse()
                    for i in range(0, len(tableScore)):
                        state = self.join(tableScore[i][0], tableScore[i][1])
                        inputs.append(state)
                        if i == 0:
                            score = 1
                        else:
                            score = score * self.gamma
                        targets.append(score)
                    fits.append((inputs, targets))
                    print("FIT SUCESSO =)")
                else:
                    print("FIT FRACASSO =(")
                    
            print("\nTotal de Fits Sucedidos:", len(fits))
            print("Treinando...")
            for fit in fits: 
                try:
                    self.model.fit(fit[0], fit[1])
                except:
                    pass
                
    def bkpTrain(self, repetitions):
        inputs = []
        targets = []
        for repetition in range(repetitions):
            print("Rodada:", repetition + 1, "/", repetitions)
            pi = self.randomState()
            bkpPi = self.getNumBkp(pi)
            sigmas = self.getSigmas(pi)
            score = None
            
            for sigma in sigmas:
                bkpSigma = self.getNumBkp(sigma)
                diference = bkpPi - bkpSigma
                if diference == 0:
                    score = 0
                elif diference >= 1:
                    score = 1
                else:
                    score = -1
                entry = self.join(pi, sigma)
                inputs.append(entry)
                targets.append(score)
        print("Treinando...")
        self.model.fit(inputs, targets)
            
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
        movimentos = 0
        while (self.isIdentity(pi) == False and movimentos < 2*self.permutation_size):
            movimentos += 1
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
        if movimentos < 2*self.permutation_size:
            print("\nEstado inicial", start)
            print("Caminho Percorrido: ")
            for el in tableScore:
                print("-->", el[1], "\tScore:", '{:.4f}'.format(el[2]))
            print("Total de reversões:", qtdReversoes)
        else:
            print("Rede não convergiu!")
# ----------------------------------------------------------------

idiota = Jogador(20)
idiota.setBkpReversals(True)
idiota.bkpTrain(100000)
idiota.goIdentity(idiota.randomState())


