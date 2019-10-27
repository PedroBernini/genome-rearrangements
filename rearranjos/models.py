# encoding: utf-8

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD


from reinforcement import Player

#################################################### 
## Criei este módulo para poder colocar todos os modelos de
## aprendizagem de máquina que podem ser usados. É interessante
## colocar todos aqui e depois colocar uma listagem do que foi usado
## no arquivo train ou no arquivo predict.
####################################################

def generate_ann_model(input_dim = 9, num_hidden_nodes = [18]) :    
    model = Sequential()
    model.add(Dense(num_hidden_nodes[0], input_dim=input_dim, activation='relu', kernel_initializer='normal'))

    for hidden_node in num_hidden_nodes[1:] :
        model.add(Dense(hidden_node, activation='relu',kernel_initializer='normal'))
        model.add(Dropout(0.1))
    model.add(Dense(1, activation='tanh',kernel_initializer='normal'))
    #model.summary()

    learning_rate = 0.001
    momentum      = 0.8
    sgd           = SGD(lr=learning_rate, momentum=momentum,nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def select_player(i, op, permutation_size) :
    model = None

    if i == 0:
        model = generate_ann_model(input_dim = permutation_size, 
                                        num_hidden_nodes = [6,3])
    elif i == 1 :
        model = generate_ann_model(input_dim = permutation_size, 
                                        num_hidden_nodes = [9])
    elif i == 2 :
        model = generate_ann_model(input_dim = permutation_size, 
                                        num_hidden_nodes = [18,9])    
    elif i == 3 :
        model = generate_ann_model(input_dim = permutation_size, 
                                        num_hidden_nodes = [27])
    elif i == 4 :
        model = generate_ann_model(input_dim = permutation_size, 
                                        num_hidden_nodes = [450,150])    
    elif i == 5 :
        model = generate_ann_model(input_dim = permutation_size, 
                                        num_hidden_nodes = [600])    
        
    else :
        raise ValueError("Unknown model: " + i)     
    
    return Player(model, op)
