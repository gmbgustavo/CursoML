import warnings
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Preparando o dataset
warnings.filterwarnings('ignore')    # Ignora os diversos warnings de gpu do tensorflow
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)
y = np_utils.to_categorical(y)    # One hot encode para as 3 classes de saída
# Separando em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Configuracoes da rede neural
otimizador = SGD()


# Criação do modelo
modelo = Sequential()

# Construimos a rede neural adicionando camadas (.add) sendo o primeiro parametro a quantidade de neuronios
# que a camada oculta terá (10 no caso abaixo). A camada de entrada(features) é definida pelo argumento input_dim
# que deve ter o mesmo numero de features do datase obrigatoriamente.
# kernel_initializer é a escolha dos pesos iniciais (randomico, normal, etc)
# activation é a função de ativação (relu, sigmoid, linear)
modelo.add(Dense(10, input_dim=4, kernel_initializer='normal', activation='relu'))

# Camada de saída - Nesse problema de classificação a saída deve ter o número de neurônios com base nas classes
# target existentes no dataset
modelo.add(Dense(3, kernel_initializer='normal', activation='softmax'))

# Instanciando a rede - loss é o calculo do custo, otimizador é o metodo de descida do gradiente e metrics
# é métrica de avaliacao de desempenho(acuracia)
modelo.compile(loss='categorical_crossentropy', optimizer=otimizador, metrics=['acc'])

# Rodando o modelo - Epochs é o numero de iterações do modelo, batch_size a quantidade de amostras do
# dataset usada por iteração, validation_data são os dados para testar(respostas verdadeiras)
# para saber se acertou  ou errou a previsao - não é obrigatorio, pesa no processamento
modelo.fit(x_treino, y_treino, epochs=700, batch_size=50, validation_data=(x_teste, y_teste), verbose=1)

predicoes = modelo.predict(x_teste)
np.set_printoptions(formatter={'float': lambda ft: '{0:0.2f}'.format(ft)})
print(predicoes)
