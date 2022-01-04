import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Preparando o dataset para treino da IA (sem as falhas)
warnings.filterwarnings('ignore')    # Ignora os diversos warnings de gpu do tensorflow
x = pd.read_csv('./dados/sem_zeros.csv')
y = x[['L3']]    # Variavel alvo
x.drop('L3', axis=1, inplace=True)
x.drop('Falha', axis=1, inplace=True)
x.drop('Velocidade ', axis=1, inplace=True)
x.drop('A_inicial', axis=1, inplace=True)
x.drop('Malha', axis=1, inplace=True)

# Normalização dos dados de treino (sem as falhas)
normalizador = MinMaxScaler(feature_range=(0, 1))
# Separando as colunas para normalizar
feats_norm = ['A_S_parede', 'A_S_curva', 'A_S_fundo', 'A_S_total']
x[feats_norm] = normalizador.fit_transform(x[feats_norm])
normalizada = x.copy()

# Separando em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Configuracoes da rede neural
otimizador = Adam()
# Criação do modelo
modelo = Sequential()

# Construimos a rede neural adicionando camadas (.add) sendo o primeiro parametro a quantidade de neuronios
# que a camada oculta terá. A camada de entrada(features) é definida pelo argumento input_dim
# que deve ter o mesmo numero de features do dataset obrigatoriamente.
# kernel_initializer é a escolha dos pesos iniciais (randomico, normal, etc)
# activation é a função de ativação (relu, sigmoid, linear)
modelo.add(Dense(100, input_dim=13, kernel_initializer='glorot_uniform', activation='relu'))
modelo.add(Dropout(0.1))
modelo.add(Dense(40, kernel_initializer='normal', activation='relu', kernel_regularizer='l2'))
modelo.add(Dense(20, kernel_initializer='normal', activation='relu'))
# Camada de saída - Nesse problema de classificação a saída deve ter o número de neurônios com base nas classes
# target existentes no dataset
modelo.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Instanciando a rede - loss é o calculo do custo, otimizador é o metodo de descida do gradiente e metrics
# é métrica de avaliacao de desempenho(acuracia)
# Problema de classificação = binary_crossentropy - regressão - mean_squared_error
modelo.compile(loss='mean_squared_error', optimizer=otimizador, metrics=['ac'])

# Rodando o modelo - Epochs é o numero de iterações do modelo, batch_size a quantidade de amostras do
# dataset usada por iteração, validation_data são os dados para testar(respostas verdadeiras)
# para saber se acertou  ou errou a previsao - não é obrigatorio, pesa no processamento
historico = modelo.fit(x_treino, y_treino, epochs=1000, batch_size=100,
                       validation_data=(x_teste, y_teste), verbose=1)


# ---------------------------------------------------
# Treino finalizado.

# Faz as predições

predicoes = modelo.predict(x)

# Gráfico para análise
acc_treino = historico.history['mse']    # Pega a variável alvo da base de dados
acc_teste = historico.history['val_mse']    # Pega as variaveis alvo que foram preditas por IA
epocas = range(1, len(acc_treino) + 1)    # De 1 até a ultima posicao do vetor treino (+1 por começar em 0)
# print(np.median(historico.history['val_loss']))
plot.plot(epocas, acc_treino, '-b', label='Valor real')
plot.plot(epocas, acc_teste, '-r', label='Valor estimado pela IA')
plot.legend()
plot.xlabel('Ensaios')
plot.ylabel('Valor de L3')
plot.show()
