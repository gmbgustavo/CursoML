import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Nadam


# preparação dos dados
dados = load_boston()
x = pd.DataFrame(dados.data, columns=dados.feature_names)
y = dados.target
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.5)
otimizador = Nadam()


# Criação do modelo
modelo = Sequential()
modelo.add(Dense(30, input_dim=13, kernel_initializer='normal', activation='relu', kernel_regularizer='l2'))
modelo.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_regularizer='l2'))
modelo.add(Dense(20, kernel_initializer='normal', activation='relu', kernel_regularizer='l1'))
modelo.add(Dense(1, kernel_initializer='normal', activation='linear'))    # Problema de regressão ativação linear


# Rodando o modelo
modelo.compile(loss='mean_squared_error', optimizer=otimizador, metrics=['mae'])
historico = modelo.fit(x_treino, y_treino, epochs=1000, batch_size=354, validation_data=(x_teste, y_teste), verbose=1)


# Verificando os resultados
mae_treino = historico.history['mae']
mae_teste = historico.history['val_mae']
epocas = range(1, len(mae_treino)+1)

plot.plot(epocas, mae_treino, '-g', label='Treino')
plot.plot(epocas, mae_teste, '-r', label='Teste')
plt.legend()
plt.xlabel('Epocas')
plt.ylabel('MAE')
plt.show()
