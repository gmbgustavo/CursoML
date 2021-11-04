from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plot


# Desempacotando dataset - esse métido é devido a organização interna desse dataset
(x_teste, y_teste), (x_treino, y_treino) = mnist.load_data()

# Criação da matriz para as classes de saída
y_teste = np_utils.to_categorical(y_teste)
y_treino = np_utils.to_categorical(y_treino)

# Modelando os dados de entrada - transformando a matriz da imagem e uma fila (unidimensional)
x_teste = x_teste.reshape(60000, 784)    # 28 x 28 linha x coluna
x_treino = x_treino.reshape(10000, 784)
# Normalização da escala de pixels de 0 a 255 para entre 0 e 1
x_teste = x_teste.astype('float32') / 255
x_treino = x_treino.astype('float32') / 255

# Criando o modelo - 784 Entradas, cada pixel. 3 camadas
modelo = Sequential()
modelo.add(Dense(50, input_dim=784, kernel_initializer='normal', activation='relu', kernel_regularizer='l2'))
modelo.add(Dropout(0.2))    # Camada de dropout com 20% dos neuronios desligados - Evita overfitting
modelo.add(Dense(50, kernel_initializer='normal', activation='relu', kernel_regularizer='l2'))
modelo.add(Dropout(0.33))
modelo.add(Dense(50, kernel_initializer='normal', activation='relu', kernel_regularizer='l2'))
# 10 saídas, representando os 10 digitos possiveis
modelo.add(Dense(10, kernel_initializer='normal', activation='softmax'))    # Camada de saida
otimizador = Adam(amsgrad=True)

# Rodando o modelo
# 1 - Compilar
modelo.compile(loss='categorical_crossentropy', optimizer=otimizador, metrics=['acc'])
# 2 - Treinar(fit) - O Fit retorna um objeto History que pode ser analisado caso deseje
#     Lembrando que o batch_size é quantos dados ele vai manipular por vez, se são 60.000 dados e o
#     batch é 100, ele vai fazer 600 rodadas até usar todos os dados.
historico = modelo.fit(x_treino, y_treino, epochs=1000, batch_size=1000, validation_data=(x_teste, y_teste), verbose=1)

# Gráfico para análise
acc_treino = historico.history['acc']
acc_teste = historico.history['val_acc']
epocas = range(1, len(acc_treino) + 1)    # De 1 até a ultima posicao do vetor treino (+1 por começar em 0)
plot.plot(epocas, acc_treino, '-b', label='Precisão do treino')
plot.plot(epocas, acc_teste, '-r', label='Precisão do teste')
plot.legend()
plot.xlabel('Épocas')
plot.ylabel('Precisão')
plot.show()
