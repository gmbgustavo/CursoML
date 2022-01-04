from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Desempacotando dataset - esse métido é devido a organização interna desse dataset
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()
# Aplicando o one hot encode
y_treino = to_categorical(y_treino)
y_teste = to_categorical(y_teste)

# O modelo Conv2D espera que na entrada seja informado 3 parametros: largura, altura e padrão de cores
x_treino = x_treino.reshape(60000, 28, 28, 1)    # 1 - Escala de cinza
x_teste = x_teste.reshape(10000, 28, 28, 1)

# Criando a CNN
modelo = Sequential()
# Camada de entrada - 32 filtros de imagem, filtro de escaneamento da matriz(kernel) 5x5
# funcao de ativacao relu e formato a passar para o feature map 28x28 escala de cinza(1)
modelo.add(Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
# Camada de pooling - tamanho de cada layer do pool, strides(tamanho do passo do filtro, nesse caso o mesmo do pool)
# padding é a adição de borda valid = não / same = borda
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
# Segunda camada e seu pooling
modelo.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
# Transforma a matriz em uma linha única para poder ligar a uma camada densa de neuronios
# a camada Dense só tem 1 dimensão
modelo.add(Flatten())
# terceira camada - Densa
modelo.add(Dense(80, kernel_initializer='normal', activation='relu'))
modelo.add(Dropout(0.2))
# Camada de saida representando cada digito
modelo.add(Dense(10, kernel_initializer='normal', activation='softmax'))

# Definindo o otimizador e função de custo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
modelo.fit(x_treino, y_treino, batch_size=250, epochs=10, validation_data=(x_teste, y_teste), verbose=1)


