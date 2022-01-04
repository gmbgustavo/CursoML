from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plot
import numpy as np


# Desempacotando dataset - esse métido é devido a organização interna desse dataset
(x_treino, y_treino), (x_teste, y_teste) = cifar10.load_data()
# Aplicando o one hot encode
y_treino = to_categorical(y_treino)
y_teste = to_categorical(y_teste)
# x_treino = (50000, 32, 32, 3) 50 mil imagens, 32x32 pixels padrão RGB(3)
# y_treino = (50000, 1) Depois do to_categorical ele tera 10 colunas(pois tem 10 classes)

'''
# Para visualizar algumas imagens
for i in range(20):
    plot.subplot(5, 4, i+1)    # Linhas, colunas, indice - começa em 1, não tem 0
    plot.imshow(x_treino[i])
plot.show()
'''

# Normalizando a numeração da escala de cores de 0-255 para 0-1
x_treino = x_treino.astype('float32')
x_teste = x_teste.astype('float32')
x_treino = x_treino / 255.0
x_teste = x_teste / 255.0


# Criando a CNN
modelo = Sequential()
# Camada de entrada - 32 filtros de imagem, filtro de escaneamento da matriz(kernel) 5x5
# funcao de ativacao relu e formato a passar para o feature map 28x28 escala de cinza(1)
modelo.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
modelo.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
# Camada de pooling - tamanho de cada layer do pool, strides(tamanho do passo do filtro, nesse caso o mesmo do pool)
# padding é a adição de borda valid = não / same = borda
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
modelo.add(Dropout(0.3))
# Segunda camada e seu pooling
modelo.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
modelo.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
modelo.add(Dropout(0.3))
# Terceira camada
modelo.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
modelo.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
# Transforma a matriz em uma linha única para poder ligar a uma camada densa de neuronios
# a camada Dense só tem 1 dimensão
modelo.add(Flatten())
# quarta camada - Densa - Inicialização dos pesos Xavier(glotor_uniform), no conv2d ela é usada por padrão
modelo.add(Dense(130, kernel_initializer='glorot_uniform', activation='relu'))
modelo.add(Dropout(0.3))
# Camada de saida representando cada digito
modelo.add(Dense(10, kernel_initializer='glorot_uniform', activation='softmax'))

# Configurando o data augumentation
aug_data = ImageDataGenerator(width_shift_range=0.09, height_shift_range=0.09, horizontal_flip=True)
treino_aug = aug_data.flow(x_treino, y_treino, batch_size=200)

# Definindo o otimizador e função de custo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Treinando o modelo
n_passadas = int(x_treino.shape[0] / 200)
historico = modelo.fit(treino_aug, steps_per_epoch=n_passadas, epochs=20, validation_data=(x_teste, y_teste),
                       verbose=1)

modelo.save('../dados/cifar10.h5')


'''
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
'''

'''
imagem = x_teste[155]
plot.imshow(imagem)
plot.show()
imagem = imagem.astype('float32')
imagem = imagem / 255.0
imagem = np.expand_dims(imagem, axis=0)
resultado = modelo.predict_classes(imagem)
print(resultado[0])
'''
