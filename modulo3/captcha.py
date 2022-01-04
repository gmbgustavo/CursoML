"""
Rede neural para detecção de captcha.
Cada imagem com o desafio captcha possui 5 caracteres que podem ser números ou letras
São 1070 imagens com o formato (50, 200, 3)
Quebraremos em 5 partes para fazer a identificação por caractere e não a imagem completa
Cada quebra ficará com o shape (5350, 38, 20)
"""

import warnings
import os
import numpy as np
import string
from cv2 import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

PASTA_DADOS = '../dados/captcha/'
ARQUIVOS_DADOS = os.listdir(PASTA_DADOS)    # 1070 arquivos - shape original da imagem (50, 200, 3)
warnings.filterwarnings('ignore')


# Carrega as imagens numa lista a partir do nome do arquivo e o caminho relativo
def carrega_imagens(lista: list):
    """
    :param lista: lista com o nome dos arquivos a serem carregados
    :return: uma lista com as imagens carregadas já convertidas para escala de cinza
    """
    lista_caminhos = []
    for i in lista:
        caminho = PASTA_DADOS + i    # Concatena o nome da pasta com o nome do arquivo
        img = cv2.imread(caminho)
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lista_caminhos.append(img_cinza)
    return lista_caminhos


def quebra_imagem(lista_img):
    """
    A dimensão (1070, 50, 200) vai ser transformada em (5350, 38, 20)
    Foi quebrada em 5 partes horizontais de 20px e removido uma borda superior e inferior de 6px cada
    :param lista_img: a lista com imagens
    :return: uma lista com a imagem quebrada em caracteres (aproximado)
    """
    x_novo = np.zeros((len(lista_img) * 5, 38, 20))
    ix, iy, larg, alt = 30, 12, 20, 50    # ponto inicial x e y, largura e ponto final y
    for k in range(len(lista_img)):
        px = ix
        for q in range(5):
            x_novo[k*5+q] = lista_img[k][iy:alt, px:px+larg]
            px += larg
    return x_novo


# Programa principal
imagens = carrega_imagens(ARQUIVOS_DADOS)
dataset = quebra_imagem(imagens)


# Normalizar as imagens
x_inicial = np.zeros((dataset.shape[0], dataset.shape[1], dataset.shape[2], 1))
for i in range(dataset.shape[0]):    # normaliza os pixels para entre 0 e 1
    dataset[i] = dataset[i] / 255
    x_inicial[i] = np.reshape(dataset[i], (dataset.shape[1], dataset.shape[2], 1))
# O Shape é uma tupla (5350, 38, 20) passamos cada posição do shape mais uma posição da cor

# Definindo o target
y_alvo = ARQUIVOS_DADOS.copy()
for i in range(len(y_alvo)):
    y_alvo[i] = y_alvo[i][0:5]    # tira o .png

# Separa cada letra das strings da lista
y_separado = []
for i in y_alvo:    # para cada elemento do y (desempacota)
    for j in range(0, 5):    # para cada letra da primeira string da lista y
        y_separado.append(i[j])


# Fazendo o one hot encode manual
simbolos = string.ascii_lowercase + '1234567890'
y_final = np.zeros((len(y_separado), 36))    # adiciona 36 colunas para classificar nos 36 simbolos existentes
for i in range(len(y_separado)):
    char = y_separado[i]
    posicao_classe = simbolos.find(char)
    y_final[i, posicao_classe] = 1

# Eliminando variaveis não utilizadas
del y_alvo, y_separado, imagens, dataset

# Separando os dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x_inicial, y_final, test_size=0.25)
# Definindo parametros gerais
learning_rate = 0.001
epocas = 200
batch_size = 480

# Criando o modelo
modelo = Sequential()
modelo.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(38, 20, 1)))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
modelo.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
modelo.add(Flatten())
modelo.add(Dense(120, kernel_initializer='normal', activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(36, kernel_initializer='normal', activation='softmax'))

# Definindo o otimizador e função de custo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# Treinando o modelo
modelo.fit(x_treino, y_treino, batch_size=batch_size, epochs=epocas, validation_data=(x_teste, y_teste), verbose=1)


