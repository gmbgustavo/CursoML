import matplotlib.pyplot as plot
import numpy as np
from cv2 import cv2    # Importando dessa maneira para funcionar o autocomplete


# Carregando uma imagem
tela = cv2.imread('../../../dados/atari/nave.png')
(altura, largura) = tela.shape[:2]
centro = (altura // 2, largura // 2)


# Mostrando algumas informações básicas
print(f'Altura {tela.shape[0]} pixels')
print(f'Largura {tela.shape[1]} pixels')
print(f'Canais de cores {tela.shape[2]}')
tela = cv2.cvtColor(tela, cv2.COLOR_RGB2GRAY)
tela = np.array(tela)
print(tela.shape)


'''
cv2.imshow('Imagem', tela)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvando uma copia
cv2.imwrite('../dados/imagens/salvada.png', tela)

tela = cv2.cvtColor(tela, cv2.COLOR_BGR2RGB)    # O opencv guarda as imagens no formato BGR e não RGB
tela = cv2.cvtColor(tela, cv2.COLOR_BGR2GRAY)    # Converter para escala de cinza
plot.imshow(tela)
plot.show()
'''

# Split e merge

# Carregando os canais de cores BGR separadamente
# (b, g, r) = cv2.split(tela)    # Unpacking para uma tupla

# Cria uma matriz de zeros com dimensoes da imagem 989x1319 no caso do mmx3 capturado
zeros = np.zeros(tela.shape[:2], dtype='uint8')

'''
cv2.imshow('Azul', cv2.merge([b, zeros, zeros]))
cv2.imshow('Verde', cv2.merge([zeros, g, zeros]))
cv2.imshow('Vermelho', cv2.merge([zeros, zeros, r]))
cv2.waitKey(delay=None)
cv2.destroyAllWindows()


print(tela[500, 500])    # Retorna 41, 48, 24
tela[500, 500] = (100, 255, 50)    # Opencv recebe no formato BGR

tela_canvas = tela.copy()    # copy() para deixar a variavel independente
# tela_canvas[500:599, 500:599] = (100, 255, 50)    # Retangulo cada x relaciona com cada y
cv2.imshow('Imagem modificada', tela_canvas[100:250, 500:1000])
cv2.waitKey(delay=None)
cv2.destroyAllWindows()
'''
