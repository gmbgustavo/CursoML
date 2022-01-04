import numpy as np
from cv2 import cv2    # Importando dessa maneira para funcionar o autocomplete

canvas = np.full((480, 640, 3), 255, dtype='uint8')    # uint8 = 8 bits (255) - checar se é x-y ou y-x
print(canvas.shape)
# canvas = np.zeros((480, 640, 3), dtype='uint8')

# Desenhando no espaço criado
verde = (0, 255, 0)
laranja = (1, 190, 200)
cv2.line(canvas, (0, 0), (640, 480), verde, 2)    # ultimo argumento expessura
cv2.rectangle(canvas, (10, 10), (50, 50), laranja, -1)    # -1 preenchido

# circulos concentricos
(centro_x, centro_y) = (canvas.shape[1] // 2, canvas.shape[0] // 2)    # cv2 faz y-x, trocando para x-y
for raio in range(25, 250, 25):
    cv2.circle(canvas, (centro_x, centro_y), raio, laranja, 2)

# Texto
cv2.putText(canvas, 'Paula Tejano', (180, 40), cv2.FONT_HERSHEY_PLAIN, 3,  verde, 3, cv2.LINE_AA)
# imagem, texto, posicao inicial, fonte, tamanho, cor, espessura, linha

mascara = np.zeros(canvas.shape[:2], dtype='uint8')

cv2.imshow('Linha verde', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


