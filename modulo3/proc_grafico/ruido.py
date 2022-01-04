from cv2 import cv2

# Imagem com ruido
img_orig = cv2.imread('../../../dados/imagens/cerebro_ruido.jpeg')


# Aplicando alguns filtros - tem que ser impar para ter um centro
kernel3 = cv2.medianBlur(img_orig, 3)
kernel5 = cv2.medianBlur(img_orig, 5)
kernel7 = cv2.medianBlur(img_orig, 7)
print(f'Tamanho da imagem em pixels: {img_orig.shape}')

gaussian = cv2.GaussianBlur(img_orig, (5, 5), 255)

cv2.imshow('Original', img_orig)
cv2.imshow('Gaussian', gaussian)
cv2.imshow('Kernel 3', kernel3)
cv2.imshow('Kernel 5', kernel5)
cv2.imshow('Kernel 7', kernel7)
cv2.waitKey(0)
cv2.destroyAllWindows()

