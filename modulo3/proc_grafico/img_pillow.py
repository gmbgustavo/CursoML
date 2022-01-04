from PIL import Image

imagem = Image.open('../../../dados/imagens/mmx3.png')

# imagem.show()

print(imagem.size)

# Passando para escala de cinza
img_cinza = imagem.convert('L')

# Pillow trabalha com RGB



