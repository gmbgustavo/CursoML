from transformers import pipeline


modelo = pipeline('fill-mask')

frase = 'I would rather <mask> than losing my life.'

respostas = modelo(frase)

for resposta in respostas:
    frase = resposta['token_str']



