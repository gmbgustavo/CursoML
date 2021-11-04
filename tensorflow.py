import tensorflow as tf


frase = tf.constant('Olá mundo!')
with tf.Session() as sessao:
    rodar = sessao.run(frase)


print(rodar)

