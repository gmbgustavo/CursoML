"""
O ambitente Cart Pole, consiste em um carrinho equilibrando um mastro.
O objetivo é equilibrar o mastro no carrinho, dando comandos +1 ou -1 de força ao carrinho
O jogo termina quando o mastro desalinha mais de 15 graus do centro, ou o carrinho se move mais que 2.4 unidades
de si mesmo para esquerda ou direita
O estado do ambiente é descrito por:
1. Posição do carrinho 2 - Velocidade do carrinho - 3. Angulo do mastro - 4. Velocidade angular do mastro
"""
import gym
import time
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

# Criando um ambiente
ambiente = gym.make('CartPole-v0')

'''
# Verificando o escopo de ações e a quantidade de ações disponíveis
print(ambiente.action_space)
print(ambiente.action_space.n)

# Gerando algumas ações aleatórias
for i in range(10):
    print(ambiente.action_space.sample())    # Função sample sorteia uma ação aleatória
'''

lista_visual = []    # Cria uma lista de todas as visualizacoes para informacao e analise
obs = ambiente.reset()    # Reseta o ambiente e retorna uma observação do estado inicial do ambiente

for i in range(100):    # Vamos rodar 100 ações
    ambiente.render()    # Mostra o ambiente na tela
    lista_visual.append((ambiente.render(mode='rgb_array')))    # Salva cada renderizacao para analise posterior
    # mode='rgb_array' retorna um array numpy com os valores RGB das posicoes. mostrado por imshow do matplotlib
    time.sleep(0.05)    # Faz uma breve pausa para podermos acompanhar a renderização que é muito rapida
    # gera uma nova ação aleatoria - retorna a acao tomada ambiente.action_space.sample()
    if obs[2] > 0:
        acao = 1
    else:
        acao = 0
    obs, recompensa, terminou, info = ambiente.step(acao)
    # A partir de cada ação, vai para um novo estado e recebe a recompensa, diz se o episodio terminou(done) e
    # traz informações extras, se houver(depende do ambiente)
    print(obs)
    if terminou:
        print(f'\nEpisodio finalizado depois de {i+1} passos')
        break

ambiente.close()    # Fecha a janela depois do término






