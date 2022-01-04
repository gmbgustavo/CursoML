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


# Criando o ambiente
ambiente = gym.make('CartPole-v0')


# Criando o modelo de aprendizado
modelo = DQN(policy=MlpPolicy, env=ambiente, gamma=0.99, learning_rate=0.0005, buffer_size=5_000,
             exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.1,
             batch_size=32, learning_starts=5_000, verbose=1)
# exploration_initial_eps = Exploration inicial, nesse exemplo começa em 100
# exploration_final_eps = porcentagem de exploration
# exploration_fraction = porcentagem de iteracoes em que decai de exploration_initial para o final (1 até 0.1 no caso)
# learning starts = a partir de qual iteracao deixa de ser randomico e começa a policy (aquecimento)

# Treinando o modelo
modelo.learn(total_timesteps=100_000, log_interval=100)

# Fazendo ele jogar depois de treinado
obs = ambiente.reset()
terminou = False
recompensa_total = 0

while not terminou:
    acao, estado = modelo.predict(obs)
    obs, recompensa, terminou, info = ambiente.step(acao)
    recompensa_total += recompensa
    ambiente.render()
    time.sleep(0.02)

ambiente.close()
print(f'Recompensa total = {recompensa_total} (max=200)')

