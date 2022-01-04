"""
Para que o pré-processamento seja feito, vamos usar função make_atari_env. Essa função recebe como entrada
o jogo atari e retorna na variavem um shape de (84, 84, 1), reduzindo o tamanho da imagem de entrada e deixa uma
dimensão extra para fazermos stacking. Para agrupar 4 frames no staking, existe a função VecFrameStack, informando
env = VecFrameStack(env, n_stack=4). Isso muda o shape para (1, 84, 84, 4). Essa coluna extra no inicio é para o
parametro num_env que se passa para make_atari_env dizendo quantos environments vai se rodar em paralelo.
A função make_atari_env também faz skipping, pulando de 4 em 4.
Em caso de ambientes paralelos, a variavel de recompensa e de fim retornarão uma lista com cada resultado
Obs.: nem todos os algoritmos suportam múltiplos envs. DQN não suporta.
"""

import multiprocessing
import time
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from gym import spaces

ambiente = make_atari_env('BreakoutNoFrameskip-v4')
ambiente = VecFrameStack(ambiente, n_stack=4)    # Faz stack de 4 frames em um estado

# Criando o modelo de aprendizado
modelo = A2C(policy='CnnPolicy', env=ambiente, gamma=0.99, n_steps=128, learning_rate=0.00025,
             vf_coef=0.5, verbose=1)

# Configurando o checkpoint para que o modelo seja salvo de tempos em tempos
save = CheckpointCallback(save_freq=50_000, save_path='../../dados/breakout', name_prefix='state_')

# Treinando o modelo
modelo.learn(total_timesteps=1_000_000, log_interval=20, callback=save)   # Log é por episodio

# Fazendo ele jogar depois de treinado
obs = ambiente.reset()
terminou = False
recompensa_total = 0

# Vendo ele jogar
while not terminou:
    acao, estado = modelo.predict(obs)
    obs, recompensa, terminou, info = ambiente.step(acao)
    recompensa_total += recompensa
    ambiente.render()
    time.sleep(0.001)

ambiente.close()
print(f'Recompensa total = {recompensa_total} (max=21)')

