# GYM-RETRO

import retro
from gym.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv
import callback

# Variaveis globais
SAVE_DIR = './save'
LOG_DIR = './logs'
JOGO = 'GalagaDemonsOfDeath-Nes'
STATE_SAVE = '1Player.Level1'
save = callback.TrainAndLoggingCallback(check_freq=15_000, save_path=SAVE_DIR)


# Criando o ambiente do jogo
env = retro.make(game=JOGO, state=STATE_SAVE)


# Preprocessamento dos frames para passar para a IA
# Escala de cinza
env = GrayScaleObservation(env, keep_dim=True)    # Mantem as dimensões do array
# Wrap no DummyEnvironment
env = DummyVecEnv([lambda: env])
# Stack nos frames
env = VecFrameStack(env, 4, channels_order='last')


# Criando o modelo
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.00025, n_steps=256, gamma=0.99)

# Carregar depois de treinado
model = PPO.load('./save/best_model_45000.zip', env=env)

# Treinando
model.learn(total_timesteps=4_000_000, callback=save)

'''
# Fazer ele jogar com o treinamento carregado anteriormente
state = env.reset()
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
'''
