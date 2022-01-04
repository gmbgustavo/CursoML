from stable_baselines3.common.vec_env import SubprocVecEnv    # Ambientes paralelos

iterations = 0
lista = []
progresso = 0
print('\n')

for n in range(5, 100000):
    if n % 1_000 == 0:
        progresso += 1
        print(f'\rCálculos: {progresso}000', end='')
    while n != 1:
        iterations += 1
        if n % 2 == 0:
            n /= 2
        else:
            n = 3 * n + 1
    lista.append(iterations)
    iterations = 0


maior = max(lista)
print(f'\n\nMaior iteração: {maior} passos')


