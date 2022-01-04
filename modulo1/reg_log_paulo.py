from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd

# Lendo o arquivo de dados
dados = pd.read_csv('../../dados/multi_step_clean.csv')

# Organizando os dados. Removendo colunas desnecessarias
# dados.drop('Procedimento', axis=1, inplace=True)
# dados.drop('Metodo', axis=1, inplace=True)
# dados.dropna()

'''
# Criando os valores para ajuste do modelo
valores_C = [0.1, 0.45, 0.48, 0.5, 0.55, 0.6, 0.7, 1, 1.6]
regularicacao = ['l1', 'l2']
valores_grid = {'C': valores_C, 'penalty': regularicacao}    # Unpacking em Python. Dicionario.
'''

# Variavel preditora e variavel alvo
y = dados['L1']
x = dados.drop('L1', axis=1)

# Cfriando o modelo

dobras = StratifiedKFold(n_splits=6)    # Numero de folds para validacao cruzada
# Cria o modelo de regressao logistica com o maximo de iteracoes(padrao=100). E o principal.
# Em novas versoes, especificar esses parametros
# modelo = LogisticRegression(C=0.45, penalty='l1', max_iter=50000, solver='saga')
modelo = Ridge()
# resultado = cross_val_score(modelo, x, y, cv=dobras)    # Faz o fit usando cross validation
# print(f'A precisao do modelo foi: {resultado.mean()}')
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2)
modelo.fit(x_treino, y_treino)
# pontuacao da precisao
resultado_final = modelo.score(x_teste, y_teste)
print(f'Precisao: {resultado_final}')

predicao = modelo.predict(x_teste)
# matriz = confusion_matrix(y_teste, predicao)
# print(f'Confusion Matrix: \n{matriz}')

'''
# Roda o modelo usando GridSearch, onde e possivel ajustar, encontrando os melhores valores
# grid_rl = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=dobras, verbose=3)
# grid_rl.fit(x, y)

print(f'Melhor precisao: {grid_rl.best_score_}')
print(f'Parametro C: {grid_rl.best_estimator_.C}')
print(f'Regularizacao: {grid_rl.best_estimator_.penalty}')
'''