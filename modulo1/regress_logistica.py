from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import pandas as pd

# Lendo o arquivo de dados
dados = pd.read_csv('../../dados/perfume.csv')

# Organizando os dados. Removendo colunas desnecessarias
dados.drop('Respondent.ID', axis=1, inplace=True)
dados.drop('Product', axis=1, inplace=True)
dados.dropna()


# Preenchendo dados faltantes com a media
dados['q8.12'].fillna(dados['q8.12'].median(), inplace=True)
dados['q8.7'].fillna(dados['q8.7'].median(), inplace=True)

# Criando os valores para ajuste do modelo
valores_C = [0.1, 0.45, 0.5, 0.55, 0.6, 0.7, 1, 1.6]
regularicacao = ['l1', 'l2']
valores_grid = {'C': valores_C, 'penalty': regularicacao}    # Unpacking em Python. Dicionario.


# Variavel preditora e variavel alvo
y = dados['Instant.Liking']
x = dados.drop('Instant.Liking', axis=1)

# Cfriando o modelo

dobras = StratifiedKFold(n_splits=10)    # Numero de folds para validacao cruzada
# Cria o modelo de regressao logistica com o maximo de iteracoes(padrao=100). E o principal.
# Em novas versoes, especificar esses parametros
modelo = LogisticRegression(C=1, penalty='l2', max_iter=50000, solver='liblinear')
# resultado = cross_val_score(modelo, x, y, cv=dobras)    # Faz o fit usando cross validation
# print(f'A precisao do modelo foi: {resultado.mean()}')

# Roda o modelo usando GridSearch, onde e possivel ajustar, encontrando os melhores valores
grid_rl = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=dobras, verbose=3)
grid_rl.fit(x, y)

print(f'Melhor precisao: {grid_rl.best_score_}')
print(f'Parametro C: {grid_rl.best_estimator_.C}')
print(f'Regularizacao: {grid_rl.best_estimator_.penalty}')

