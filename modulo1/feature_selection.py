from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Definindo variaveis preditoras e target
dados = pd.read_csv('../../dados/multi_step_clean.csv')
y = dados['Falha']
dados.drop('L1', axis=1, inplace=True)
dados.drop('L2', axis=1, inplace=True)
dados.drop('L3', axis=1, inplace=True)
dados.drop('E1', axis=1, inplace=True)
dados.drop('E2', axis=1, inplace=True)
dados.drop('E3', axis=1, inplace=True)
x = dados.drop('Falha', axis=1)

'''
# Selecionando duas variaveis com o maior chi-quadrado ou f_classif
algoritmo = SelectKBest(score_func=f_classif, k=4)    # k informa o numero de colunas que vc quer
dados_best_preditoras = algoritmo.fit_transform(x, y)

# Resultados
print(f'Scores:\n {algoritmo.scores_}')
print(f'Total de colunas: {len(algoritmo.scores_)}')
print(f'Resultado da transformacao: \n {dados_best_preditoras}')
'''

# Elininacao recursova de features
modelo = LogisticRegression(max_iter=5000)
rfe = RFE(estimator=modelo, n_features_to_select=5)
modelo_learning = rfe.fit_transform(x, y)

print(f'Atributos Selecionados: \n {modelo_learning}')
print(f'Ranking dos atributos: \n {modelo_learning}')

