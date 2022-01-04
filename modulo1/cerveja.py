import pandas as pd
import matplotlib.pyplot as plot
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

dados = pd.read_csv('../../dados/recipeData.csv', encoding='latin')

# Ver percentual de dados faltantes de cada coluna
# faltantes = dados.isnull().sum()
# faltantes_percentual = (faltantes / len(dados['StyleID'])) * 100
# print(faltantes_percentual)

# Selecionar apenas os dados com mais de X amostras
# print(dados['StyleID'].value_counts())
selecao = dados.loc[dados['StyleID'].isin([7, 10, 134, 9, 4, 30, 86, 12, 92, 6, 175, 39])]
del dados

# Excluir colunas sem dados relevantes
selecao.drop('BeerID', axis=1, inplace=True)
selecao.drop('Name', axis=1, inplace=True)
selecao.drop('URL', axis=1, inplace=True)
selecao.drop('Style', axis=1, inplace=True)
selecao.drop('UserId', axis=1, inplace=True)
selecao.drop('PrimingMethod', axis=1, inplace=True)
selecao.drop('PrimingAmount', axis=1, inplace=True)
selecao.drop('MashThickness', axis=1, inplace=True)
selecao.drop('PrimaryTemp', axis=1, inplace=True)

# Transforma a classe SugarScale em 0 e 1 por so ter dois valores texto
selecao['SugarScale'] = selecao['SugarScale'].replace('Specific Gravity', 0)
selecao['SugarScale'] = selecao['SugarScale'].replace('Plato', 1)

# Transofmando a coluna brewmethod usando o Onehotencode, por ser mais complexa que dois valores
brewmethod_encode = pd.get_dummies(selecao['BrewMethod'])
selecao.drop('BrewMethod', axis=1, inplace=True)
selecao = pd.concat([selecao, brewmethod_encode], axis=1)

'''
# Verificar a distribuição dos dados faltantes
selecao.boxplot(column=['BoilGravity', 'MashThickness', 'PitchRate', 'PrimaryTemp'])
plot.show()
selecao.hist(column=['BoilGravity', 'MashThickness', 'PitchRate', 'PrimaryTemp'])
plot.show()
'''

# Preenchendo valores vazios e nao numericos com a media ou mediana
selecao['BoilGravity'].fillna(selecao['BoilGravity'].median(), inplace=True)   # Inplace - ela recebe ela mesma
# selecao.fillna(selecao.median(), inplace=True)

# Preenchendo valores faltantes com regressão
x_treino = selecao[selecao['PitchRate'].notnull()]    # Todas as variaveis que atendam a condicao
x_treino.drop('PitchRate', axis=1, inplace=True)
y_treino = selecao[selecao['PitchRate'].notnull()]['PitchRate']    # y_treino = selecao['PitchRate´]
x_preench = selecao[selecao['PitchRate'].isnull()]
x_preench.drop('PitchRate', axis=1, inplace=True)
y_preench = selecao[selecao['PitchRate'].isnull()]['PitchRate']

modelo = DecisionTreeRegressor()
modelo.fit(x_treino, y_treino)

# Predicao de novos valores
y_preench = modelo.predict(x_preench)

# Uso dos valores, concatena os valores encontrados nas linhas onde esta vazio
selecao.PitchRate[selecao.PitchRate.isnull()] = y_preench

# Ver percentual de dados faltantes de cada coluna
faltantes = selecao.isnull().sum()
faltantes_percentual = (faltantes / len(selecao['StyleID'])) * 100
print(faltantes_percentual)

# Variaveis preditoras e target
y = selecao['StyleID']
x = selecao.drop('StyleID', axis=1)


def modelosclassificacao(a, b) -> None:
    # Selecionando os parametros
    dobras = StratifiedKFold(n_splits=3)
    x = a
    y = b
    # Normaliza as variaveis preditoras para o KNN
    normalizador = MinMaxScaler(feature_range=(0, 1))
    x_norm = normalizador.fit_transform(x)

    logist = LogisticRegression(max_iter=100, verbose=2, n_jobs=3)
    naive = GaussianNB()
    decision_tree = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    gboost = GradientBoostingClassifier(n_estimators=100, verbose=3)
    r_logist = cross_val_score(logist, x, y, cv=dobras)
    r_naive = cross_val_score(naive, x, y, cv=dobras)
    r_knn = cross_val_score(knn, x_norm, y, cv=dobras)    # O KNN precisa de normalizacao
    r_decision_tree = cross_val_score(decision_tree, x, y, cv=dobras)
    r_gboost = cross_val_score(gboost, x, y, cv=dobras)

    # Formatando resultados para mostrar
    dicio_res = {'Logistica': r_logist.mean(), 'Naive': r_naive.mean(), 'Arvore': r_decision_tree.mean(),
                 'KNN': r_knn.mean(), 'Gradient Boost': r_gboost.mean()}
    melhor_modelo = max(dicio_res, key=dicio_res.get)
    print(f'O Melhor resultado foi: {melhor_modelo} com um valor de {dicio_res[melhor_modelo]}')


modelosclassificacao(x, y)
