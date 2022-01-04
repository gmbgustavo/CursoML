import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge

arquivo = pd.read_csv('../../dados/kc_house_data.csv')

# Funcao que troca o tipo
# arquivo['Metodo'] = arquivo['Metodo'].astype(int)

# Removendo dados irrelevantes
arquivo.drop('id', axis=1, inplace=True)
arquivo.drop('date', axis=1, inplace=True)
arquivo.drop('zipcode', axis=1, inplace=True)
arquivo.drop('lat', axis=1, inplace=True)
arquivo.drop('long', axis=1, inplace=True)
arquivo.dropna()

# Criacao do modelo
y = arquivo['price']
x = arquivo.drop('price', axis=1)


def linear_reg_kfold(a, b):
    dobras = KFold(n_splits=10)
    x = a
    y = b
    reg = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elastic = ElasticNet()
    result_reg = cross_val_score(reg, x, y, cv=dobras)
    result_ridge = cross_val_score(ridge, x, y, cv=dobras)
    result_lasso = cross_val_score(lasso, x, y, cv=dobras)
    result_elastic = cross_val_score(elastic, x, y, cv=dobras)
    dict_resultados = {'Linear': result_reg.mean(), 'Ridge': result_ridge.mean(), 'Lasso': result_lasso.mean(),
                       'Elastic': result_elastic.mean()}
    melhor_resultado = max(dict_resultados, key=dict_resultados.get)
    print(f'Linear: {result_reg.mean()}, Ridge: {result_ridge.mean()}, Lasso: {result_lasso.mean()} '
          f'Elastic {result_elastic.mean()}')
    print(f'O melhor modelo foi: {melhor_resultado}')


linear_reg_kfold(x, y)
