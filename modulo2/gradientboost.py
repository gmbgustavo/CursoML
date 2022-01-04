import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

dados = pd.read_csv('../../dados/multi_step_clean.csv')
y = dados['Falha']
x = dados.drop('Falha', axis=1)


modelo = GradientBoostingClassifier()
dobras = StratifiedKFold(n_splits=12)


resultado = cross_val_score(modelo, x, y, cv=dobras, n_jobs=3)

print(resultado.mean())
