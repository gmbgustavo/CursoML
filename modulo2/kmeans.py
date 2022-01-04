import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

x = pd.read_csv('../../dados/multi_step_clean.csv')
# Converter os dados internos do datase para DataFrame do Pandas
y = x['Falha']

# Para KMeans é necessário normalizaçao dos dados
normalizador = MinMaxScaler(feature_range=(0, 1))
x_norm = normalizador.fit_transform(x)

# Criação do modelo
modelo = KMeans(n_clusters=2, n_init=1000, max_iter=20000, algorithm='auto')
modelo.fit(x_norm)


# Informando distancia dos centros dos clusters
print(f'Modelo do Paulo:\n {modelo.cluster_centers_}')


predicoes = modelo.predict(x_norm)
precisao = accuracy_score(y, predicoes)

print(f'Precisão da predição de falha: {precisao*100}')
print(predicoes)

