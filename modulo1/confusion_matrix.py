import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.datasets import load_breast_cancer


# Carrega os dados internos dos datasets de exemplo do pacote
dados = load_breast_cancer()

# Variaveeis preditoras e variavel alvo
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series(dados.target)

# Cria o modelo de regressao logistica com o maximo de iteracoes(padrao=100). E o principal.
# Em novas versoes, especificar esses parametros
modelo = LogisticRegression(C=95, penalty='l1', max_iter=50000, solver='liblinear')
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.4)
modelo.fit(x_treino, y_treino)

# pontuacao da precisao
resultado_final = modelo.score(x_teste, y_teste)
print(f'Precisao: {resultado_final}')

predicao = modelo.predict(x_teste)
matriz = confusion_matrix(y_teste, predicao)
print(f'Confusion Matrix: \n{matriz}')

probabilidades = modelo.predict_proba(x_teste)
probabilidades = probabilidades[:, 1]    # Todas as linhas, coluna 1
fpr, tpr, limiares = roc_curve(y_teste, probabilidades)
print(roc_auc_score(y_teste, probabilidades))

'''
verdadeiro |  falso
positivo   |  positivo
__________ |____________
falso      |  verdadeiro
negativo   |  negativo
'''
