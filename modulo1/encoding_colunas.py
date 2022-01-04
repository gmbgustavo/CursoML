import pandas as pd
import matplotlib.pyplot as plot


# Arquivo de entrada de dados
dados = pd.read_csv('../../dados/Traffic_Collision_Data_LA.csv')

# Encoding
area_encode = pd.get_dummies(dados['Area Name'])

# Concatenando o set encodado no set original
dados_concatenados = pd.concat([dados, area_encode])
dados_concatenados.drop(['Area Name'], axis=1, inplace=True)
dados_concatenados.boxplot(column='Crime Code')
plot.show()

