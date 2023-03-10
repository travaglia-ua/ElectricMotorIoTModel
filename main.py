# Importar as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados de sensores em um DataFrame do Pandas
data = pd.read_csv('data.csv')

# Selecionar as características relevantes para o modelo
X = data[['temperatura', 'vibração', 'corrente']]
y = data['falha']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pré-processar os dados de treinamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Pré-processar os dados de teste
X_test = scaler.transform(X_test)

# Fazer previsões com o modelo treinado
y_pred = model.predict(X_test)

# Avaliar a precisão do modelo
mse = mean_squared_error(y_test, y_pred)
print('Erro médio quadrático: {:.2f}'.format(mse))
