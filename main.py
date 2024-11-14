# Importar bibliotecas
import pandas as pd
from scipy.io import arff
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
data = arff.loadarff('analcatdata_challenger.arff')
df = pd.DataFrame(data[0])

# Conversão dos dados para strings sem bytes, se necessário
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Transformar colunas numéricas
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df['Pressure'] = pd.to_numeric(df['Pressure'], errors='coerce')
df['Damaged'] = pd.to_numeric(df['Damaged'], errors='coerce')
df['O-rings'] = pd.to_numeric(df['O-rings'], errors='coerce')

# Calcular medidas descritivas
descriptive_stats = df[['Temperature', 'Pressure', 'Damaged', 'O-rings']].describe(percentiles=[0.25, 0.5, 0.75])
print("Medidas descritivas:")
print(descriptive_stats)

# Calcular a matriz de correlação
correlation_matrix = df[['Temperature', 'Pressure', 'Damaged', 'O-rings']].corr()
print("\nMatriz de correlação:")
print(correlation_matrix)

# Plotar o heatmap da matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

# Gráfico de dispersão entre Temperature e Damaged
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Temperature', y='Damaged', hue='O-rings', palette='viridis')
plt.title('Relação entre Temperatura e Danos')
plt.xlabel('Temperatura')
plt.ylabel('Danos')
plt.legend(title='O-rings')
plt.show()