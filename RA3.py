import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configuração de estilo dos gráficos
sns.set_theme(style="whitegrid")

print("--- 1. Carregamento e Preparação dos Dados ---")
df = pd.read_csv('campeonato_brasileiro.csv')
print(f"Total de registros carregados: {len(df)}")

# Criando coluna auxiliar de total de gols
df['total_gols'] = df['mandante_placar'] + df['visitante_placar']

print("\n--- 2. Análise Estatística Descritiva ---")
print("Gols do Mandante -> Média:", round(df['mandante_placar'].mean(), 2), 
      "| Mediana:", df['mandante_placar'].median(), 
      "| Desvio Padrão:", round(df['mandante_placar'].std(), 2),
      "| Min/Max:", f"{df['mandante_placar'].min()}/{df['mandante_placar'].max()}")

print("Gols do Visitante -> Média:", round(df['visitante_placar'].mean(), 2), 
      "| Mediana:", df['visitante_placar'].median(), 
      "| Desvio Padrão:", round(df['visitante_placar'].std(), 2),
      "| Min/Max:", f"{df['visitante_placar'].min()}/{df['visitante_placar'].max()}")

print("Total de Gols -> Média:", round(df['total_gols'].mean(), 2), 
      "| Mediana:", df['total_gols'].median(), 
      "| Desvio Padrão:", round(df['total_gols'].std(), 2))

# Distribuição de resultados
def get_resultado(row):
    if row['vencedor'] == '-':
        return 'Empate'
    elif row['vencedor'] == row['mandante']:
        return 'Vitória Mandante'
    else:
        return 'Vitória Visitante'

df['resultado_tipo'] = df.apply(get_resultado, axis=1)
print("\nProporção de Resultados:")
print(df['resultado_tipo'].value_counts(normalize=True) * 100)

print("\n--- 3. Geração de Visualizações ---")
# Gráfico 1: Distribuição de Gols
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='mandante_placar', color='royalblue', label='Mandante', alpha=0.7)
sns.countplot(data=df, x='visitante_placar', color='orange', label='Visitante', alpha=0.7)
plt.title('Distribuição de Gols por Partida (Mandante vs Visitante)')
plt.xlabel('Número de Gols')
plt.ylabel('Frequência (Partidas)')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_1_distribuicao_gols.png', dpi=300)
plt.close()

# Gráfico 2: Proporção de Resultados
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='resultado_tipo', order=['Vitória Mandante', 'Empate', 'Vitória Visitante'], palette='viridis')
plt.title('Proporção de Resultados no Campeonato Brasileiro')
plt.xlabel('Tipo de Resultado')
plt.ylabel('Quantidade de Partidas')
plt.tight_layout()
plt.savefig('grafico_2_resultados_partidas.png', dpi=300)
plt.close()

# Gráfico 3: Média de Gols por Rodada
gols_por_rodada = df.groupby('rodata')['total_gols'].mean().reset_index()
plt.figure(figsize=(12, 5))
sns.lineplot(data=gols_por_rodada, x='rodata', y='total_gols', color='darkgreen', marker='o')
plt.title('Média de Gols por Rodada do Campeonato')
plt.xlabel('Rodada')
plt.ylabel('Média de Gols')
plt.tight_layout()
plt.savefig('grafico_3_media_gols_rodada.png', dpi=300)
plt.close()
print("Gráficos salvos com sucesso!")

print("\n--- 4. Aplicação de Machine Learning (Classificação) ---")
# Criar coluna alvo binária: 1 se mandante venceu, 0 caso contrário
df['mandante_venceu'] = (df['vencedor'] == df['mandante']).astype(int)

# Seleção de Features e Target
features_df = df[['rodata', 'mandante_estado', 'visitante_estado']].copy()
features_df = pd.get_dummies(features_df, columns=['mandante_estado', 'visitante_estado'], drop_first=True)

X = features_df
y = df['mandante_venceu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento com Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Acurácia do Modelo: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Processo concluído com sucesso!")