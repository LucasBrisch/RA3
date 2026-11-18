"""
RA3 - Data Analytics: Campeonato Brasileiro Série A (2003-2025)

Problema investigado:
    Existe vantagem para o time mandante (fator casa) no Campeonato Brasileiro?
    E é possível prever se o mandante vencerá uma partida a partir do
    desempenho recente das equipes?

Etapas:
    1. Análise estatística descritiva
    2. Visualizações
    3. Amostragem e representatividade
    4. Machine Learning (classificação)
"""
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # salva os gráficos sem precisar abrir janela
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Garante que o CSV seja lido e os gráficos salvos na pasta do script,
# independentemente de onde o comando for executado
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# Carregamento e preparação
# ---------------------------------------------------------------------------
print("=" * 70)
print("CARREGAMENTO DOS DADOS")
print("=" * 70)
df = pd.read_csv("campeonato_brasileiro.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values(["data", "id"]).reset_index(drop=True)

# A temporada 2020 foi adiada pela pandemia e terminou em fevereiro de 2021.
# Por isso a temporada é definida pelo ano da data, exceto jogos de jan-mar/2021.
df["temporada"] = df["data"].dt.year
df.loc[(df["data"].dt.year == 2021) & (df["data"].dt.month <= 3), "temporada"] = 2020

df["total_gols"] = df["mandante_placar"] + df["visitante_placar"]
df["resultado"] = "Empate"
df.loc[df["mandante_placar"] > df["visitante_placar"], "resultado"] = "Vitória Mandante"
df.loc[df["mandante_placar"] < df["visitante_placar"], "resultado"] = "Vitória Visitante"

print(f"Partidas: {len(df)} | Temporadas: {df['temporada'].min()} a {df['temporada'].max()}")

# ---------------------------------------------------------------------------
# 1. Análise estatística descritiva
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. ANÁLISE ESTATÍSTICA DESCRITIVA")
print("=" * 70)

colunas = {"mandante_placar": "Gols mandante",
           "visitante_placar": "Gols visitante",
           "total_gols": "Total de gols"}
tabela = pd.DataFrame({
    nome: {
        "Média": df[c].mean(),
        "Mediana": df[c].median(),
        "Moda": df[c].mode()[0],
        "Desvio-padrão": df[c].std(),
        "Mínimo": df[c].min(),
        "Máximo": df[c].max(),
        "Amplitude": df[c].max() - df[c].min(),
        "Q1 (25%)": df[c].quantile(0.25),
        "Q3 (75%)": df[c].quantile(0.75),
    } for c, nome in colunas.items()
}).round(2)
print("\nTendência central e dispersão:")
print(tabela)

print("\nDistribuição - frequência de gols por partida (% das partidas):")
freq = pd.DataFrame({
    "Mandante (%)": df["mandante_placar"].value_counts(normalize=True).sort_index() * 100,
    "Visitante (%)": df["visitante_placar"].value_counts(normalize=True).sort_index() * 100,
}).fillna(0).round(1)
print(freq)

print("\nDistribuição dos resultados:")
resultados = df["resultado"].value_counts()
print(pd.DataFrame({"Partidas": resultados,
                    "%": (resultados / len(df) * 100).round(1)}))

print("\nPlacares mais frequentes (mandante x visitante):")
placares = (df["mandante_placar"].astype(str) + " x " + df["visitante_placar"].astype(str))
print((placares.value_counts(normalize=True).head(5) * 100).round(1).to_string())

# ---------------------------------------------------------------------------
# 2. Visualizações
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. VISUALIZAÇÕES")
print("=" * 70)

# Gráfico 1 - Distribuição de gols (barras lado a lado)
fig, ax = plt.subplots(figsize=(10, 5))
largura = 0.4
ax.bar(freq.index - largura / 2, freq["Mandante (%)"], largura, label="Mandante", color="royalblue")
ax.bar(freq.index + largura / 2, freq["Visitante (%)"], largura, label="Visitante", color="darkorange")
ax.set_title("Distribuição de gols marcados por partida: mandante x visitante")
ax.set_xlabel("Gols marcados na partida")
ax.set_ylabel("% das partidas")
ax.set_xticks(freq.index)
ax.legend()
plt.tight_layout()
plt.savefig("grafico_1_distribuicao_gols.png", dpi=150)
plt.close()

# Gráfico 2 - Proporção de resultados
ordem = ["Vitória Mandante", "Empate", "Vitória Visitante"]
perc = (resultados / len(df) * 100).reindex(ordem)
fig, ax = plt.subplots(figsize=(8, 5))
barras = ax.bar(ordem, perc, color=["seagreen", "gray", "indianred"])
ax.bar_label(barras, fmt="%.1f%%")
ax.set_title("Resultados das partidas - Brasileirão 2003-2025")
ax.set_xlabel("Resultado")
ax.set_ylabel("% das partidas")
plt.tight_layout()
plt.savefig("grafico_2_resultados_partidas.png", dpi=150)
plt.close()

# Gráfico 3 - % de vitórias do mandante por temporada
vit_temp = df.groupby("temporada")["resultado"].apply(lambda s: (s == "Vitória Mandante").mean() * 100)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(vit_temp.index, vit_temp.values, marker="o", color="seagreen", label="Vitórias do mandante")
ax.axhline(vit_temp.mean(), linestyle="--", color="gray", label=f"Média ({vit_temp.mean():.1f}%)")
ax.annotate("2020: jogos sem torcida\n(pandemia)", xy=(2020, vit_temp[2020]),
            xytext=(2010.3, 44.5), arrowprops={"arrowstyle": "->"})
ax.set_title("Percentual de vitórias do mandante por temporada")
ax.set_xlabel("Temporada")
ax.set_ylabel("% de vitórias do mandante")
ax.set_xticks(vit_temp.index)
ax.tick_params(axis="x", rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("grafico_3_vitorias_mandante_temporada.png", dpi=150)
plt.close()

print("Gráficos salvos: grafico_1_distribuicao_gols.png, grafico_2_resultados_partidas.png,")
print("                 grafico_3_vitorias_mandante_temporada.png")
print(f"% vitórias do mandante - maior: {vit_temp.idxmax()} ({vit_temp.max():.1f}%), "
      f"menor: {vit_temp.idxmin()} ({vit_temp.min():.1f}%)")

# ---------------------------------------------------------------------------
# 3. Amostragem e representatividade
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. AMOSTRAGEM E REPRESENTATIVIDADE")
print("=" * 70)
print("Não foi usada amostra: o conjunto contém todas as partidas da Série A em pontos corridos.")
print("\nPartidas por temporada:")
print(df["temporada"].value_counts().sort_index().to_string())
print(f"\nValores ausentes em 'arrecadacao': {df['arrecadacao'].isna().mean() * 100:.1f}%")
nao_inf = (df["formacao_mandante"] == "Não Informado").mean() * 100
print(f"Formação tática 'Não Informado' (mandante): {nao_inf:.1f}%")
inconsist = (((df["mandante_placar"] > df["visitante_placar"]) & (df["vencedor"] != df["mandante"])) |
             ((df["mandante_placar"] < df["visitante_placar"]) & (df["vencedor"] != df["visitante"]))).sum()
print(f"Partidas em que a coluna 'vencedor' não bate com o placar (grafia do nome): {inconsist}")
print("-> por isso o resultado foi calculado a partir do placar, e não da coluna 'vencedor'.")

# ---------------------------------------------------------------------------
# 4. Machine Learning - Classificação
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. MACHINE LEARNING - CLASSIFICAÇÃO")
print("=" * 70)
print("Problema: prever se o mandante vence a partida (1) ou não vence - empate/derrota (0).")

# Features: desempenho recente de cada time nos 5 jogos ANTERIORES à partida
# (pontos por jogo e saldo de gols por jogo). Usa-se apenas informação do passado.
casa = pd.DataFrame({"idx": df.index, "lado": "mand", "data": df["data"], "time": df["mandante"],
                     "gp": df["mandante_placar"], "gc": df["visitante_placar"]})
fora = pd.DataFrame({"idx": df.index, "lado": "visit", "data": df["data"], "time": df["visitante"],
                     "gp": df["visitante_placar"], "gc": df["mandante_placar"]})
jogos = pd.concat([casa, fora]).sort_values(["data", "idx"])
jogos["pontos"] = 0
jogos.loc[jogos["gp"] > jogos["gc"], "pontos"] = 3
jogos.loc[jogos["gp"] == jogos["gc"], "pontos"] = 1
jogos["saldo"] = jogos["gp"] - jogos["gc"]
g = jogos.groupby("time")
jogos["pts_ult5"] = g["pontos"].transform(lambda s: s.shift(1).rolling(5).mean())
jogos["saldo_ult5"] = g["saldo"].transform(lambda s: s.shift(1).rolling(5).mean())

for lado in ["mand", "visit"]:
    parte = jogos[jogos["lado"] == lado].set_index("idx")
    df[f"{lado}_pts_ult5"] = parte["pts_ult5"]
    df[f"{lado}_saldo_ult5"] = parte["saldo_ult5"]
df["mandante_venceu"] = (df["resultado"] == "Vitória Mandante").astype(int)

features = ["mand_pts_ult5", "mand_saldo_ult5", "visit_pts_ult5", "visit_saldo_ult5"]
base = df.dropna(subset=features)  # remove jogos em que o time ainda não tinha 5 jogos
print(f"Features: {features}")
print("Target: mandante_venceu")
print(f"Partidas usadas: {len(base)} (removidas {len(df) - len(base)} sem histórico de 5 jogos)")

# Separação treino/teste por tempo: treina no passado, testa nas temporadas mais recentes
treino = base[base["temporada"] <= 2021]
teste = base[base["temporada"] >= 2022]
X_train, y_train = treino[features], treino["mandante_venceu"]
X_test, y_test = teste[features], teste["mandante_venceu"]
print(f"Treino: temporadas 2003-2021 ({len(treino)} partidas)")
print(f"Teste:  temporadas 2022-2025 ({len(teste)} partidas)")

modelo = LogisticRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Baselines (referências mínimas):
#  - sempre prever a classe mais frequente no treino
#  - sempre prever a classe mais frequente no teste (referência mais exigente)
classe_maj = y_train.mode()[0]
acc_base = (y_test == classe_maj).mean()
acc_base_teste = max(y_test.mean(), 1 - y_test.mean())
acc_modelo = accuracy_score(y_test, y_pred)

print(f"\nBaseline 1 - sempre prever a classe majoritária do treino ('{classe_maj}'): {acc_base:.3f}")
print(f"Baseline 2 - sempre prever a classe majoritária do teste:          {acc_base_teste:.3f}")
print(f"Acurácia do modelo (Regressão Logística):                          {acc_modelo:.3f}")
print("\nMatriz de confusão (linhas = real, colunas = previsto):")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=["Real: não venceu", "Real: venceu"],
                   columns=["Prev: não venceu", "Prev: venceu"]))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=["Não venceu", "Venceu"]))
print("Coeficientes (positivo = aumenta chance de vitória do mandante):")
print(pd.Series(modelo.coef_[0], index=features).round(3).to_string())
print("\nProcesso concluído.")
