import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Carregando o dataset
df = pd.read_csv("STI-20250619093223815.csv", delimiter=';',skiprows=1)

# Renomeando colunas
df.columns = ["Date", "Median_Value"]

# Excluindo todas as linhas invalidas
df = df[df["Date"].str.match(r"\d{2}/\d{4}", na=False)]
df = df[df["Median_Value"].str.contains(r"\d", na=False)]  # Só mantém linhas com número

# Corrigindo valores: removendo ponto como milhar e trocando a vírgula por ponto decimal
df["Median_Value"] = df["Median_Value"].str.replace(".", "", regex=False)
df["Median_Value"] = df["Median_Value"].str.replace(",", ".", regex=False)
df["Median_Value"] = df["Median_Value"].astype(float)

# Convertendo a coluna de data e definindo como índice
df["Date"] = pd.to_datetime(df["Date"], format="%m/%Y")
df.set_index("Date", inplace=True)

# Plotando a série
plt.figure(figsize=(12, 5))
plt.plot(df["Median_Value"], label="Valor Mediano (R$)", color="purple")
plt.title("Série Temporal - Valor Mediano dos Imóveis")
plt.xlabel("Ano")
plt.ylabel("Preço Médio dos Imóveis (R$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Decompondo a série
decomposicao = seasonal_decompose(df["Median_Value"], model="additive", period=12)
decomposicao.plot()
plt.suptitle("Decomposição da Série - Valor dos Imóveis", fontsize=14)
plt.tight_layout()
plt.show()


# ---------  Respostas ----------

#A. A série possui Tendência?
# - Sim, A tendência é claramente crescente ao longo do tempo. O preço médio dos imóveis aumenta de forma consistente ao longo dos meses.

#B. A série possui Sazonalidade?
# - Sim, o padrão se repete a cada 12 meses (1 ano), típico de
# séries econômicas com flutuações mensais influenciadas por fatores como juros, inflação, ou sazonalidade do setor imobiliário.

#C. A série apresenta um Ciclo?
# -  Não de forma clara, no gráfico, não há sinais visíveis de ciclos econômicos marcantes, apenas tendência e sazonalidade bem definidas.
