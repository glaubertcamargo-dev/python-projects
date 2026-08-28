import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ==============================================
# 1) DADOS DE EXEMPLO
# ==============================================
# Em um projeto real, poderiamos usar:
# df = pd.read_csv("vendas.csv")

csv_simulado = """id,nome,idade,cidade,vendas
1,Ana,28,Goiania,1500
2,Bruno,33,Sao Paulo,2200
3,Carla,35,Goiania,1800
4,Diego,41,Curitiba,2600
5,Elaine,42,Goiania,1950
6,Felipe,29,Sao Paulo,2100
"""

df = pd.read_csv(StringIO(csv_simulado))

# Garante que a coluna "vendas" seja numerica.
# Valores invalidos sao convertidos em NaN.
df["vendas"] = pd.to_numeric(df["vendas"], errors="coerce")

# Remove linhas que ficaram sem valor valido de vendas.
df = df.dropna(subset=["vendas"])

print("DADOS UTILIZADOS NA ANALISE:")
print(df.to_string(index=False))

# ====================================================
# 2) CRIA UMA JANELA COM DOIS GRÁFICOS: BARRAS E PIZZA
# ====================================================

fig, (ax_barras, ax_pizza) = plt.subplots(1, 2, figsize=(14, 6))

# ===========================================
# 3) GRÁFICO DE BARRAS: VENDAS POR VENDEDOR
# ===========================================
barras = ax_barras.bar(df["nome"], df["vendas"], color="skyblue", edgecolor="black")
ax_barras.set_title("Vendas por Vendedor", fontsize=14, fontweight="bold")
ax_barras.set_xlabel("Vendedor")
ax_barras.grid(axis="y", linestyle="--", alpha=0.4)

    # Aumenta o limite superior para os valores não ficarem cortados
ax_barras.set_ylim(0, df["vendas"].max() * 1.15)

    # Escreve o valor acima de cada barra.
for barra in barras:
    valor = barra.get_height()

    ax_barras.text(barra.get_x() + barra.get_width() / 2, valor + df["vendas"].max() * 0.02, f"R$ {valor:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")


# ===========================================
# 4) GRÁFICO DE PIZZA: VENDAS POR CIDADE
# ===========================================
vendas_por_cidade = df.groupby("cidade")["vendas"].sum()
ax_pizza.pie(vendas_por_cidade, labels=vendas_por_cidade.index, autopct="%1.1f%%", startangle=90, colors=["#ff9999", "#66b3ff", "#99ff99"])
ax_pizza.set_title("Participação de Vendas por Cidade", fontsize=14, fontweight="bold")

    # Ajusta o espaçamento entre os gráficos para que não fiquem sobrepostos.
plt.tight_layout()

    # Salva a janela inteira com os dois gráficos.
plt.savefig("graficos_vendas.png", dpi=150)

    # Exibe a janela com os dois gráficos.
plt.show()

