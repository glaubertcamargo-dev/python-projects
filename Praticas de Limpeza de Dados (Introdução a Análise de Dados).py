import pandas as pd
from io import StringIO

#===============================
# 1) CARREGANDO DADOS
#===============================

csv_simulado = """id,nome,idade,cidade,data
1,Alice,30,São Paulo,2023-01-01
2,Joao da Silva,25,Rio de Janeiro,2023-02-15
3,Marli,35,Belo Horizonte,2023-03-10
4,David,28,Curitiba,2023-04-05
5,Eva,22,Porto Alegre,2023-05-20
6, Leonardo,20,Recife,2023-06-15
7, Pedro ,30, Salvador,2023-07-30
"""
df = pd.read_csv(StringIO(csv_simulado))

print("=" * 60)
print("DADOS CARREGADOS")
print("=" * 60)
print(df)


#==================================
# 2) VERIFICANDO VALORES FALTANTES
#==================================

print("\n" + "=" * 60)
print("2) VERIFICANDO VALORES FALTANTES")
print("=" * 60)
print(df.isnull().sum())

#======================================================
# 3) PREENCHENDO VALORES FALTANTES COM A MEDIA DA COLUNA
#======================================================

print("\n" + "=" * 60)
print("3) PREENCHENDO VALORES FALTANTES COM A MEDIA")
print("=" * 60)

# Correcao: .mean() precisa dos parenteses, e inplace com "i" minusculo
df["idade"] = df["idade"].fillna(df["idade"].mean())
print(df)

# Alternativa(comentada): remover linhas com valores faltantes
# df = df.dropna()

#==================================
# 4) REMOVENDO DUPLICATAS
#==================================
print("\n" + "=" *60)
print("4) REMOVENDO DUPLICATAS")
print("=" * 60)

# Removendo todas as linhas totalmente duplicadas
df = df.drop_duplicates()
print("Apos remover duplicatas completas:")
print(df)

# Removendo duplicatas com base em uma coluna específica (ex: 'nome')
df_unico = df.drop_duplicates(subset='nome', keep='first')
print("\nApos remover duplicatas com base na coluna 'nome':")
print(df_unico)

#==============================================
# 5) REMOVENDO OUTLIERS COM BASE NO Z-SCORE
#==============================================

print("\n" + "=" * 60)
print("5) REMOVENDO OUTLIERS COM BASE NO Z-SCORE")
print("=" * 60)

# Correcao: aspas balanceadas e parenteses corretos

df_unico["idade_normalizada"] = (df_unico["idade"] - df_unico["idade"].mean()) / df_unico["idade"].std()
df_unico = df_unico[df_unico["idade_normalizada"].abs() <= 3]
print("Apos remover outliers com base no Z-score:")
print(df_unico)

#========================================
# 6) CONVERTENDO TIPOS DE DADOS
#========================================

print("\n" + "=" * 60)
print("6) CONVERTENDO TIPOS DE DADOS")
print("=" * 60)

# Convertendo coluna de string para datetime
df_unico["data"] = pd.to_datetime(df_unico["data"])
print("Apos converter a coluna 'data' para datetime:")
print(df_unico)

# Convertendo coluna numerica para float
df_unico["idade"] = df_unico["idade"].astype(float)
print("\nApos converter a coluna 'idade' para float:")
print(df_unico)

print(df_unico.dtypes)
print(df_unico)

print("\n" + "=" * 60)
print("LIMPEZA DE DADOS CONCLUIDA COM SUCESSO")
print("=" * 60)