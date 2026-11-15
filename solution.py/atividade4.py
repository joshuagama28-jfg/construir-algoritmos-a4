
import pandas as pd
import numpy as np

def calcular_coeficientes(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)

    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    n = len(preco)

    X = np.array([[1, area[i], quartos[i]] for i in range(n)], dtype=float)
    y = np.array(preco, dtype=float).reshape(-1, 1)

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return beta.flatten().tolist()


if __name__ == "__main__":
    resultado = calcular_coeficientes("imoveis_lar_perfeito.csv")

    beta_0, beta_1, beta_2 = resultado

    print(f"Preco = {beta_0:.2f} + {beta_1:.2f} * Area + {beta_2:.2f} * Quartos")
