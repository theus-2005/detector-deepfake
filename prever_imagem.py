"""
prever_imagem.py

Pega uma imagem nova (que o modelo NUNCA viu no treino) e diz
se ela e real ou gerada por IA, junto com a confianca da previsao.

Uso: python prever_imagem.py caminho/da/imagem.jpg
"""

import sys
import joblib
import pandas as pd
from extrair_features import extrair_features


def prever(caminho_imagem, caminho_modelo="modelo.pkl"):
    modelo = joblib.load(caminho_modelo)

    features = extrair_features(caminho_imagem)
    if features is None:
        print(f"Nao consegui abrir a imagem '{caminho_imagem}'.")
        return

    # o modelo espera as colunas na mesma ordem que foram usadas no treino,
    # entao transformamos o dicionario numa linha de tabela (DataFrame)
    X_novo = pd.DataFrame([features])

    predicao = modelo.predict(X_novo)[0]
    probabilidades = modelo.predict_proba(X_novo)[0]

    rotulo = "REAL" if predicao == 1 else "GERADA POR IA"
    confianca = probabilidades[predicao] * 100

    print(f"\nImagem: {caminho_imagem}")
    print(f"Previsao: {rotulo}")
    print(f"Confianca: {confianca:.1f}%")
    print(f"\nDetalhes:")
    print(f"  Probabilidade de ser IA:   {probabilidades[0]*100:.1f}%")
    print(f"  Probabilidade de ser real: {probabilidades[1]*100:.1f}%")

    return rotulo, confianca


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python prever_imagem.py caminho/da/imagem.jpg")
        sys.exit(1)

    prever(sys.argv[1])
