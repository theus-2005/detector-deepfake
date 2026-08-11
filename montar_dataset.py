"""
montar_dataset.py

Percorre duas pastas (imagens reais e imagens geradas por IA),
extrai as features de cada imagem, e monta uma tabela (CSV)
pronta pra treinar o modelo.

ANTES DE RODAR: organize suas imagens assim:

    dataset/
        reais/
            foto1.jpg
            foto2.jpg
            ...
        ia/
            gerada1.jpg
            gerada2.jpg
            ...
"""

import os
import csv
from extrair_features import extrair_features


def montar_dataset(pasta_reais, pasta_ia, caminho_saida="dataset.csv"):
    linhas = []
    colunas = None

    # rotulo 1 = imagem real, rotulo 0 = imagem gerada por IA
    for pasta, rotulo in [(pasta_reais, 1), (pasta_ia, 0)]:
        if not os.path.isdir(pasta):
            print(f"Aviso: pasta '{pasta}' nao encontrada, pulando.")
            continue

        arquivos = [f for f in os.listdir(pasta) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(pasta, nome_arquivo)
            features = extrair_features(caminho_completo)

            if features is None:
                print(f"Aviso: nao consegui ler '{caminho_completo}', pulando.")
                continue

            if colunas is None:
                colunas = list(features.keys())

            linha = [features[c] for c in colunas] + [rotulo]
            linhas.append(linha)
            print(f"Processada: {nome_arquivo} (rotulo={rotulo})")

    if not linhas:
        print("Nenhuma imagem processada. Confira os caminhos das pastas.")
        return

    with open(caminho_saida, "w", newline="") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(colunas + ["rotulo"])
        escritor.writerows(linhas)

    print(f"\nDataset salvo em '{caminho_saida}' com {len(linhas)} imagens.")


if __name__ == "__main__":
    montar_dataset(
        pasta_reais="dataset/reais",
        pasta_ia="dataset/ia",
        caminho_saida="dataset.csv",
    )
