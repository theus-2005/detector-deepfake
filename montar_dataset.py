"""
montar_dataset.py

Percorre varias pastas de imagens (reais e de diferentes geradores de IA),
extrai as features de cada imagem, e monta uma tabela (CSV) pronta pra
treinar o modelo. Agora tambem guarda QUAL GERADOR foi usado em cada
imagem, pra dar pra comparar GPT vs Gemini depois.

ANTES DE RODAR: organize suas imagens assim:

    dataset/
        reais/
            foto1.jpg
            foto2.jpg
            ...
        ia_gpt/
            gerada1.jpg
            ...
        ia_gemini/
            gerada1.jpg
            ...
"""

import os
import csv
from extrair_features import extrair_features


def montar_dataset(pastas_com_rotulo, caminho_saida="dataset.csv"):
    """
    pastas_com_rotulo: lista de tuplas (caminho_da_pasta, rotulo, nome_gerador)
        rotulo: 1 = real, 0 = gerada por IA
        nome_gerador: string livre, tipo "real", "gpt", "gemini"
    """
    linhas = []
    colunas = None

    for pasta, rotulo, nome_gerador in pastas_com_rotulo:
        if not os.path.isdir(pasta):
            print(f"Aviso: pasta '{pasta}' nao encontrada, pulando.")
            continue

        arquivos = [f for f in os.listdir(pasta) if f.lower().endswith((".jpg", ".jpeg", ".png", ".jfif", ".webp"))]

        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(pasta, nome_arquivo)
            features = extrair_features(caminho_completo)

            if features is None:
                print(f"Aviso: nao consegui ler '{caminho_completo}', pulando.")
                continue

            if colunas is None:
                colunas = list(features.keys())

            linha = [features[c] for c in colunas] + [rotulo, nome_gerador]
            linhas.append(linha)
            print(f"Processada: {nome_arquivo} (rotulo={rotulo}, gerador={nome_gerador})")

    if not linhas:
        print("Nenhuma imagem processada. Confira os caminhos das pastas.")
        return

    with open(caminho_saida, "w", newline="") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(colunas + ["rotulo", "gerador"])
        escritor.writerows(linhas)

    print(f"\nDataset salvo em '{caminho_saida}' com {len(linhas)} imagens.")


if __name__ == "__main__":
    montar_dataset(
        pastas_com_rotulo=[
            ("dataset/reais", 1, "real"),
            ("dataset/ia_gpt", 0, "gpt"),
            ("dataset/ia_gemini", 0, "gemini"),
        ],
        caminho_saida="dataset.csv",
    )