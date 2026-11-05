"""
montar_dataset.py

Percorre as pastas de imagens (reais e de cada gerador de IA), extrai as
features de cada imagem e monta o dataset.csv pronto pra treinar.

MUDANCAS (feedback do professor):
  - Coluna `foto_origem`: diz de QUAL FOTO REAL cada imagem veio. A foto real
    e as 2 edicoes dela (GPT e Gemini) ficam no mesmo grupo. O treino usa
    isso pra nunca separar elas entre treino e teste (evita vazamento).
  - Colunas de metadados (largura_orig, altura_orig, tamanho_kb, formato):
    servem so pro "controle" que testa se o modelo estaria so adivinhando
    pelo tamanho/formato do arquivo. Nao entram como feature do modelo.

COMO O PROGRAMA DESCOBRE A FOTO DE ORIGEM (nessa ordem):
  1) Arquivo `dataset/grupos.csv` (opcional), com as colunas:
         arquivo,foto_origem
         IMG_2231.jpg,foto001
         ChatGPT_Image_3.png,foto001
         gemini_7.png,foto001
     (arquivo = so o nome do arquivo, com extensao)
  2) Se nao existir, usa o NOME do arquivo: tira o sufixo aleatorio que o app
     coloca (_a1b2c3d4) e as palavras gpt/chatgpt/gemini/ia/real/edit...
     Entao `foto001.jpg`, `foto001_gpt.png` e `foto001_gemini.png`
     viram todos o grupo `foto001`.

ORGANIZE ASSIM:
    dataset/
        reais/       foto001.jpg ...
        ia_gpt/      foto001_gpt.png ...
        ia_gemini/   foto001_gemini.png ...
"""

import os
import re
import csv
from collections import defaultdict

from extrair_features import extrair_features, metadados_arquivo

EXTENSOES = (".jpg", ".jpeg", ".png", ".jfif", ".webp")
PALAVRAS_GERADOR = r"(chatgpt|gpt|gemini|google|openai|ia|ai|real|reais|edit|editada|edicao|gerada|gerado|original|foto)"
LIMITE_GRUPOS_COMPLETOS = 0.9  # abaixo disso o agrupamento nao e confiavel


def chave_do_nome(nome_arquivo):
    """Descobre a foto de origem a partir do nome do arquivo."""
    base = os.path.splitext(nome_arquivo)[0].lower()
    base = re.sub(r"_[0-9a-f]{8}$", "", base)          # sufixo aleatorio do app
    base = re.sub(r"(?<![a-z0-9])" + PALAVRAS_GERADOR + r"(?![a-z0-9])", " ", base)
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base or os.path.splitext(nome_arquivo)[0].lower()


def ler_mapa_grupos(caminho_mapa):
    """Le dataset/grupos.csv (se existir). Devolve {arquivo: foto_origem}."""
    if not caminho_mapa or not os.path.exists(caminho_mapa):
        return {}
    mapa = {}
    with open(caminho_mapa, newline="", encoding="utf-8-sig") as f:
        for linha in csv.DictReader(f):
            mapa[linha["arquivo"].strip()] = linha["foto_origem"].strip()
    return mapa


def verificar_grupos(grupos_por_rotulo):
    """grupos_por_rotulo: lista de (foto_origem, nome_gerador).
    Confere se cada grupo tem exatamente 1 imagem de cada gerador.
    Devolve um dicionario-resumo."""
    por_grupo = defaultdict(list)
    for grupo, gerador in grupos_por_rotulo:
        por_grupo[grupo].append(gerador)

    geradores = sorted({g for _, g in grupos_por_rotulo})
    completos, incompletos = 0, []
    for grupo, lista in por_grupo.items():
        if sorted(lista) == geradores:
            completos += 1
        else:
            incompletos.append((grupo, sorted(lista)))

    total = len(por_grupo)
    return {
        "total_grupos": total,
        "grupos_completos": completos,
        "fracao_completos": completos / total if total else 0.0,
        "incompletos": incompletos,
        "geradores": geradores,
    }


def montar_dataset(pastas_com_rotulo, caminho_saida="dataset.csv",
                   caminho_mapa_grupos="dataset/grupos.csv", padronizar=True, recorte_fixo=True):
    """
    pastas_com_rotulo: lista de tuplas (caminho_da_pasta, rotulo, nome_gerador)
        rotulo: 1 = real, 0 = gerada por IA
        nome_gerador: "real", "gpt", "gemini"...
    Devolve o resumo da verificacao de grupos (ou None se nada foi processado).
    """
    mapa = ler_mapa_grupos(caminho_mapa_grupos)
    if mapa:
        print(f"Usando mapa de grupos '{caminho_mapa_grupos}' ({len(mapa)} arquivos).")
    else:
        print("Sem 'grupos.csv': a foto de origem sera deduzida do NOME do arquivo.")

    linhas = []
    colunas = None
    pares_grupo = []

    for pasta, rotulo, nome_gerador in pastas_com_rotulo:
        if not os.path.isdir(pasta):
            print(f"Aviso: pasta '{pasta}' nao encontrada, pulando.")
            continue

        for nome_arquivo in sorted(os.listdir(pasta)):
            if not nome_arquivo.lower().endswith(EXTENSOES):
                continue
            caminho = os.path.join(pasta, nome_arquivo)
            features = extrair_features(caminho, padronizar=padronizar, recorte_fixo=recorte_fixo)
            meta = metadados_arquivo(caminho)
            if features is None or meta is None:
                print(f"Aviso: nao consegui ler '{caminho}', pulando.")
                continue

            if colunas is None:
                colunas = list(features.keys())

            grupo = mapa.get(nome_arquivo) or chave_do_nome(nome_arquivo)
            pares_grupo.append((grupo, nome_gerador))
            linhas.append(
                [features[c] for c in colunas]
                + [rotulo, nome_gerador, grupo, nome_arquivo,
                   meta["largura_orig"], meta["altura_orig"], meta["tamanho_kb"], meta["formato"]]
            )
            print(f"Processada: {nome_arquivo} (rotulo={rotulo}, gerador={nome_gerador}, grupo={grupo})")

    if not linhas:
        print("Nenhuma imagem processada. Confira os caminhos das pastas.")
        return None

    with open(caminho_saida, "w", newline="") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(colunas + ["rotulo", "gerador", "foto_origem", "arquivo",
                                     "largura_orig", "altura_orig", "tamanho_kb", "formato"])
        escritor.writerows(linhas)

    resumo = verificar_grupos(pares_grupo)
    print(f"\nDataset salvo em '{caminho_saida}' com {len(linhas)} imagens.")
    print(f"Grupos (fotos de origem): {resumo['total_grupos']} | "
          f"completos ({'+'.join(resumo['geradores'])}): {resumo['grupos_completos']}")
    if resumo["fracao_completos"] < LIMITE_GRUPOS_COMPLETOS:
        print("ATENCAO: muitos grupos incompletos. Os nomes dos arquivos nao estao "
              "casando foto real <-> edicoes. Crie 'dataset/grupos.csv' (veja o topo "
              "deste arquivo). Exemplos de grupos incompletos:")
        for grupo, lista in resumo["incompletos"][:5]:
            print(f"   {grupo}: {lista}")
    resumo["padronizado"] = padronizar
    return resumo


if __name__ == "__main__":
    import sys
    # python montar_dataset.py               -> recorte fixo 512x512 (padrao novo)
    # python montar_dataset.py --sem-recorte -> processamento antigo (pra comparar)
    recorte = "--sem-recorte" not in sys.argv
    montar_dataset(
        pastas_com_rotulo=[
            ("dataset/reais", 1, "real"),
            ("dataset/ia_gpt", 0, "gpt"),
            ("dataset/ia_gemini", 0, "gemini"),
        ],
        caminho_saida="dataset.csv",
        recorte_fixo=recorte,
    )