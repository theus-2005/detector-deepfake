"""
treinar_modelo.py

Le o dataset.csv (gerado pelo montar_dataset.py), treina um classificador,
e agora tambem mostra a acuracia SEPARADA por gerador (GPT vs Gemini),
pra comparar qual e mais facil/dificil de detectar.
"""

import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def treinar(caminho_csv="dataset.csv", caminho_modelo="modelo.pkl"):
    dados = pd.read_csv(caminho_csv)

    # X = as features (tudo menos rotulo e gerador)
    # y = o gabarito (rotulo: 1=real, 0=IA)
    # gerador fica separado, so pra analise depois -- nao entra no treino
    X = dados.drop(columns=["rotulo", "gerador"])
    y = dados["rotulo"]
    gerador = dados["gerador"]

    # dividimos tudo junto (X, y, gerador) com os MESMOS indices,
    # assim sabemos de qual gerador cada imagem de teste veio
    X_treino, X_teste, y_treino, y_teste, gerador_treino, gerador_teste = train_test_split(
        X, y, gerador, test_size=0.2, random_state=42, stratify=y
    )

    inicio = time.time()
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_treino, y_treino)
    fim = time.time()

    predicoes = modelo.predict(X_teste)
    acuracia_geral = accuracy_score(y_teste, predicoes)

    print(f"\nTempo de treino: {fim - inicio:.4f} segundos")
    print(f"Acuracia GERAL no teste: {acuracia_geral * 100:.1f}%")
    print(f"\nRelatorio detalhado:")
    print(classification_report(y_teste, predicoes, target_names=["IA", "Real"]))

    # ---- NOVO: acuracia separada por gerador ----
    print("Acuracia separada por gerador:")
    resultado_teste = pd.DataFrame({
        "y_real": y_teste.values,
        "y_previsto": predicoes,
        "gerador": gerador_teste.values,
    })

    for nome_gerador in resultado_teste["gerador"].unique():
        subgrupo = resultado_teste[resultado_teste["gerador"] == nome_gerador]
        acertos = (subgrupo["y_real"] == subgrupo["y_previsto"]).sum()
        total = len(subgrupo)
        acuracia_grupo = acertos / total * 100
        print(f"  {nome_gerador}: {acuracia_grupo:.1f}% ({acertos}/{total} imagens no teste)")

    # importancia de cada feature
    importancias = sorted(
        zip(X.columns, modelo.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    print("\nImportancia de cada feature (da mais util pra menos util):")
    for nome, importancia in importancias:
        print(f"  {nome}: {importancia:.4f}")

    joblib.dump(modelo, caminho_modelo)
    print(f"\nModelo salvo em '{caminho_modelo}'")

    return modelo


if __name__ == "__main__":
    treinar()