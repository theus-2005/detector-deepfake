"""
treinar_modelo.py

Le o dataset.csv (gerado pelo montar_dataset.py), treina um classificador,
e devolve os resultados (incluindo acuracia separada por gerador) tanto
pra quem rodar pelo terminal quanto pra quem usar o app.py.
"""

import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def treinar(caminho_csv="dataset.csv", caminho_modelo="modelo.pkl"):
    dados = pd.read_csv(caminho_csv)

    X = dados.drop(columns=["rotulo", "gerador"])
    y = dados["rotulo"]
    gerador = dados["gerador"]

    X_treino, X_teste, y_treino, y_teste, gerador_treino, gerador_teste = train_test_split(
        X, y, gerador, test_size=0.2, random_state=42, stratify=y
    )

    inicio = time.time()
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_treino, y_treino)
    tempo_treino = time.time() - inicio

    predicoes = modelo.predict(X_teste)
    acuracia_geral = accuracy_score(y_teste, predicoes)

    # acuracia separada por gerador
    resultado_teste = pd.DataFrame({
        "y_real": y_teste.values,
        "y_previsto": predicoes,
        "gerador": gerador_teste.values,
    })

    acuracia_por_gerador = {}
    for nome_gerador in resultado_teste["gerador"].unique():
        subgrupo = resultado_teste[resultado_teste["gerador"] == nome_gerador]
        acertos = int((subgrupo["y_real"] == subgrupo["y_previsto"]).sum())
        total = len(subgrupo)
        acuracia_por_gerador[nome_gerador] = {
            "acuracia": acertos / total * 100,
            "acertos": acertos,
            "total": total,
        }

    importancias = sorted(
        zip(X.columns, modelo.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    joblib.dump(modelo, caminho_modelo)

    resultado = {
        "tempo_treino": tempo_treino,
        "acuracia_geral": acuracia_geral,
        "acuracia_por_gerador": acuracia_por_gerador,
        "importancias": importancias,
        "relatorio_texto": classification_report(y_teste, predicoes, target_names=["IA", "Real"]),
    }

    # mostra no terminal tambem, pra quem rodar direto
    print(f"\nTempo de treino: {tempo_treino:.4f} segundos")
    print(f"Acuracia GERAL no teste: {acuracia_geral * 100:.1f}%")
    print(f"\n{resultado['relatorio_texto']}")
    print("Acuracia separada por gerador:")
    for nome, dados_gerador in acuracia_por_gerador.items():
        print(f"  {nome}: {dados_gerador['acuracia']:.1f}% ({dados_gerador['acertos']}/{dados_gerador['total']})")
    print("\nImportancia de cada feature:")
    for nome, importancia in importancias:
        print(f"  {nome}: {importancia:.4f}")
    print(f"\nModelo salvo em '{caminho_modelo}'")

    return resultado


if __name__ == "__main__":
    treinar()