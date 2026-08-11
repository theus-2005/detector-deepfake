"""
treinar_modelo.py

Le o dataset.csv (gerado pelo montar_dataset.py) e treina
um classificador pra distinguir imagem real de imagem gerada por IA.
"""

import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def treinar(caminho_csv="dataset.csv", caminho_modelo="modelo.pkl"):
    dados = pd.read_csv(caminho_csv)

    # X = todas as colunas menos "rotulo" (as features)
    # y = só a coluna "rotulo" (o gabarito: 1=real, 0=IA)
    X = dados.drop(columns=["rotulo"])
    y = dados["rotulo"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    inicio = time.time()
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_treino, y_treino)
    fim = time.time()

    predicoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, predicoes)

    print(f"\nTempo de treino: {fim - inicio:.4f} segundos")
    print(f"Acuracia no teste: {acuracia * 100:.1f}%")
    print(f"\nRelatorio detalhado:")
    print(classification_report(y_teste, predicoes, target_names=["IA", "Real"]))

    # importancia de cada feature -- mostra quais numeros mais ajudaram
    # o modelo a decidir. Bom material pro seu artigo!
    importancias = sorted(
        zip(X.columns, modelo.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    print("Importancia de cada feature (da mais util pra menos util):")
    for nome, importancia in importancias:
        print(f"  {nome}: {importancia:.4f}")

    joblib.dump(modelo, caminho_modelo)
    print(f"\nModelo salvo em '{caminho_modelo}'")

    return modelo


if __name__ == "__main__":
    treinar()
