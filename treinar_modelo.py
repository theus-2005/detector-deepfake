"""
treinar_modelo.py

Le o dataset.csv, treina VARIOS algoritmos diferentes (Random Forest,
SVM, Gradient Boosting, Regressao Logistica), e devolve os resultados
de cada um separadamente -- acuracia geral, por gerador, matriz de
confusao -- alem da media entre todos os algoritmos.

O Random Forest continua sendo o modelo FINAL salvo (modelo.pkl),
usado pra prever imagens novas no app.
"""

import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def construir_algoritmos():
    """Devolve um dicionario {nome: modelo} com os 4 algoritmos a comparar.

    SVM e Regressao Logistica sao sensiveis a escala dos numeros de entrada,
    entao usamos um StandardScaler (padroniza os numeros) antes deles --
    isso nao muda a informacao, so evita que uma feature com numeros
    grandes (tipo hist_b_media, que vai ate 255) domine uma feature com
    numeros pequenos (tipo ruido) so por causa da escala.
    """
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=42)),
        "Regressao Logistica": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    }


def avaliar_algoritmo(modelo, X, y, gerador, validador):
    """Treina/avalia UM algoritmo via validacao cruzada e devolve
    um dicionario com todos os resultados dele."""
    inicio = time.time()
    scores_cv = cross_val_score(modelo, X, y, cv=validador, scoring="accuracy")
    predicoes_cv = cross_val_predict(modelo, X, y, cv=validador)
    tempo = time.time() - inicio

    resultado_completo = pd.DataFrame({
        "y_real": y.values,
        "y_previsto": predicoes_cv,
        "gerador": gerador.values,
    })

    acuracia_por_gerador = {}
    for nome_gerador in resultado_completo["gerador"].unique():
        subgrupo = resultado_completo[resultado_completo["gerador"] == nome_gerador]
        acertos = int((subgrupo["y_real"] == subgrupo["y_previsto"]).sum())
        total = len(subgrupo)
        acuracia_por_gerador[nome_gerador] = {
            "acuracia": acertos / total * 100,
            "acertos": acertos,
            "total": total,
        }

    matriz_confusao = confusion_matrix(
        resultado_completo["y_real"], resultado_completo["y_previsto"], labels=[0, 1]
    ).tolist()

    return {
        "tempo": tempo,
        "cv_media": scores_cv.mean(),
        "cv_desvio_padrao": scores_cv.std(),
        "acuracia_geral": accuracy_score(resultado_completo["y_real"], resultado_completo["y_previsto"]),
        "acuracia_por_gerador": acuracia_por_gerador,
        "matriz_confusao": matriz_confusao,
        "relatorio_texto": classification_report(
            resultado_completo["y_real"], resultado_completo["y_previsto"], target_names=["IA", "Real"]
        ),
    }


def treinar(caminho_csv="dataset.csv", caminho_modelo="modelo.pkl", k_folds=5):
    dados = pd.read_csv(caminho_csv)

    X = dados.drop(columns=["rotulo", "gerador"])
    y = dados["rotulo"]
    gerador = dados["gerador"]

    menor_classe = y.value_counts().min()
    k_folds_ajustado = min(k_folds, menor_classe)
    validador = StratifiedKFold(n_splits=k_folds_ajustado, shuffle=True, random_state=42)

    algoritmos = construir_algoritmos()
    resultados_por_algoritmo = {}

    print(f"Avaliando {len(algoritmos)} algoritmos com validacao cruzada de {k_folds_ajustado} folds...\n")

    for nome, modelo in algoritmos.items():
        print(f"--- {nome} ---")
        resultado_alg = avaliar_algoritmo(modelo, X, y, gerador, validador)
        resultados_por_algoritmo[nome] = resultado_alg

        print(f"Tempo: {resultado_alg['tempo']:.4f}s | Acuracia geral: {resultado_alg['acuracia_geral']*100:.1f}%")
        for nome_gerador, dados_gerador in resultado_alg["acuracia_por_gerador"].items():
            print(f"  {nome_gerador}: {dados_gerador['acuracia']:.1f}% ({dados_gerador['acertos']}/{dados_gerador['total']})")
        print()

    # ---- MEDIA entre todos os algoritmos, por gerador ----
    todos_geradores = set()
    for resultado_alg in resultados_por_algoritmo.values():
        todos_geradores.update(resultado_alg["acuracia_por_gerador"].keys())

    media_por_gerador = {}
    for nome_gerador in todos_geradores:
        valores = [
            resultado_alg["acuracia_por_gerador"][nome_gerador]["acuracia"]
            for resultado_alg in resultados_por_algoritmo.values()
            if nome_gerador in resultado_alg["acuracia_por_gerador"]
        ]
        media_por_gerador[nome_gerador] = sum(valores) / len(valores)

    media_geral = sum(r["acuracia_geral"] for r in resultados_por_algoritmo.values()) / len(resultados_por_algoritmo)

    print("--- MEDIA ENTRE TODOS OS ALGORITMOS ---")
    print(f"Acuracia geral media: {media_geral*100:.1f}%")
    for nome_gerador, media in media_por_gerador.items():
        print(f"  {nome_gerador}: {media:.1f}%")

    # ---- MODELO FINAL salvo (Random Forest, treinado com 100% dos dados) ----
    modelo_final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    modelo_final.fit(X, y)
    importancias = sorted(
        zip(X.columns, modelo_final.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    joblib.dump(modelo_final, caminho_modelo)

    print(f"\nModelo final (Random Forest, 100% dos dados) salvo em '{caminho_modelo}'")

    return {
        "resultados_por_algoritmo": resultados_por_algoritmo,
        "media_por_gerador": media_por_gerador,
        "media_geral": media_geral,
        "importancias": importancias,
        "cv_k_folds": k_folds_ajustado,
    }


if __name__ == "__main__":
    treinar()