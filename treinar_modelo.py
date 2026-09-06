"""
treinar_modelo.py

Le o dataset.csv, treina um classificador, e devolve os resultados
(incluindo acuracia por gerador calculada via validacao cruzada,
para ser confiavel mesmo com poucas imagens).
"""

import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def treinar(caminho_csv="dataset.csv", caminho_modelo="modelo.pkl", k_folds=5):
    dados = pd.read_csv(caminho_csv)

    X = dados.drop(columns=["rotulo", "gerador"])
    y = dados["rotulo"]
    gerador = dados["gerador"]

    # k_folds nao pode ser maior que a quantidade de imagens da menor classe
    menor_classe = y.value_counts().min()
    k_folds_ajustado = min(k_folds, menor_classe)

    modelo_cv = RandomForestClassifier(n_estimators=200, random_state=42)
    validador = StratifiedKFold(n_splits=k_folds_ajustado, shuffle=True, random_state=42)

    # acuracia geral via validacao cruzada (media dos folds)
    inicio = time.time()
    scores_cv = cross_val_score(modelo_cv, X, y, cv=validador, scoring="accuracy")
    tempo_treino = time.time() - inicio

    # ---- ACURACIA POR GERADOR, usando TODAS as imagens ----
    # cross_val_predict devolve uma previsao pra CADA imagem do dataset,
    # sempre usando um modelo que nao viu aquela imagem durante o treino
    # (assim nao ha "cola", mas aproveitamos 100% dos dados pra avaliar,
    # em vez de so os 20% que sobrariam num split unico)
    predicoes_cv = cross_val_predict(modelo_cv, X, y, cv=validador)

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

    acuracia_geral = accuracy_score(resultado_completo["y_real"], resultado_completo["y_previsto"])

    # ---- MODELO FINAL, treinado com TODOS os dados (pra salvar e usar depois) ----
    modelo_final = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo_final.fit(X, y)

    importancias = sorted(
        zip(X.columns, modelo_final.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    joblib.dump(modelo_final, caminho_modelo)

    relatorio_texto = classification_report(
        resultado_completo["y_real"], resultado_completo["y_previsto"], target_names=["IA", "Real"]
    )

    matriz_confusao = confusion_matrix(
        resultado_completo["y_real"], resultado_completo["y_previsto"], labels=[0, 1]
    ).tolist()
    # matriz_confusao[i][j] = quantidade de imagens que sao DE VERDADE a classe i
    # mas foram PREVISTAS como classe j. labels=[0,1] -> 0=IA, 1=Real

    resultado = {
        "tempo_treino": tempo_treino,
        "acuracia_geral": acuracia_geral,
        "acuracia_por_gerador": acuracia_por_gerador,
        "importancias": importancias,
        "relatorio_texto": relatorio_texto,
        "cv_k_folds": k_folds_ajustado,
        "cv_scores": scores_cv.tolist(),
        "cv_media": scores_cv.mean(),
        "cv_desvio_padrao": scores_cv.std(),
        "matriz_confusao": matriz_confusao,
    }

    print(f"\nTempo total (validacao cruzada): {tempo_treino:.4f} segundos")
    print(f"\n--- Validacao cruzada ({k_folds_ajustado}-fold) ---")
    print(f"Acuracias em cada fold: {[f'{s*100:.1f}%' for s in scores_cv]}")
    print(f"Media: {scores_cv.mean()*100:.1f}% (desvio padrao: {scores_cv.std()*100:.1f}%)")
    print(f"\nAcuracia GERAL (usando todas as imagens via validacao cruzada): {acuracia_geral*100:.1f}%")
    print(f"\n{relatorio_texto}")
    print("Acuracia separada por gerador (usando TODAS as imagens de cada um):")
    for nome, dados_gerador in acuracia_por_gerador.items():
        print(f"  {nome}: {dados_gerador['acuracia']:.1f}% ({dados_gerador['acertos']}/{dados_gerador['total']})")
    print("\nMatriz de confusao:")
    print(f"                  Previsto: IA   Previsto: Real")
    print(f"  Real de fato: IA      {matriz_confusao[0][0]:^12} {matriz_confusao[0][1]:^12}")
    print(f"  Real de fato: Real    {matriz_confusao[1][0]:^12} {matriz_confusao[1][1]:^12}")
    print("\nImportancia de cada feature:")
    for nome, importancia in importancias:
        print(f"  {nome}: {importancia:.4f}")
    print(f"\nModelo salvo em '{caminho_modelo}' (treinado com 100% dos dados)")

    return resultado


if __name__ == "__main__":
    treinar()