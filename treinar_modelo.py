"""
treinar_modelo.py

Le o dataset.csv, avalia 4 algoritmos (Random Forest, Gradient Boosting,
SVM, Regressao Logistica) e salva o melhor como modelo final.

MUDANCAS (feedback do professor):
  1) Validacao cruzada AGRUPADA por foto de origem (StratifiedGroupKFold):
     a foto real e suas edicoes de IA nunca ficam separadas entre treino e
     teste -> acaba o vazamento de dados (data leakage).
  2) Alem da acuracia: precisao, revocacao e F1 de cada classe, F1 macro e
     ACURACIA BALANCEADA (comparavel com a linha de base de 50%).
  3) Os 4 algoritmos recebem o MESMO tratamento de desbalanceamento
     (peso balanceado das classes). No Gradient Boosting isso e feito com
     sample_weight, ja que ele nao tem o parametro class_weight.
  4) CONTROLE de confusao: um classificador que ve SO metadados do arquivo
     (largura, altura, tamanho, formato). Se ele acertar bem, o modelo
     principal pode estar aprendendo o formato e nao a edicao por IA.
  5) O modelo final salvo NAO e mais fixo: e o algoritmo de maior acuracia
     balanceada na validacao agrupada (o criterio e registrado em
     modelo_info.json). O RF continua sendo usado pra mostrar a importancia
     das features.
"""

import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

COLUNAS_NAO_FEATURE = ["rotulo", "gerador", "foto_origem", "arquivo",
                       "largura_orig", "altura_orig", "tamanho_kb", "formato"]
COLUNAS_METADADOS = ["largura_orig", "altura_orig", "tamanho_kb"]  # + formato (one-hot)

# Minimos aceitaveis (sugestao -- ajuste e justifique no documento do projeto).
# Linha de base: classificador trivial "tudo e IA" = 66,7% de acuracia e
# 50% de acuracia balanceada.
LIMITES_MINIMOS = {
    "acuracia_balanceada": 0.60,
    "revocacao_real": 0.50,
    "f1_macro": 0.55,
}
LINHA_BASE_ACURACIA = 2 / 3
LINHA_BASE_BALANCEADA = 0.5


def construir_algoritmos():
    """4 algoritmos, todos com peso balanceado das classes.
    (No Gradient Boosting o peso entra via sample_weight no .fit -- ver
    `ajustar`.) SVM e Regressao Logistica usam StandardScaler."""
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True, random_state=42)),
        "Regressao Logistica": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    }


def ajustar(nome, modelo, X, y):
    """Treina o modelo. So o Gradient Boosting precisa de sample_weight
    (os outros ja fazem o balanceamento via class_weight)."""
    if nome == "Gradient Boosting":
        modelo.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        modelo.fit(X, y)
    return modelo


def _metricas(y_real, y_prev):
    precisao, revocacao, f1, _ = precision_recall_fscore_support(
        y_real, y_prev, labels=[0, 1], zero_division=0
    )
    return {
        "acuracia_balanceada": float(balanced_accuracy_score(y_real, y_prev)),
        "f1_macro": float(np.mean(f1)),
        "por_classe": {
            "IA": {"precisao": float(precisao[0]), "revocacao": float(revocacao[0]), "f1": float(f1[0])},
            "Real": {"precisao": float(precisao[1]), "revocacao": float(revocacao[1]), "f1": float(f1[1])},
        },
        "revocacao_real": float(revocacao[1]),
    }


def validacao_cruzada(nome, modelo_base, X, y, splits):
    """Faz a validacao cruzada manualmente (necessario pra passar o
    sample_weight ao GB) e devolve as predicoes de todos os folds."""
    predicoes = np.zeros(len(y), dtype=int)
    acuracias_fold = []
    for idx_treino, idx_teste in splits:
        modelo = ajustar(nome, clone(modelo_base), X.iloc[idx_treino], y.iloc[idx_treino])
        prev = modelo.predict(X.iloc[idx_teste])
        predicoes[idx_teste] = prev
        acuracias_fold.append(accuracy_score(y.iloc[idx_teste], prev))
    return predicoes, acuracias_fold


def avaliar_algoritmo(nome, modelo, X, y, gerador, splits):
    inicio = time.time()
    predicoes, acc_folds = validacao_cruzada(nome, modelo, X, y, splits)
    tempo = time.time() - inicio

    tabela = pd.DataFrame({"y_real": y.values, "y_previsto": predicoes, "gerador": gerador.values})

    acuracia_por_gerador = {}
    for nome_gerador in tabela["gerador"].unique():
        sub = tabela[tabela["gerador"] == nome_gerador]
        acertos = int((sub["y_real"] == sub["y_previsto"]).sum())
        acuracia_por_gerador[nome_gerador] = {
            "acuracia": acertos / len(sub) * 100, "acertos": acertos, "total": len(sub),
        }

    resultado = {
        "tempo": tempo,
        "cv_media": float(np.mean(acc_folds)),
        "cv_desvio_padrao": float(np.std(acc_folds)),
        "acuracia_geral": float(accuracy_score(tabela["y_real"], tabela["y_previsto"])),
        "acuracia_por_gerador": acuracia_por_gerador,
        "matriz_confusao": confusion_matrix(tabela["y_real"], tabela["y_previsto"], labels=[0, 1]).tolist(),
        "relatorio_texto": classification_report(
            tabela["y_real"], tabela["y_previsto"], target_names=["IA", "Real"], zero_division=0
        ),
    }
    resultado.update(_metricas(tabela["y_real"], tabela["y_previsto"]))
    return resultado


def _controle_metadados(dados, y, splits):
    """Classificador que enxerga SO metadados do arquivo. Se ele ficar bem
    acima de 50% de acuracia balanceada, formato/tamanho entregam a classe."""
    if not all(c in dados.columns for c in COLUNAS_METADADOS + ["formato"]):
        return None
    Xm = pd.concat(
        [dados[COLUNAS_METADADOS], pd.get_dummies(dados["formato"], prefix="fmt").astype(int)], axis=1
    )
    modelo = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    pred, _ = validacao_cruzada("Random Forest", modelo, Xm, y, splits)
    res = {"acuracia_geral": float(accuracy_score(y, pred))}
    res.update(_metricas(y, pred))
    return res


def _criar_splits(X, y, grupos, k_folds):
    """StratifiedGroupKFold agrupado por foto. Sem coluna de grupo, cai pra
    StratifiedKFold e AVISA (resultado pode estar inflado)."""
    if grupos is not None and grupos.nunique() >= 2:
        k = max(2, min(k_folds, grupos.nunique()))
        validador = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
        return list(validador.split(X, y, groups=grupos)), k, True
    k = max(2, min(k_folds, int(y.value_counts().min())))
    validador = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return list(validador.split(X, y)), k, False


def treinar(caminho_csv="dataset.csv", caminho_modelo="modelo.pkl", k_folds=5,
            caminho_info="modelo_info.json"):
    dados = pd.read_csv(caminho_csv)

    X = dados.drop(columns=[c for c in COLUNAS_NAO_FEATURE if c in dados.columns])
    y = dados["rotulo"]
    gerador = dados["gerador"]
    grupos = dados["foto_origem"] if "foto_origem" in dados.columns else None

    splits, k, agrupado = _criar_splits(X, y, grupos, k_folds)

    aviso = None
    if not agrupado:
        aviso = ("Sem coluna 'foto_origem': validacao NAO agrupada, resultado pode estar inflado. "
                 "Rode montar_dataset.py de novo.")
    else:
        # grupos que misturam real + IA = agrupamento funcionando
        mistos = dados.groupby("foto_origem")["rotulo"].nunique()
        if (mistos == 2).mean() < 0.9:
            aviso = ("Menos de 90% dos grupos tem foto real + edicao de IA. Os nomes dos arquivos "
                     "provavelmente nao casam; crie dataset/grupos.csv (veja montar_dataset.py).")
    if aviso:
        print(f"ATENCAO: {aviso}\n")

    algoritmos = construir_algoritmos()
    resultados = {}
    print(f"Avaliando {len(algoritmos)} algoritmos | validacao cruzada de {k} folds "
          f"{'AGRUPADA por foto' if agrupado else '(sem grupos)'} | {len(X)} imagens\n")

    for nome, modelo in algoritmos.items():
        r = avaliar_algoritmo(nome, modelo, X, y, gerador, splits)
        resultados[nome] = r
        print(f"--- {nome} --- acuracia {r['acuracia_geral']*100:.1f}% | "
              f"acur. balanceada {r['acuracia_balanceada']*100:.1f}% | F1 macro {r['f1_macro']*100:.1f}% | "
              f"revocacao Real {r['revocacao_real']*100:.1f}% | {r['tempo']:.1f}s")
        for g, d in r["acuracia_por_gerador"].items():
            print(f"    {g}: {d['acuracia']:.1f}% ({d['acertos']}/{d['total']})")

    # ---- medias entre algoritmos ----
    todos_geradores = {g for r in resultados.values() for g in r["acuracia_por_gerador"]}
    media_por_gerador = {
        g: float(np.mean([r["acuracia_por_gerador"][g]["acuracia"] for r in resultados.values()
                          if g in r["acuracia_por_gerador"]]))
        for g in todos_geradores
    }
    media_geral = float(np.mean([r["acuracia_geral"] for r in resultados.values()]))
    media_balanceada = float(np.mean([r["acuracia_balanceada"] for r in resultados.values()]))

    # ---- controle de metadados ----
    controle = _controle_metadados(dados, y, splits)
    if controle:
        print(f"\nCONTROLE (so metadados: tamanho/formato): acuracia {controle['acuracia_geral']*100:.1f}% | "
              f"acur. balanceada {controle['acuracia_balanceada']*100:.1f}%")
        if controle["acuracia_balanceada"] >= 0.60:
            print("  -> LIMITACAO DO DATASET: os arquivos originais (formato/tamanho) ja separam as classes. "
                  "Isso mede o ARQUIVO ORIGINAL e nao muda com o recorte/padronizacao das features. "
                  "Registre como limitacao; compare com o modelo principal (ele nao deve depender disso).")

    # ---- escolha do modelo final: maior acuracia balanceada (desempate: F1 macro) ----
    melhor = max(resultados, key=lambda n: (resultados[n]["acuracia_balanceada"], resultados[n]["f1_macro"]))
    modelo_final = ajustar(melhor, clone(algoritmos[melhor]), X, y)
    joblib.dump(modelo_final, caminho_modelo)

    # importancia das features (sempre via Random Forest, treinado em 100%)
    rf = ajustar("Random Forest", clone(algoritmos["Random Forest"]), X, y)
    importancias = sorted(zip(X.columns, rf.feature_importances_), key=lambda i: i[1], reverse=True)

    atende = {m: resultados[melhor][m] >= v for m, v in LIMITES_MINIMOS.items()}
    info = {
        "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo_final": melhor,
        "criterio_escolha": "maior acuracia balanceada na validacao cruzada agrupada (desempate: F1 macro)",
        "validacao": f"StratifiedGroupKFold k={k}" if agrupado else f"StratifiedKFold k={k} (SEM grupos)",
        "n_imagens": int(len(X)),
        "features": list(X.columns),
        "metricas_modelo_final": {m: resultados[melhor][m] for m in
                                  ["acuracia_geral", "acuracia_balanceada", "f1_macro", "revocacao_real"]},
        "limites_minimos": LIMITES_MINIMOS,
        "atende_limites": atende,
        "controle_metadados": controle,
        "aviso": aviso,
    }
    with open(caminho_info, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nModelo final: {melhor} (maior acuracia balanceada) salvo em '{caminho_modelo}'.")
    print("Obs.: o modelo foi escolhido com os mesmos dados da validacao; a estimativa dele e "
          "levemente otimista. Detalhes em '" + caminho_info + "'.")

    return {
        "resultados_por_algoritmo": resultados,
        "media_por_gerador": media_por_gerador,
        "media_geral": media_geral,
        "media_balanceada": media_balanceada,
        "importancias": importancias,
        "cv_k_folds": k,
        "agrupado": agrupado,
        "aviso": aviso,
        "controle_metadados": controle,
        "modelo_final": melhor,
        "atende_limites": atende,
        "limites_minimos": LIMITES_MINIMOS,
    }


if __name__ == "__main__":
    treinar()