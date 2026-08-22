"""
app.py

Interface visual do projeto, rodando no navegador local.
Tem duas partes:
  1. Testar imagem: arrasta/seleciona uma imagem e ve a previsao
  2. Adicionar ao dataset: manda novas imagens pra treino (real ou IA)
     e um botao pra retreinar o modelo com elas

Para rodar: streamlit run app.py
"""

import os
import streamlit as st
from PIL import Image
import joblib
import pandas as pd

from extrair_features import extrair_features
from montar_dataset import montar_dataset
from treinar_modelo import treinar

PASTA_REAIS = "dataset/reais"
PASTA_IA = "dataset/ia"
CAMINHO_MODELO = "modelo.pkl"

os.makedirs(PASTA_REAIS, exist_ok=True)
os.makedirs(PASTA_IA, exist_ok=True)

st.set_page_config(page_title="Detector de Imagem Real vs IA", layout="centered")
st.title("Detector de Imagem: Real vs Gerada por IA")

aba_testar, aba_adicionar = st.tabs(["Testar imagem", "Adicionar ao dataset"])

# ---------- ABA 1: TESTAR IMAGEM ----------
with aba_testar:
    st.subheader("Envie uma imagem pra ver a previsao do modelo")

    if not os.path.exists(CAMINHO_MODELO):
        st.warning("Ainda nao existe um modelo treinado (modelo.pkl). Treine primeiro na aba 'Adicionar ao dataset'.")
    else:
        arquivo = st.file_uploader(
            "Arraste uma imagem aqui ou clique pra selecionar",
            type=["jpg", "jpeg", "png"],
            key="teste",
        )

        if arquivo is not None:
            caminho_temporario = os.path.join("temp_teste.jpg")
            imagem_pil = Image.open(arquivo).convert("RGB")
            imagem_pil.save(caminho_temporario)

            st.image(imagem_pil, caption="Imagem enviada", width=300)

            modelo = joblib.load(CAMINHO_MODELO)
            features = extrair_features(caminho_temporario)

            if features is None:
                st.error("Nao consegui processar essa imagem.")
            else:
                X_novo = pd.DataFrame([features])
                predicao = modelo.predict(X_novo)[0]
                probabilidades = modelo.predict_proba(X_novo)[0]

                rotulo = "REAL" if predicao == 1 else "GERADA POR IA"
                confianca = probabilidades[predicao] * 100

                if predicao == 1:
                    st.success(f"Previsao: {rotulo} ({confianca:.1f}% de confianca)")
                else:
                    st.error(f"Previsao: {rotulo} ({confianca:.1f}% de confianca)")

                st.write(f"Probabilidade de ser IA: {probabilidades[0]*100:.1f}%")
                st.write(f"Probabilidade de ser real: {probabilidades[1]*100:.1f}%")

            os.remove(caminho_temporario)

# ---------- ABA 2: ADICIONAR AO DATASET ----------
with aba_adicionar:
    st.subheader("Adicionar novas imagens ao dataset de treino")

    classe = st.radio("Essa(s) imagem(ns) e/sao:", ["Real", "Gerada por IA"], key="classe")
    pasta_destino = PASTA_REAIS if classe == "Real" else PASTA_IA

    arquivos_novos = st.file_uploader(
        "Arraste uma ou varias imagens aqui",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="upload_dataset",
    )

    if arquivos_novos:
        if st.button(f"Salvar {len(arquivos_novos)} imagem(ns) como '{classe}'"):
            for arquivo in arquivos_novos:
                imagem_pil = Image.open(arquivo).convert("RGB")
                caminho_salvar = os.path.join(pasta_destino, arquivo.name)
                imagem_pil.save(caminho_salvar)
            st.success(f"{len(arquivos_novos)} imagem(ns) salva(s) em '{pasta_destino}'.")

    st.divider()

    qtd_reais = len([f for f in os.listdir(PASTA_REAIS) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    qtd_ia = len([f for f in os.listdir(PASTA_IA) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    st.write(f"Dataset atual: **{qtd_reais}** imagens reais, **{qtd_ia}** imagens de IA")

    if st.button("Retreinar modelo com o dataset atual"):
        if qtd_reais < 2 or qtd_ia < 2:
            st.error("Precisa de pelo menos 2 imagens de cada classe pra treinar.")
        else:
            with st.spinner("Montando dataset e treinando..."):
                montar_dataset(PASTA_REAIS, PASTA_IA, "dataset.csv")
                modelo = treinar("dataset.csv", CAMINHO_MODELO)
            st.success("Modelo retreinado com sucesso! Va na aba 'Testar imagem' pra experimentar.")