"""
app.py

Interface visual do projeto, rodando no navegador.
3 categorias: Real, GPT e Gemini. Mostra a comparacao de
acuracia entre geradores direto na tela.

Para rodar: streamlit run app.py
"""

import os
import uuid
import streamlit as st
from PIL import Image
import joblib
import pandas as pd
from datetime import datetime

from extrair_features import extrair_features
from montar_dataset import montar_dataset
from treinar_modelo import treinar

PASTAS = {
    "Real": "dataset/reais",
    "GPT": "dataset/ia_gpt",
    "Gemini": "dataset/ia_gemini",
}
CAMINHO_MODELO = "modelo.pkl"

for pasta in PASTAS.values():
    os.makedirs(pasta, exist_ok=True)

st.set_page_config(page_title="Detector de Imagem Real vs IA", layout="centered")
st.title("Detector de Imagem: Real vs Gerada por IA")

aba_testar, aba_adicionar, aba_resultados = st.tabs(
    ["Testar imagem", "Adicionar ao dataset", "Resultados do treino"]
)

# ---------- ABA 1: TESTAR IMAGEM ----------
with aba_testar:
    st.subheader("Envie uma imagem pra ver a previsao do modelo")

    if not os.path.exists(CAMINHO_MODELO):
        st.warning("Ainda nao existe um modelo treinado. Treine primeiro na aba 'Adicionar ao dataset'.")
    else:
        arquivo = st.file_uploader(
            "Arraste uma imagem aqui ou clique pra selecionar",
            type=["jpg", "jpeg", "png", "jfif", "webp"],
            key="teste",
        )

        if arquivo is not None:
            caminho_temporario = "temp_teste.jpg"
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

    categoria = st.radio("Essa(s) imagem(ns) e/sao de qual categoria:", list(PASTAS.keys()), key="categoria")
    pasta_destino = PASTAS[categoria]

    arquivos_novos = st.file_uploader(
        "Arraste uma ou varias imagens aqui",
        type=["jpg", "jpeg", "png", "jfif", "webp"],
        accept_multiple_files=True,
        key="upload_dataset",
    )

    if arquivos_novos:
        if st.button(f"Salvar {len(arquivos_novos)} imagem(ns) como '{categoria}'"):
            for arquivo in arquivos_novos:
                imagem_pil = Image.open(arquivo).convert("RGB")
                nome_base, extensao = os.path.splitext(arquivo.name)
                nome_unico = f"{nome_base}_{uuid.uuid4().hex[:8]}{extensao}"
                caminho_salvar = os.path.join(pasta_destino, nome_unico)
                imagem_pil.save(caminho_salvar)
            st.success(f"{len(arquivos_novos)} imagem(ns) salva(s) em '{pasta_destino}'.")

    st.divider()

    contagens = {}
    for nome, pasta in PASTAS.items():
        contagens[nome] = len([f for f in os.listdir(pasta) if f.lower().endswith((".jpg", ".jpeg", ".png", ".jfif", ".webp"))])

    st.write(
        f"Dataset atual: "
        f"**{contagens['Real']}** reais, "
        f"**{contagens['GPT']}** GPT, "
        f"**{contagens['Gemini']}** Gemini"
    )

    if st.button("Retreinar modelo com o dataset atual"):
        if contagens["Real"] < 2 or (contagens["GPT"] + contagens["Gemini"]) < 2:
            st.error("Precisa de pelo menos 2 imagens reais e 2 de IA (somando GPT+Gemini) pra treinar.")
        else:
            with st.spinner("Montando dataset e treinando..."):
                montar_dataset(
                    pastas_com_rotulo=[
                        (PASTAS["Real"], 1, "real"),
                        (PASTAS["GPT"], 0, "gpt"),
                        (PASTAS["Gemini"], 0, "gemini"),
                    ],
                    caminho_saida="dataset.csv",
                )
                resultado = treinar("dataset.csv", CAMINHO_MODELO)
                st.session_state["ultimo_resultado"] = resultado

                linha_historico = {
                    "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "qtd_real": contagens["Real"],
                    "qtd_gpt": contagens["GPT"],
                    "qtd_gemini": contagens["Gemini"],
                    "acuracia_geral": round(resultado["acuracia_geral"] * 100, 1),
                }
                for nome_gerador, dados_gerador in resultado["acuracia_por_gerador"].items():
                    linha_historico[f"acuracia_{nome_gerador}"] = round(dados_gerador["acuracia"], 1)

                caminho_historico = "historico_resultados.csv"
                df_nova_linha = pd.DataFrame([linha_historico])
                if os.path.exists(caminho_historico):
                    df_nova_linha.to_csv(caminho_historico, mode="a", header=False, index=False)
                else:
                    df_nova_linha.to_csv(caminho_historico, mode="w", header=True, index=False)

            st.success("Modelo retreinado! Confira a aba 'Resultados do treino' pra ver a comparacao.")

# ---------- ABA 3: RESULTADOS DO TREINO ----------
with aba_resultados:
    st.subheader("Comparacao de resultados do ultimo treino")

    if "ultimo_resultado" not in st.session_state:
        st.info("Ainda nao ha resultado nessa sessao. Va na aba 'Adicionar ao dataset' e clique em 'Retreinar modelo'.")
    else:
        resultado = st.session_state["ultimo_resultado"]

        col1, col2 = st.columns(2)
        col1.metric("Acuracia geral", f"{resultado['acuracia_geral']*100:.1f}%")
        col2.metric("Tempo de treino", f"{resultado['tempo_treino']:.3f}s")

        st.write("### Acuracia por gerador")
        tabela_geradores = pd.DataFrame([
            {
                "Gerador": nome,
                "Acuracia (%)": f"{dados['acuracia']:.1f}",
                "Acertos": f"{dados['acertos']}/{dados['total']}",
            }
            for nome, dados in resultado["acuracia_por_gerador"].items()
        ])
        st.table(tabela_geradores)

        grafico_dados = pd.DataFrame({
            nome: [dados["acuracia"]]
            for nome, dados in resultado["acuracia_por_gerador"].items()
        }, index=["Acuracia (%)"]).T
        st.bar_chart(grafico_dados)

        st.write("### Importancia de cada caracteristica (feature)")
        tabela_features = pd.DataFrame(resultado["importancias"], columns=["Feature", "Importancia"])
        st.bar_chart(tabela_features.set_index("Feature"))

        st.download_button(
            "Baixar tabela de acuracia por gerador (CSV)",
            data=tabela_geradores.to_csv(index=False),
            file_name="acuracia_por_gerador.csv",
            mime="text/csv",
        )

        st.write("### Matriz de confusao")
        st.caption("Mostra onde o modelo acerta e onde confunde real com IA (e vice-versa)")
        matriz = resultado["matriz_confusao"]
        tabela_matriz = pd.DataFrame(
            matriz,
            index=["Real de fato: IA", "Real de fato: Real"],
            columns=["Previsto: IA", "Previsto: Real"],
        )
        st.table(tabela_matriz)

        with st.expander("Ver relatorio tecnico completo"):
            st.text(resultado["relatorio_texto"])

        st.divider()
        st.write("### Historico de todos os treinos (evolucao ao longo do tempo)")
        caminho_historico = "historico_resultados.csv"
        if os.path.exists(caminho_historico):
            df_historico = pd.read_csv(caminho_historico)
            st.dataframe(df_historico)
            st.download_button(
                "Baixar historico completo (CSV)",
                data=df_historico.to_csv(index=False),
                file_name="historico_resultados.csv",
                mime="text/csv",
            )
        else:
            st.info("Ainda sem historico salvo.")