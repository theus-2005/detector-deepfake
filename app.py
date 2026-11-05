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
            caminho_temporario = "temp_teste.png"
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

    st.caption("Dica: use o MESMO nome-base pra foto real e suas edicoes (ex.: foto001.jpg, "
               "foto001_gpt.png, foto001_gemini.png). Isso agrupa as 3 na validacao.")
    arquivos_novos = st.file_uploader(
        "Arraste uma ou varias imagens aqui",
        type=["jpg", "jpeg", "png", "jfif", "webp"],
        accept_multiple_files=True,
        key="upload_dataset",
    )

    if arquivos_novos:
        if st.button(f"Salvar {len(arquivos_novos)} imagem(ns) como '{categoria}'"):
            for arquivo in arquivos_novos:
                nome_base, extensao = os.path.splitext(arquivo.name)
                nome_unico = f"{nome_base}_{uuid.uuid4().hex[:8]}{extensao}"
                caminho_salvar = os.path.join(pasta_destino, nome_unico)
                # grava os bytes ORIGINAIS (sem recomprimir), pra nao mexer na compressao do arquivo
                with open(caminho_salvar, "wb") as destino:
                    destino.write(arquivo.getvalue())
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
                resumo_grupos = montar_dataset(
                    pastas_com_rotulo=[
                        (PASTAS["Real"], 1, "real"),
                        (PASTAS["GPT"], 0, "gpt"),
                        (PASTAS["Gemini"], 0, "gemini"),
                    ],
                    caminho_saida="dataset.csv",
                )
                resultado = treinar("dataset.csv", CAMINHO_MODELO)
                resultado["resumo_grupos"] = resumo_grupos
                st.session_state["ultimo_resultado"] = resultado

                linha_historico = {
                    "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "qtd_real": contagens["Real"],
                    "qtd_gpt": contagens["GPT"],
                    "qtd_gemini": contagens["Gemini"],
                    "acuracia_media_geral": round(resultado["media_geral"] * 100, 1),
                    "acuracia_balanceada_media": round(resultado["media_balanceada"] * 100, 1),
                    "validacao": "agrupada por foto" if resultado["agrupado"] else "NAO agrupada",
                    "modelo_final": resultado["modelo_final"],
                }
                for nome_gerador, media in resultado["media_por_gerador"].items():
                    linha_historico[f"acuracia_media_{nome_gerador}"] = round(media, 1)
                for nome_algoritmo, dados_alg in resultado["resultados_por_algoritmo"].items():
                    linha_historico[f"acuracia_{nome_algoritmo}"] = round(dados_alg["acuracia_geral"] * 100, 1)
                    linha_historico[f"balanceada_{nome_algoritmo}"] = round(dados_alg["acuracia_balanceada"] * 100, 1)
                    linha_historico[f"f1_{nome_algoritmo}"] = round(dados_alg["f1_macro"] * 100, 1)

                caminho_historico = "historico_resultados.csv"
                df_nova_linha = pd.DataFrame([linha_historico])

                formato_mudou = False
                if os.path.exists(caminho_historico):
                    try:
                        colunas_antigas = list(pd.read_csv(caminho_historico, nrows=0).columns)
                        if colunas_antigas != list(df_nova_linha.columns):
                            formato_mudou = True
                    except Exception:
                        formato_mudou = True

                if formato_mudou:
                    marca_tempo = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_backup = f"historico_resultados_antigo_{marca_tempo}.csv"
                    os.rename(caminho_historico, nome_backup)
                    st.info(f"O formato do historico mudou -- o arquivo antigo foi salvo como '{nome_backup}', e um novo historico foi iniciado.")

                if os.path.exists(caminho_historico):
                    df_nova_linha.to_csv(caminho_historico, mode="a", header=False, index=False)
                else:
                    df_nova_linha.to_csv(caminho_historico, mode="w", header=True, index=False)

            if resultado.get("aviso"):
                st.warning(resultado["aviso"])
            st.success(f"Modelo retreinado! Modelo final: {resultado['modelo_final']}. "
                       "Confira a aba 'Resultados do treino'.")

# ---------- ABA 3: RESULTADOS DO TREINO ----------
with aba_resultados:
    st.subheader("Comparacao de resultados do ultimo treino")

    if "ultimo_resultado" not in st.session_state:
        st.info("Ainda nao ha resultado nessa sessao. Va na aba 'Adicionar ao dataset' e clique em 'Retreinar modelo'.")
    else:
        resultado = st.session_state["ultimo_resultado"]
        resultados_por_algoritmo = resultado["resultados_por_algoritmo"]

        # ---- RESUMO: tabela comparando todos os algoritmos + media ----
        st.write("### Comparacao entre algoritmos")
        st.caption(
            "Validacao cruzada "
            + ("AGRUPADA por foto de origem" if resultado["agrupado"] else "NAO agrupada (resultado pode estar inflado)")
            + f", {resultado['cv_k_folds']} folds. Linha de base (tudo = IA): acuracia 66,7%, acuracia balanceada 50%. "
            f"Modelo salvo no app: **{resultado['modelo_final']}** (maior acuracia balanceada)."
        )
        if resultado.get("aviso"):
            st.warning(resultado["aviso"])
        linhas_resumo = []
        for nome_algoritmo, dados_alg in resultados_por_algoritmo.items():
            linha = {
                "Algoritmo": nome_algoritmo,
                "Acuracia geral (%)": f"{dados_alg['acuracia_geral']*100:.1f}",
                "Acur. balanceada (%)": f"{dados_alg['acuracia_balanceada']*100:.1f}",
                "F1 macro (%)": f"{dados_alg['f1_macro']*100:.1f}",
                "Revocacao Real (%)": f"{dados_alg['revocacao_real']*100:.1f}",
            }
            for nome_gerador, dados_gerador in dados_alg["acuracia_por_gerador"].items():
                linha[f"{nome_gerador} (%)"] = f"{dados_gerador['acuracia']:.1f}"
            linhas_resumo.append(linha)

        linha_media = {"Algoritmo": "MEDIA", "Acuracia geral (%)": f"{resultado['media_geral']*100:.1f}",
                       "Acur. balanceada (%)": f"{resultado['media_balanceada']*100:.1f}",
                       "F1 macro (%)": "", "Revocacao Real (%)": ""}
        for nome_gerador, media in resultado["media_por_gerador"].items():
            linha_media[f"{nome_gerador} (%)"] = f"{media:.1f}"
        linhas_resumo.append(linha_media)

        tabela_resumo = pd.DataFrame(linhas_resumo)
        st.table(tabela_resumo)

        st.download_button(
            "Baixar comparacao entre algoritmos (CSV)",
            data=tabela_resumo.to_csv(index=False),
            file_name="comparacao_algoritmos.csv",
            mime="text/csv",
        )

        # ---- CONTROLE de metadados ----
        controle = resultado.get("controle_metadados")
        if controle:
            st.write("##### Controle: classificador que ve so tamanho/formato do arquivo")
            st.write(f"Acuracia {controle['acuracia_geral']*100:.1f}% | "
                     f"acuracia balanceada {controle['acuracia_balanceada']*100:.1f}% "
                     "(perto de 50% = o formato/tamanho nao entrega a classe).")
            if controle["acuracia_balanceada"] >= 0.60:
                st.warning("Limitacao do dataset: o formato/tamanho do arquivo original ja separa as classes (isso nao muda com a padronizacao das features). Registre no relatorio.")

        st.write("##### Atende os minimos definidos? (modelo final)")
        nomes_limite = {"acuracia_balanceada": "Acuracia balanceada", "revocacao_real": "Revocacao da classe Real", "f1_macro": "F1 macro"}
        st.table(pd.DataFrame([
            {"Metrica": nomes_limite[m], "Minimo": f"{resultado['limites_minimos'][m]*100:.0f}%",
             "Atende": "sim" if ok else "nao"}
            for m, ok in resultado["atende_limites"].items()
        ]))

        st.divider()

        # ---- DETALHE POR ALGORITMO: tabela + grafico + matriz de confusao ----
        st.write("### Detalhes por algoritmo")
        abas_algoritmos = st.tabs(list(resultados_por_algoritmo.keys()))

        for aba, (nome_algoritmo, dados_alg) in zip(abas_algoritmos, resultados_por_algoritmo.items()):
            with aba:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Acuracia", f"{dados_alg['acuracia_geral']*100:.1f}%")
                col2.metric("Acur. balanceada", f"{dados_alg['acuracia_balanceada']*100:.1f}%")
                col3.metric("F1 macro", f"{dados_alg['f1_macro']*100:.1f}%")
                col4.metric("Tempo", f"{dados_alg['tempo']:.1f}s")

                st.table(pd.DataFrame([
                    {"Classe": c, "Precisao (%)": f"{m['precisao']*100:.1f}",
                     "Revocacao (%)": f"{m['revocacao']*100:.1f}", "F1 (%)": f"{m['f1']*100:.1f}"}
                    for c, m in dados_alg["por_classe"].items()
                ]))

                tabela_geradores = pd.DataFrame([
                    {
                        "Gerador": nome,
                        "Acuracia (%)": f"{dados['acuracia']:.1f}",
                        "Acertos": f"{dados['acertos']}/{dados['total']}",
                    }
                    for nome, dados in dados_alg["acuracia_por_gerador"].items()
                ])
                st.table(tabela_geradores)

                grafico_dados = pd.DataFrame({
                    nome: [dados["acuracia"]]
                    for nome, dados in dados_alg["acuracia_por_gerador"].items()
                }, index=["Acuracia (%)"]).T
                st.bar_chart(grafico_dados)

                st.write("##### Matriz de confusao")
                matriz = dados_alg["matriz_confusao"]
                tabela_matriz = pd.DataFrame(
                    matriz,
                    index=["Real de fato: IA", "Real de fato: Real"],
                    columns=["Previsto: IA", "Previsto: Real"],
                )
                st.table(tabela_matriz)

                with st.expander("Ver relatorio tecnico completo"):
                    st.text(dados_alg["relatorio_texto"])

        st.divider()

        st.write("### Importancia de cada caracteristica (feature)")
        st.caption("Calculada a partir do modelo final (Random Forest)")
        tabela_features = pd.DataFrame(resultado["importancias"], columns=["Feature", "Importancia"])
        st.bar_chart(tabela_features.set_index("Feature"))

        st.divider()
        st.write("### Historico de todos os treinos (evolucao ao longo do tempo)")
        caminho_historico = "historico_resultados.csv"
        if os.path.exists(caminho_historico):
            try:
                df_historico = pd.read_csv(caminho_historico)
                st.dataframe(df_historico)
                st.download_button(
                    "Baixar historico completo (CSV)",
                    data=df_historico.to_csv(index=False),
                    file_name="historico_resultados.csv",
                    mime="text/csv",
                )
            except Exception as erro:
                st.warning(f"Nao foi possivel ler o historico ({erro}). Ele pode estar num formato antigo/corrompido.")
        else:
            st.info("Ainda sem historico salvo.")