# Detector de imagem Real vs Gerada por IA (ChatGPT / Gemini)

Protótipo do projeto: extrai 16 características de cada imagem, compara 4 algoritmos
(Random Forest, Gradient Boosting, SVM, Regressão Logística) e usa o melhor no app.

## 1. Instalar (uma vez)
```
python -m venv .venv
.venv\Scripts\Activate.ps1        (Windows PowerShell)
pip install -r requirements.txt
```

## 2. Organizar as imagens
```
dataset/
  reais/       foto001.jpg ...            (rótulo 1 = real)
  ia_gpt/      foto001_gpt.png ...        (rótulo 0 = IA)
  ia_gemini/   foto001_gemini.png ...     (rótulo 0 = IA)
```
**Regra do nome:** a foto real e as duas edições dela têm o mesmo nome-base (`foto001`).
É isso que agrupa as 3 imagens na validação (evita vazamento de dados).
Se os nomes forem diferentes, crie `dataset/grupos.csv`:
```
arquivo,foto_origem
IMG_2231.jpg,foto001
ChatGPT_Image_3.png,foto001
gemini_7.png,foto001
```
As versões e datas dos geradores estão em `registro_geradores.csv` (veja a seção 8).

## 3. Montar o dataset e treinar
```
python montar_dataset.py
python treinar_modelo.py
```
- `montar_dataset.py` gera `dataset.csv` e informa quantos grupos estão completos
  (1 real + 1 GPT + 1 Gemini). Se aparecer "ATENÇÃO", os nomes não casam.
- `treinar_modelo.py` imprime, para cada algoritmo: acurácia, acurácia balanceada,
  F1, revocação, resultado por gerador e o **controle de metadados**; salva `modelo.pkl`
  (algoritmo de maior acurácia balanceada) e `modelo_info.json`.

## 4. Rodar o app
```
streamlit run app.py
```
Abas: testar imagem, adicionar ao dataset/retreinar, resultados do treino.

## 5. Como a avaliação funciona
- **Validação agrupada por foto** (StratifiedGroupKFold, 5 folds): a foto real e suas
  edições nunca ficam separadas entre treino e teste.
- **Métricas:** acurácia, acurácia balanceada, precisão/revocação/F1 por classe, F1 macro.
  Linha de base ("tudo é IA"): acurácia 66,7%, acurácia balanceada 50%.
  Mínimos de aceitação (editáveis em `treinar_modelo.py`, `LIMITES_MINIMOS`):
  acurácia balanceada ≥ 60%, revocação da classe real ≥ 50%, F1 macro ≥ 55%.
- **Mesmo tratamento dos 4 algoritmos:** peso balanceado das classes em todos
  (no Gradient Boosting, via `sample_weight`).
- **Controle de formato/compressão:** todas as imagens passam pela mesma recompressão JPEG
  (qualidade 95) em memória; além disso, um classificador que vê só largura, altura,
  tamanho do arquivo e formato mostra se o formato sozinho entrega a classe
  (perto de 50% de acurácia balanceada = bom sinal).
- **Escolha do modelo final:** maior acurácia balanceada na validação agrupada
  (critério registrado em `modelo_info.json`).

## 6. Cinza × cor
`ruido`, `freq_alta`, `lbp_*` e `wavelet_*` usam a imagem em **tons de cinza**.
`hist_*` (6 valores de cor) e `ela_*` usam a imagem **colorida**.

## 7. Reproduzir (checklist para a Gabriela)
1. Instalar (seção 1) e colocar as imagens (seção 2).
2. `python montar_dataset.py` → conferir "grupos completos".
3. `python treinar_modelo.py` → conferir a validação "AGRUPADA" e o modelo final.
4. `streamlit run app.py` → testar uma imagem real e uma de IA.

## 8. Registro de geração das imagens
Foi usada uma conversa por gerador (ChatGPT e Gemini). A instrução de edição é a mesma
para as duas IAs em cada foto e muda de foto para foto. Os prompts completos estão
preservados no histórico dessas conversas. Plano, interface, modelo e período de cada
gerador estão em `registro_geradores.csv`.