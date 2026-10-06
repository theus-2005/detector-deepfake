"""
extrair_features.py

Pega UMA imagem e transforma ela em uma lista de numeros (features).
Essas features depois viram uma linha na tabela que a IA vai treinar em cima.
"""

import cv2
import numpy as np
import pywt


def feature_ruido(imagem_cinza):
    """Mede a quantidade de textura fina/ruido na imagem.
    Camera real tem ruido de sensor; imagem gerada por IA
    costuma ser mais 'lisa' em certas regioes."""
    laplaciano = cv2.Laplacian(imagem_cinza, cv2.CV_64F)
    return laplaciano.std()


def feature_frequencia(imagem_cinza):
    """Aplica FFT e mede a energia nas altas frequencias.
    Modelos generativos costumam deixar padroes sutis nessa regiao."""
    f = np.fft.fft2(imagem_cinza)
    fshift = np.fft.fftshift(f)
    espectro = np.log(np.abs(fshift) + 1)

    h, w = espectro.shape
    cy, cx = h // 2, w // 2
    raio = min(h, w) // 4

    y, x = np.ogrid[:h, :w]
    mascara_alta_freq = (x - cx) ** 2 + (y - cy) ** 2 > raio ** 2

    return espectro[mascara_alta_freq].mean()


def feature_histograma_cor(imagem_colorida):
    """Calcula estatisticas do histograma de cada canal de cor (B, G, R).
    Devolve media e desvio padrao de cada canal (6 numeros no total)."""
    resultado = []
    for canal in range(3):
        hist = cv2.calcHist([imagem_colorida], [canal], None, [256], [0, 256])
        hist = hist.flatten()
        valores = np.arange(256)
        media = np.sum(hist * valores) / np.sum(hist)
        variancia = np.sum(hist * (valores - media) ** 2) / np.sum(hist)
        resultado.append(media)
        resultado.append(np.sqrt(variancia))
    return resultado


def feature_ela(imagem_colorida, qualidade=90):
    """ELA (Error Level Analysis): salva a imagem de novo como JPEG com
    qualidade fixa (so na memoria) e mede o quanto ela mudou.
    Imagens com historico de compressao/ruido diferentes reagem de forma
    diferente a essa recompressao. Devolve media e desvio padrao da diferenca."""
    ok, buffer = cv2.imencode(".jpg", imagem_colorida, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    recomprimida = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    diferenca = cv2.absdiff(imagem_colorida, recomprimida).astype(np.float64)
    return diferenca.mean(), diferenca.std()


def _codigos_uniformes():
    """Marca quais dos 256 codigos LBP sao 'uniformes' (no maximo 2
    trocas entre 0 e 1 quando se le os 8 bits em circulo)."""
    uniformes = np.zeros(256, dtype=bool)
    for codigo in range(256):
        girado = ((codigo << 1) | (codigo >> 7)) & 0xFF
        trocas = bin(codigo ^ girado).count("1")
        uniformes[codigo] = trocas <= 2
    return uniformes


_UNIFORMES = _codigos_uniformes()


def feature_lbp(imagem_cinza):
    """LBP (Local Binary Pattern): pra cada pixel, compara ele com os 8
    vizinhos (vizinho >= centro vira 1, senao 0) e forma um codigo de 8 bits.
    Depois resume a distribuicao desses codigos em 3 numeros:
    entropia (quao variada e a textura), fracao de padroes uniformes
    (bordas/cantos regulares) e fracao de padroes planos (todos 0 ou todos 1)."""
    c = imagem_cinza.astype(np.int16)
    centro = c[1:-1, 1:-1]
    vizinhos = [
        c[:-2, :-2], c[:-2, 1:-1], c[:-2, 2:], c[1:-1, 2:],
        c[2:, 2:], c[2:, 1:-1], c[2:, :-2], c[1:-1, :-2],
    ]
    codigo = np.zeros(centro.shape, dtype=np.uint8)
    for i, vizinho in enumerate(vizinhos):
        codigo |= ((vizinho >= centro).astype(np.uint8) << i)

    hist = np.bincount(codigo.ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum()

    nao_zero = hist[hist > 0]
    entropia = -(nao_zero * np.log2(nao_zero)).sum()
    uniformes = hist[_UNIFORMES].sum()
    planos = hist[0] + hist[255]
    return entropia, uniformes, planos


def feature_wavelet(imagem_cinza):
    """Wavelet (transformada de Haar, 1 nivel): separa a imagem em 4 partes:
    uma versao 'resumida' (aproximacao, ignorada aqui) e 3 'mapas de detalhe'
    -- horizontal, vertical e diagonal -- que capturam borda/textura em
    cada direcao. Imagens de IA tendem a ter menos energia nesses mapas
    (mais uma vez, o efeito de serem 'lisas demais').
    Devolve a energia (media dos valores ao quadrado) de cada mapa."""
    _, (detalhe_h, detalhe_v, detalhe_d) = pywt.dwt2(imagem_cinza.astype(np.float64), "haar")
    energia_h = np.mean(detalhe_h ** 2)
    energia_v = np.mean(detalhe_v ** 2)
    energia_d = np.mean(detalhe_d ** 2)
    return energia_h, energia_v, energia_d


def extrair_features(caminho_imagem, tamanho_maximo=512):
    """FUNCAO PRINCIPAL: recebe o caminho de uma imagem e devolve
    um dicionario com todas as features calculadas.

    IMPORTANTE sobre redimensionamento: imagens MAIORES que
    tamanho_maximo sao reduzidas, pra evitar que geradores que
    produzem imagens muito grandes (ex: GPT) tenham vantagem
    artificial nas features so por causa do tamanho.

    Imagens MENORES ou iguais a tamanho_maximo NAO sao ampliadas.
    Ampliar (upscale) borraria a imagem e destruiria o ruido real
    de sensor de camera -- que e justamente o sinal mais importante
    pra diferenciar foto real de imagem gerada. Ampliar fotos reais
    comprimidas faria elas parecerem 'lisas demais', confundindo o
    modelo a favor de classifica-las como IA por engano."""
    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        return None

    altura, largura = imagem.shape[:2]
    maior_lado = max(altura, largura)

    if maior_lado > tamanho_maximo:
        escala = tamanho_maximo / maior_lado
        nova_largura = int(largura * escala)
        nova_altura = int(altura * escala)
        imagem = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    # se for menor ou igual, mantem o tamanho original -- sem upscale


    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    ruido = feature_ruido(cinza)
    frequencia = feature_frequencia(cinza)
    hist_b_media, hist_b_std, hist_g_media, hist_g_std, hist_r_media, hist_r_std = feature_histograma_cor(imagem)
    ela_media, ela_std = feature_ela(imagem)
    lbp_entropia, lbp_uniformes, lbp_planos = feature_lbp(cinza)
    wavelet_h, wavelet_v, wavelet_d = feature_wavelet(cinza)

    return {
        "ruido": float(ruido),
        "freq_alta": float(frequencia),
        "hist_b_media": float(hist_b_media),
        "hist_b_std": float(hist_b_std),
        "hist_g_media": float(hist_g_media),
        "hist_g_std": float(hist_g_std),
        "hist_r_media": float(hist_r_media),
        "hist_r_std": float(hist_r_std),
        "ela_media": float(ela_media),
        "ela_std": float(ela_std),
        "lbp_entropia": float(lbp_entropia),
        "lbp_uniformes": float(lbp_uniformes),
        "lbp_planos": float(lbp_planos),
        "wavelet_h": float(wavelet_h),
        "wavelet_v": float(wavelet_v),
        "wavelet_d": float(wavelet_d),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python extrair_features.py caminho/da/imagem.jpg")
        sys.exit(1)

    caminho = sys.argv[1]
    imagem = cv2.imread(caminho)
    
    if imagem is None:
        print("Nao consegui abrir a imagem. Confira o caminho.")
        sys.exit(1)
        
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Testando diretamente as tres que voce quer ver
    print("\n--- TESTANDO LABELS ESPECIFICAS ---")
    
    ela_m, ela_s = feature_ela(imagem)
    print(f"ELA (Media / Desvio): {ela_m:.4f} / {ela_s:.4f}")
    
    lbp_e, lbp_u, lbp_p = feature_lbp(cinza)
    print(f"LBP (Entropia / Uniformes / Planos): {lbp_e:.4f} / {lbp_u:.4f} / {lbp_p:.4f}")
    
    w_h, w_v, w_d = feature_wavelet(cinza)
    print(f"Wavelet (Horizontal / Vertical / Diagonal): {w_h:.4f} / {w_v:.4f} / {w_d:.4f}")
    print("-----------------------------------\n")

    # Execucao do dicionario completo
    resultado = extrair_features(caminho)
    if resultado is not None:
        print("Todas as features salvas no dicionario:")
        for chave, valor in resultado.items():
            print(f"  {chave}: {valor:.4f}")
