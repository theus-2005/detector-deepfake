"""
extrair_features.py

Pega UMA imagem e transforma ela em uma lista de numeros (features).
Essas features depois viram uma linha na tabela que a IA vai treinar em cima.
"""

import cv2
import numpy as np


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


def extrair_features(caminho_imagem):
    """FUNCAO PRINCIPAL: recebe o caminho de uma imagem e devolve
    um dicionario com todas as features calculadas."""
    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        return None

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    ruido = feature_ruido(cinza)
    frequencia = feature_frequencia(cinza)
    hist_b_media, hist_b_std, hist_g_media, hist_g_std, hist_r_media, hist_r_std = feature_histograma_cor(imagem)

    return {
        "ruido": float(ruido),
        "freq_alta": float(frequencia),
        "hist_b_media": float(hist_b_media),
        "hist_b_std": float(hist_b_std),
        "hist_g_media": float(hist_g_media),
        "hist_g_std": float(hist_g_std),
        "hist_r_media": float(hist_r_media),
        "hist_r_std": float(hist_r_std),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python extrair_features.py caminho/da/imagem.jpg")
        sys.exit(1)

    resultado = extrair_features(sys.argv[1])
    if resultado is None:
        print("Nao consegui abrir a imagem. Confira o caminho.")
    else:
        print("\nFeatures extraidas:")
        for chave, valor in resultado.items():
            print(f"  {chave}: {valor:.4f}")
