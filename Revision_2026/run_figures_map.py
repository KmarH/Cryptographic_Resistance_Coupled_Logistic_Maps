"""Figuras 1 y 2 del informe: el mapa acoplado y la evaluacion de bits.

Equivalen a `fig3` y `fig2` de `m1_alter.py` y `en.py`: el diagrama de fases de
la orbita libre, y la senal del transmisor a lo largo de las 50 iteraciones que
cifran un bit, para un bit en 1 y un bit en 0.

Uso:  python run_figures_map.py [--outdir proyecto]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clm_cipher import GLM_CONFIG, GLM_KEY, LM_CONFIG, LM_KEY, block_ciphertexts, keystream


def figure_phase(path, key, cfg, n_points=10000):
    xs, ys = keystream(key, cfg, n_points)
    plt.figure(figsize=(5.4, 5.2))
    plt.scatter(xs[1:], ys[1:], s=0.6, color="tab:blue")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def figure_bits(path, key, cfg):
    blocks = block_ciphertexts(key, cfg, 2)
    steps = np.arange(1, cfg.block_len + 1)
    plt.figure(figsize=(5.4, 5.2))
    plt.plot(steps, blocks[0, 1], "o", markersize=3.5, color="tab:blue", label="Bit en 1")
    plt.plot(steps, blocks[0, 0], "o", markersize=3.5, color="tab:red", label="Bit en 0")
    plt.xlabel("$n$")
    plt.ylabel("Señal del transmisor")
    plt.xlim(0, cfg.block_len + 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="proyecto")
    args = parser.parse_args()
    images = Path(args.outdir) / "Imagenes"
    images.mkdir(parents=True, exist_ok=True)

    figure_phase(images / "glmleft.png", LM_KEY, LM_CONFIG)
    figure_bits(images / "glmright.png", LM_KEY, LM_CONFIG)
    figure_phase(images / "glmleftG.png", GLM_KEY, GLM_CONFIG)
    figure_bits(images / "glmrightG.png", GLM_KEY, GLM_CONFIG)
    print(f"[figuras del mapa escritas en {images}]")


if __name__ == "__main__":
    main()
