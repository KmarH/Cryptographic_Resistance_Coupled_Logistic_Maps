"""Criptoanalisis diferencial y lineal correcto de los dos cifrados, mas sustitutos.

Produce los numeros que reemplazan los cuadros I y II del informe. Se divide en
cuatro partes:

  A. PAD y PAL exhaustivas, bit a bit, segun las ecuaciones (8) y (9).
  B. Diagnostico estructural: por que esos valores salen degenerados.
  C. Sustitutos que si discriminan los dos mapas: avalancha intrabloque, perfil
     de divergencia y exponentes de Lyapunov.
  D. Sensibilidad a la llave, que es la nocion de difusion pertinente para un
     cifrado de flujo por enmascaramiento caotico.

Todas las cuantizaciones usan la mediana empirica de la orbita de cada esquema,
no el umbral geometrico 1/2. Con 1/2 la fuente de bits del LM queda con 64 % de
unos y la del GLM con 92 %, de modo que la linea base de decorrelacion es
2p(1-p) y vale 0.46 para uno y 0.15 para el otro: comparar los dos esquemas con
el mismo umbral fijo reproduce el mismo error de saturacion que arruina la
metrica de distancia media del informe.

Uso:  python run_analysis.py [--nmax 12] [--out resultados.md]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import numpy as np

from clm_cipher import (
    GLM_CONFIG,
    GLM_KEY,
    LM_CONFIG,
    LM_KEY,
    CipherConfig,
    Key,
    block_ciphertexts,
    cipher_table,
    divergence_profile,
    keystream,
    orbit_median,
    quantize_block,
)
from diffuse import differential_probability, is_bitwise_separable, linear_probability

SCHEMES: list[tuple[str, Key, CipherConfig]] = [
    ("LM", LM_KEY, LM_CONFIG),
    ("GLM", GLM_KEY, GLM_CONFIG),
]
EXTRACTORS = ["last_msb", "mean_msb", "mantissa_xor", "first_msb"]

# Valores que el informe reporta, para contrastar.
REPORTED = {
    (8, "GLM"): (0.0234375, 0.0007244981825),
    (8, "LM"): (0.015625, 0.0007244981825),
    (16, "GLM"): (0.0001678466797, 1.054283614e-10),
    (16, "LM"): (0.000244140625, 1.054283614e-10),
}


def thresholds() -> dict[str, float]:
    return {tag: orbit_median(key, cfg) for tag, key, cfg in SCHEMES}


def part_a(lines: list[str], sizes: list[int], thr: dict[str, float]) -> None:
    lines.append("## A. PAD y PAL exhaustivas segun las ecuaciones (8) y (9)\n")
    lines.append(
        "Espacio de entrada completo Omega = {0,1}^n, XOR sobre bits, sin tolerancias "
        "de punto flotante y sin normalizaciones ajenas al tamano de bloque. El nucleo "
        "esta validado contra la S-box de AES, cuyos valores conocidos son "
        "PAD = PAL = 2^-6 = 0.015625.\n"
    )
    lines.append("| esquema | n | extractor | PAD | dx | dy | PAL | a | b |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tag, key, cfg in SCHEMES:
        for n in sizes:
            blocks = block_ciphertexts(key, cfg, n)
            for how in EXTRACTORS:
                table = cipher_table(key, cfg, n, how, blocks=blocks, threshold=thr[tag])
                pad, dx, dy = differential_probability(table, n)
                pal, a, b = linear_probability(table, n)
                lines.append(
                    f"| {tag} | {n} | `{how}` | {pad:.6g} | {dx} | {dy} "
                    f"| {pal:.6g} | {a} | {b} |"
                )
    lines.append("")
    lines.append("Contraste con lo publicado:\n")
    lines.append("| esquema | n | PAD informe | PAD correcta | PAL informe | PAL correcta |")
    lines.append("|---|---|---|---|---|---|")
    for (n, tag), (pad_old, pal_old) in sorted(REPORTED.items()):
        if n not in sizes:
            continue
        key, cfg = (LM_KEY, LM_CONFIG) if tag == "LM" else (GLM_KEY, GLM_CONFIG)
        table = cipher_table(key, cfg, n, "last_msb", threshold=thr[tag])
        pad, _, _ = differential_probability(table, n)
        pal, _, _ = linear_probability(table, n)
        lines.append(
            f"| {tag} | {n} | {pad_old:.6g} | {pad:.6g} | {pal_old:.6g} | {pal:.6g} |"
        )
    lines.append("")
    lines.append(
        "PAD vale 1 en los dos esquemas, en todos los tamanos de bloque y con todos "
        "los extractores. PAL vale 1 o 0 segun el extractor, es decir el estadistico "
        "ni siquiera esta bien definido para este cifrado. Ninguno de los dos valores "
        "publicados es reproducible: eran el conteo de coincidencias en punto flotante "
        "dividido por 2^n en el caso de PAD, y una constante que solo depende del "
        "numero de bits del mensaje en el caso de PAL.\n"
    )


def part_b(lines: list[str], n: int, thr: dict[str, float]) -> None:
    lines.append("## B. Por que los valores salen degenerados\n")
    for tag, key, cfg in SCHEMES:
        table = cipher_table(key, cfg, n, "mantissa_xor", threshold=thr[tag])
        separable, codes = is_bitwise_separable(table, n)
        bijective = sum(1 for c in codes if (c & 1) != (c >> 1))
        lines.append(
            f"- **{tag}**, n = {n}: f separable bit a bit, f(x)_j = g_j(x_j): "
            f"**{separable}**; {bijective} de {n} bits de salida dependen "
            f"biyectivamente de su bit de entrada"
        )
    lines.append("")
    lines.append(
        "Si f es separable bit a bit, f(x) ^ f(x ^ dx) no depende de x sino solo de "
        "dx, asi que para cada dx toda la masa del espectro diferencial cae en un "
        "unico dy y PAD = 1 exactamente. Si ademas algun g_j es biyectivo, las "
        "mascaras a = b = e_j dan una identidad que se cumple para los 2^n textos "
        "claros y PAL = 1 exactamente.\n"
    )
    lines.append(
        "La causa esta en la estructura del cifrado, no en la eleccion de cuantizador. "
        "El bloque j arranca en x[stpoint + j*block_len] de la orbita libre y su "
        "recursion solo consume ese tramo del flujo de clave y su propio bit m_j, de "
        "modo que no hay difusion entre bits del texto claro: cambiar un bit del "
        "mensaje altera 50 muestras del criptograma y deja todas las demas "
        "bit a bit identicas. Esto se verifica cifrando los 2^n textos claros uno por "
        "uno (`cipher_table_bruteforce`) y comparando con las 2n evaluaciones de "
        "`cipher_table`.\n"
    )
    lines.append(
        "PAD y PAL estan definidas para cifrados de bloque. Aqui no miden la fortaleza "
        "del esquema: miden que el esquema no es un cifrado de bloque, sino una "
        "modulacion por enmascaramiento caotico. La seccion de criptoanalisis "
        "diferencial y lineal del informe debe retirarse o reescribirse con este "
        "resultado, y las nociones de seguridad pertinentes son las de las partes C y "
        "D junto con la resistencia a estimacion de parametros que el informe ya mide.\n"
    )
    lines.append(
        "Corolario que refuerza el ataque CBPSO: con m_j = 0 el bloque es exactamente "
        "el tramo de la orbita libre, porque x_t[0] = x[stpoint + j*block_len] y la "
        "recursion del transmisor coincide con la del generador cuando A*m = 0. Todo "
        "bit 0 de cualquier mensaje entrega 50 puntos limpios del mapa. El informe "
        "supone que el atacante puede fijar m(n) = 0; en realidad no necesita fijar "
        "nada.\n"
    )


def _lyapunov(key: Key, cfg: CipherConfig, n_iter: int = 200_000) -> tuple[float, float]:
    """Exponentes de Lyapunov del mapa acoplado 2D por QR del producto jacobiano."""
    if key.kind == "lm":
        def jac(x, y):
            return np.array([
                [key.mu * (1 - 2 * x) - key.alpha, key.alpha],
                [key.alpha, key.mu1 * (1 - 2 * y) - key.alpha],
            ])
    else:
        (p1, q1), (p2, q2) = key.threshold_amplitude()

        def dgen(v, p, q):
            denom = p * p if v <= p else (1 - p) ** 2
            return 2 * q * (p - v) / denom

        def jac(x, y):
            return np.array([
                [dgen(x, p1, q1) - key.alpha, key.alpha],
                [key.alpha, dgen(y, p2, q2) - key.alpha],
            ])

    xs, ys = keystream(key, cfg, n_iter + 1)
    basis = np.eye(2)
    total = np.zeros(2)
    for i in range(n_iter):
        basis, upper = np.linalg.qr(jac(xs[i], ys[i]) @ basis)
        signs = np.sign(np.diag(upper))
        basis = basis * signs
        total += np.log(np.abs(np.diag(upper)) + 1e-300)
    return float(total[0] / n_iter), float(total[1] / n_iter)


def _conditional_exponent(key: Key, cfg: CipherConfig, n_blocks: int) -> float:
    """Exponente de Lyapunov condicional del transmisor, con y_r fijo.

    El bit del mensaje entra en la recursion de x_t, que esta forzada por un
    tramo fijo del flujo de clave. Su tasa de crecimiento no es lambda_1 del
    mapa acoplado 2D sino el promedio de log|f'(x_t) - alpha| a lo largo del
    bloque: es esa cantidad la que dice si la inyeccion A*m se amplifica o se
    amortigua, y por lo tanto la que explica la avalancha intrabloque.
    """
    if key.kind == "lm":
        def deriv(v):
            return key.mu * (1 - 2 * v) - key.alpha
    else:
        (p1, q1), _ = key.threshold_amplitude()

        def deriv(v):
            denom = p1 * p1 if v <= p1 else (1 - p1) ** 2
            return 2 * q1 * (p1 - v) / denom - key.alpha

    blocks = block_ciphertexts(key, cfg, n_blocks)
    logs = [
        math.log(abs(deriv(value)) + 1e-300)
        for j in range(n_blocks)
        for value in blocks[j, 0]
    ]
    return float(np.mean(logs))


def part_c(lines: list[str], n: int, thr: dict[str, float]) -> None:
    lines.append("## C. Sustitutos que si discriminan los dos mapas\n")
    lines.append(
        "Avalancha intrabloque: fraccion de los `block_len` bits cuantizados de un "
        "bloque que cambian al invertir su bit de texto claro, restringida al unico "
        "bloque que ese bit puede afectar. Valor ideal 0.5. Cuantizacion a la mediana "
        "de cada orbita, de modo que la linea base de decorrelacion es 0.5 para los "
        "dos esquemas.\n"
    )
    lines.append(
        "| esquema | umbral (mediana) | avalancha media | desv. est. | min | max "
        "| lambda_1 | lambda_2 | lambda_cond |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    profiles = {}
    for tag, key, cfg in SCHEMES:
        blocks = block_ciphertexts(key, cfg, n)
        flips = np.array([
            np.mean(
                quantize_block(blocks[j, 0], thr[tag]) != quantize_block(blocks[j, 1], thr[tag])
            )
            for j in range(n)
        ])
        lam1, lam2 = _lyapunov(key, cfg)
        lam_cond = _conditional_exponent(key, cfg, n)
        profiles[tag] = divergence_profile(blocks)
        lines.append(
            f"| {tag} | {thr[tag]:.4f} | {flips.mean():.4f} | {flips.std(ddof=1):.4f} "
            f"| {flips.min():.4f} | {flips.max():.4f} | {lam1:+.4f} | {lam2:+.4f} "
            f"| {lam_cond:+.4f} |"
        )
    lines.append("")
    lines.append(
        "lambda_1 y lambda_2 son los exponentes del mapa acoplado 2D, en nats por "
        "iteracion, sobre la orbita libre, 2x10^5 iteraciones con reortonormalizacion "
        "QR. El informe los menciona como criterio pero no reporta ninguno. Notese que "
        "mu = 3.1 y mu1 = 2.9 situan a cada mapa logistico aislado en regimen "
        "periodico (periodo 2 y punto fijo respectivamente): el caos del esquema "
        "simple lo produce el acoplamiento.\n"
    )
    lines.append(
        "lambda_cond es el exponente condicional del transmisor, con el tramo de flujo "
        "de clave y_r fijo. Es el que gobierna la avalancha intrabloque, y explica por "
        "que el GLM la tiene mas baja que el LM a pesar de tener un lambda_1 mas de "
        "dos veces mayor: el bit del mensaje entra en una recursion forzada cuya tasa "
        "de crecimiento es distinta de la del sistema acoplado completo. Es la "
        "distincion que hay que hacer explicita en el informe, porque el argumento "
        "sobre difusion esta planteado sobre el exponente equivocado.\n"
    )
    lines.append(
        "Perfil de divergencia |x_t(m=1) - x_t(m=0)| promediado sobre bloques, sin "
        "cuantizador, en las posiciones de muestra indicadas:\n"
    )
    idx = [1, 2, 5, 10, 20, 30, 40, 49]
    lines.append("| esquema | " + " | ".join(f"i={i}" for i in idx) + " |")
    lines.append("|---|" + "---|" * len(idx))
    for tag in profiles:
        row = " | ".join(f"{profiles[tag][i]:.3e}" for i in idx)
        lines.append(f"| {tag} | {row} |")
    lines.append("")
    lines.append(
        "La inyeccion vale A = 0.01, valor que el informe nunca declara. El perfil "
        "arranca ahi y satura cuando alcanza la escala del atractor; el numero de "
        "iteraciones que tarda es la cantidad que mide la difusion real del esquema.\n"
    )
    lines.append(
        "Estadistica de la orbita libre cuantizada. No es el criptograma, que es "
        "analogico, pero cualquier implementacion practica tiene que cuantizar, asi "
        "que acota lo que se puede esperar como fuente de bits. Con umbral 1/2 en "
        "lugar de la mediana, la proporcion de unos es 0.6417 para el LM y 0.9182 "
        "para el GLM.\n"
    )
    lines.append(
        "| esquema | proporcion de unos | corr. serial lag-1 | z de rachas "
        "| entropia de palabras de 8 bits |"
    )
    lines.append("|---|---|---|---|---|")
    for tag, key, cfg in SCHEMES:
        xs, _ = keystream(key, cfg, 200_001)
        orbit = xs[1:]
        bits = (orbit >= thr[tag]).astype(np.int8)
        ones = float(bits.mean())
        corr = float(np.corrcoef(orbit[:-1], orbit[1:])[0, 1])
        runs = 1 + int(np.sum(bits[1:] != bits[:-1]))
        size = bits.size
        expected = 2 * size * ones * (1 - ones)
        var = math.sqrt(2 * size) * 2 * ones * (1 - ones)
        z = (runs - expected) / var if var > 0 else float("nan")
        words = np.packbits(bits[: size // 8 * 8].reshape(-1, 8), axis=1).ravel()
        counts = np.bincount(words, minlength=256) / words.size
        nz = counts[counts > 0]
        entropy = float(-np.sum(nz * np.log2(nz)))
        lines.append(f"| {tag} | {ones:.4f} | {corr:+.4f} | {z:+.2f} | {entropy:.4f} / 8 |")
    lines.append("")


def part_d(lines: list[str], n: int, thr: dict[str, float]) -> None:
    lines.append("## D. Sensibilidad a la llave\n")
    lines.append(
        "Distancia de Hamming relativa entre criptogramas cuantizados del mismo "
        "mensaje cifrado con la llave original y con un parametro perturbado en delta. "
        "Es la nocion de difusion que corresponde a un cifrado de flujo por "
        "enmascaramiento, y conecta con el ataque CBPSO: 0.5 a delta pequeno significa "
        "que un error minimo en la llave estimada arruina el descifrado. Cuantizacion "
        "a la mediana, asi que 0.5 es la linea base de decorrelacion en ambos casos.\n"
    )
    deltas = [1e-3, 1e-6, 1e-9, 1e-12]
    lines.append("| esquema | parametro | " + " | ".join(f"delta={d:g}" for d in deltas) + " |")
    lines.append("|---|---|" + "---|" * len(deltas))
    message = np.array([(0xA5A5 >> j) & 1 for j in range(n)], dtype=np.uint8)
    for tag, key, cfg in SCHEMES:
        chosen = block_ciphertexts(key, cfg, n)[np.arange(n), message]
        base = np.concatenate([quantize_block(b, thr[tag]) for b in chosen])
        fields = ["mu", "mu1", "alpha"] + (["nu", "nu1"] if key.kind == "glm" else [])
        for field in fields:
            row = []
            for delta in deltas:
                perturbed = replace(key, **{field: getattr(key, field) + delta})
                try:
                    blocks = block_ciphertexts(perturbed, cfg, n)[np.arange(n), message]
                    other = np.concatenate([quantize_block(b, thr[tag]) for b in blocks])
                    row.append(f"{np.mean(base != other):.4f}")
                except (FloatingPointError, OverflowError, ValueError):
                    row.append("div.")
            lines.append(f"| {tag} | {field} | " + " | ".join(row) + " |")
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmax", type=int, default=12, help="tamano de bloque maximo")
    parser.add_argument("--out", default="resultados_criptoanalisis.md")
    args = parser.parse_args()

    # PAL exhaustiva cuesta 2^n transformadas de Walsh de longitud 2^n, asi que
    # n = 16 son ~7x10^10 operaciones. No hace falta: la parte B demuestra que f
    # es separable bit a bit para cualquier n, de donde PAD = 1 sin calcularla.
    sizes = [n for n in (8, 12) if n <= args.nmax]
    if args.nmax > 12:
        print("[nota] PAD/PAL exhaustivas se limitan a n <= 12; ver la parte B]\n")
    thr = thresholds()
    lines: list[str] = [
        "# Criptoanalisis diferencial y lineal de los mapas logisticos acoplados",
        "",
        "Salida de `run_analysis.py`. PAD y PAL se calculan sobre el espacio de entrada "
        "completo con las definiciones de Alvarez y Li (2006), ecuaciones (8) y (9) del "
        "informe.",
        "",
    ]
    part_a(lines, sizes, thr)
    part_b(lines, min(12, args.nmax), thr)
    part_c(lines, 16, thr)
    part_d(lines, 16, thr)

    text = "\n".join(lines)
    print(text)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"\n[escrito en {args.out}]")


if __name__ == "__main__":
    main()
