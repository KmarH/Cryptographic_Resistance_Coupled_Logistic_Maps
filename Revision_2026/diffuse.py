"""Probabilidades de aproximacion diferencial y lineal, exhaustivas y a nivel de bit.

Implementa las definiciones de Alvarez y Li, Int. J. Bifurcat. Chaos 16, 2129
(2006), ecuaciones (8) y (9) del informe:

    PAD = max_{dx != 0, dy}  #{x in Omega : f(x) ^ f(x ^ dx) = dy} / 2^n

    PAL = max_{a, b != 0}  ( (#{x in Omega : x.a = f(x).b} - 2^(n-1)) / 2^(n-1) )^2

donde Omega = {0,1}^n, ^ es XOR y . es el producto bit a bit (paridad del AND).
Ambas se calculan sobre el espacio de entrada completo, sin muestreo, sin
tolerancias de punto flotante y sin normalizaciones ajenas al tamano de bloque.

La funcion f se pasa como tabla de verdad: un arreglo de 2^n enteros sin signo,
`table[x] = f(x)`, con n bits de salida.

Convenciones que conviene tener presentes al comparar con la literatura:

  - PAL se expresa como el cuadrado de la correlacion (convencion de Matsui).
    Con W(a,b) = sum_x (-1)^(x.a ^ f(x).b) el coeficiente de Walsh, se tiene
    #coincidencias = (W + 2^n)/2 y por lo tanto PAL = (W / 2^n)^2.
  - `require_nonzero_input_mask` decide si se exige a != 0, y por defecto es
    True, que es la lectura estricta de `max_{a,b != 0}` en la ecuacion (9).
    Admitir a = 0 convierte el estadistico en el sesgo de f(x).b por si solo, y
    entonces cualquier f con un bit de salida constante da PAL = 1, lo cual no
    dice nada sobre linealidad. Se reporta en ambas variantes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bit_parity",
    "differential_probability",
    "differential_spectrum",
    "fwht",
    "is_bitwise_separable",
    "linear_probability",
    "walsh_spectrum",
]


def _popcount(values: np.ndarray) -> np.ndarray:
    try:
        return np.bitwise_count(values)
    except AttributeError:  # numpy < 2.0
        table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        out = np.zeros(values.shape, dtype=np.uint8)
        work = values.astype(np.uint64, copy=True)
        while np.any(work):
            out += table[(work & 0xFF).astype(np.uint8)]
            work >>= np.uint64(8)
        return out


def bit_parity(values: np.ndarray) -> np.ndarray:
    """Paridad del numero de bits en 1, es decir el producto bit a bit con 1...1."""
    return (_popcount(np.asarray(values)) & 1).astype(np.int8)


def _check_table(table: np.ndarray, n: int) -> np.ndarray:
    table = np.asarray(table)
    if table.ndim != 1 or table.size != 1 << n:
        raise ValueError(f"la tabla debe tener 2^{n} = {1 << n} entradas, tiene {table.size}")
    if table.min() < 0 or table.max() >= (1 << n):
        raise ValueError(f"las salidas deben caber en {n} bits")
    return table.astype(np.int64, copy=False)


def differential_spectrum(table: np.ndarray, n: int) -> np.ndarray:
    """Conteos #{x : f(x) ^ f(x ^ dx) = dy} como matriz (2^n, 2^n), indexada [dx, dy].

    La fila dx = 0 se incluye por completitud (toda su masa esta en dy = 0) pero
    queda excluida del maximo que devuelve `differential_probability`.
    """
    table = _check_table(table, n)
    size = 1 << n
    x = np.arange(size, dtype=np.int64)
    spectrum = np.zeros((size, size), dtype=np.int64)
    for dx in range(size):
        diffs = table ^ table[x ^ dx]
        spectrum[dx] = np.bincount(diffs, minlength=size)
    return spectrum


def differential_probability(table: np.ndarray, n: int) -> tuple[float, int, int]:
    """PAD exhaustiva. Devuelve (pad, dx, dy) con el par que alcanza el maximo."""
    table = _check_table(table, n)
    size = 1 << n
    x = np.arange(size, dtype=np.int64)
    best = (-1, 0, 0)
    for dx in range(1, size):
        counts = np.bincount(table ^ table[x ^ dx], minlength=size)
        dy = int(np.argmax(counts))
        if counts[dy] > best[0]:
            best = (int(counts[dy]), dx, dy)
    return best[0] / size, best[1], best[2]


def fwht(signs: np.ndarray) -> np.ndarray:
    """Transformada rapida de Walsh-Hadamard con nucleo (-1)^popcount(a & x).

    Devuelve W[a] = sum_x (-1)^popcount(a & x) * signs[x]. La longitud debe ser
    potencia de dos.
    """
    out = np.array(signs, dtype=np.int64)
    size = out.size
    if size & (size - 1):
        raise ValueError("la longitud debe ser potencia de dos")
    half = 1
    while half < size:
        view = out.reshape(-1, 2, half)
        lo = view[:, 0, :].copy()
        hi = view[:, 1, :].copy()
        view[:, 0, :] = lo + hi
        view[:, 1, :] = lo - hi
        out = view.reshape(size)
        half <<= 1
    return out


def walsh_spectrum(table: np.ndarray, n: int) -> np.ndarray:
    """Coeficientes W(a, b) = sum_x (-1)^(x.a ^ f(x).b) como matriz (2^n, 2^n)."""
    table = _check_table(table, n)
    size = 1 << n
    spectrum = np.zeros((size, size), dtype=np.int64)
    for b in range(size):
        signs = 1 - 2 * bit_parity(table & b).astype(np.int64)
        spectrum[:, b] = fwht(signs)
    return spectrum


def linear_probability(
    table: np.ndarray,
    n: int,
    *,
    require_nonzero_input_mask: bool = True,
) -> tuple[float, int, int]:
    """PAL exhaustiva. Devuelve (pal, a, b) con las mascaras que alcanzan el maximo.

    Siempre se exige b != 0. `require_nonzero_input_mask` agrega a != 0, que es
    el valor por defecto y la lectura estricta de la ecuacion (9).
    """
    table = _check_table(table, n)
    size = 1 << n
    best = (-1, 0, 0)
    for b in range(1, size):
        signs = 1 - 2 * bit_parity(table & b).astype(np.int64)
        walsh = np.abs(fwht(signs))
        if require_nonzero_input_mask:
            walsh[0] = -1
        a = int(np.argmax(walsh))
        if walsh[a] > best[0]:
            best = (int(walsh[a]), a, b)
    return (best[0] / size) ** 2, best[1], best[2]


def is_bitwise_separable(table: np.ndarray, n: int) -> tuple[bool, list[int] | None]:
    """Comprueba si f es separable bit a bit, es decir f(x)_j = g_j(x_j).

    Un cifrado con esta propiedad no tiene difusion alguna entre bits del texto
    claro: la diferencia de salida depende solo de dx y nunca de x, de modo que
    PAD = 1 de forma estructural. Devuelve (separable, g) donde g[j] codifica la
    funcion de un bit g_j como el par (g_j(0), g_j(1)) empaquetado en 2 bits.
    """
    table = _check_table(table, n)
    size = 1 << n
    codes: list[int] = []
    for j in range(n):
        mask = 1 << j
        out_lo = table[0] >> j & 1
        out_hi = table[mask] >> j & 1
        codes.append(int(out_lo) | (int(out_hi) << 1))
    rebuilt = np.zeros(size, dtype=np.int64)
    for j in range(n):
        g0, g1 = codes[j] & 1, codes[j] >> 1
        bit = np.where((np.arange(size) >> j & 1) == 1, g1, g0)
        rebuilt |= bit.astype(np.int64) << j
    return bool(np.array_equal(rebuilt, table)), codes
