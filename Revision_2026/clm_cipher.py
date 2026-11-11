"""Reimplementacion fiel de los dos cifrados basados en mapas logisticos acoplados.

Reproduce `m1_alter.py` (mapa logistico simple acoplado) y `en.py` (mapa
logistico generalizado acoplado) del repositorio
KmarH/Cryptographic_Resistance_Coupled_Logistic_Maps, con tres diferencias
deliberadas, todas ellas correcciones:

  1. El texto claro se convierte a bits con ancho fijo de 8 bits por caracter.
     Los scripts originales usan `bin(int(binascii.hexlify(msg), 16))`, que
     descarta los ceros a la izquierda del primer byte: 'A' se cifra en 7 bits y
     '0' en 6, de modo que los bloques "de 8 bits" y "de 16 bits" del informe
     son en realidad de 6 o 7 y de 14 o 15.
  2. El `% 1` es un parametro explicito (`mod1`). `en.py` lo aplica y
     `m1_alter.py` no, aunque el informe escribe las mismas ecuaciones (3)-(6)
     para ambos y los notebooks del ataque CBPSO lo aplican en los dos casos.
  3. La amplitud de inyeccion del mensaje A, que en los scripts esta escrita
     como la constante literal 0.01, es un campo del objeto de configuracion.

La estructura del cifrado se conserva intacta: una orbita libre (x_n, y_n) del
mapa acoplado genera el flujo de clave, y el bit numero j del mensaje se cifra
en `block_len` iteraciones del transmisor que arrancan en el indice
`stpoint + j * block_len` de esa orbita.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "GLM_CONFIG",
    "GLM_KEY",
    "LM_CONFIG",
    "LM_KEY",
    "CipherConfig",
    "Key",
    "bits_from_text",
    "block_ciphertexts",
    "cipher_table",
    "cipher_table_bruteforce",
    "divergence_profile",
    "encrypt_message",
    "extract_bit",
    "keystream",
    "orbit_median",
    "quantize_block",
    "with_mod1",
]


@dataclass(frozen=True)
class Key:
    """Llave de cifrado.

    Para `kind="lm"` la familia es f(v) = mu * v * (1 - v) y solo se usan
    (mu, mu1, alpha).

    Para `kind="glm"` la familia es el mapa logistico generalizado de Lawnik con
    exponente r = 2. ATENCION a la convencion, que en el informe esta invertida:
    en el codigo original la llamada es `log_map(x, nu, mu)`, cuyo segundo
    argumento es el UMBRAL p y cuyo tercero es la AMPLITUD q. Es decir
    p -> nu y q -> mu, al contrario de lo que dice el texto ("se redefine
    q -> nu, p -> mu"). Con la convencion del texto la orbita converge a un
    punto fijo en lugar de ser caotica.
    """

    kind: str
    mu: float
    mu1: float
    alpha: float
    nu: float | None = None
    nu1: float | None = None

    def threshold_amplitude(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Devuelve ((p1, q1), (p2, q2)) para el mapa generalizado."""
        if self.kind != "glm":
            raise ValueError("solo aplica al mapa generalizado")
        return (self.nu, self.mu), (self.nu1, self.mu1)


@dataclass(frozen=True)
class CipherConfig:
    x0: float = 0.03
    y0: float = 0.01
    stpoint: int = 4000
    block_len: int = 50
    amplitude: float = 0.01   # la constante A, literal 0.01 en los scripts
    mod1: bool = False


LM_KEY = Key(kind="lm", mu=3.1, mu1=2.9, alpha=0.3314)
GLM_KEY = Key(kind="glm", mu=0.91, mu1=0.85, alpha=0.3314, nu=0.9, nu1=0.333)

LM_CONFIG = CipherConfig(stpoint=4000, mod1=False)    # como en m1_alter.py
GLM_CONFIG = CipherConfig(stpoint=500, mod1=True)     # como en en.py


def _logistic(v: float, mu: float) -> float:
    return mu * v * (1.0 - v)


def _generalized(v: float, threshold: float, amplitude: float) -> float:
    """Mapa logistico generalizado de Lawnik con r = 2.

    `threshold` es p y `amplitude` es q en la notacion de la ecuacion (7).
    """
    if v <= threshold:
        return -amplitude / (threshold * threshold) * (threshold - v) ** 2 + amplitude
    return -amplitude / ((1.0 - threshold) ** 2) * (threshold - v) ** 2 + amplitude


def _maps(key: Key):
    if key.kind == "lm":
        return (lambda v: _logistic(v, key.mu)), (lambda v: _logistic(v, key.mu1))
    if key.kind == "glm":
        (p1, q1), (p2, q2) = key.threshold_amplitude()
        return (lambda v: _generalized(v, p1, q1)), (lambda v: _generalized(v, p2, q2))
    raise ValueError(f"tipo de mapa desconocido: {key.kind!r}")


def keystream(key: Key, cfg: CipherConfig, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Orbita libre del mapa acoplado, (x_0..x_{n-1}, y_0..y_{n-1})."""
    f1, f2 = _maps(key)
    xs = np.empty(n_points, dtype=np.float64)
    ys = np.empty(n_points, dtype=np.float64)
    x, y = cfg.x0, cfg.y0
    xs[0], ys[0] = x, y
    for i in range(1, n_points):
        xn = f1(x) + key.alpha * (y - x)
        yn = f2(y) + key.alpha * (x - y)
        if cfg.mod1:
            xn %= 1.0
            yn %= 1.0
        x, y = xn, yn
        xs[i], ys[i] = x, y
    return xs, ys


def _encrypt_block(
    key: Key,
    cfg: CipherConfig,
    xs: np.ndarray,
    ys: np.ndarray,
    index: int,
    bit: int,
) -> np.ndarray:
    """Un bloque de texto cifrado, identico a `encryp(m, pp)` de los scripts."""
    f1, _ = _maps(key)
    start = cfg.stpoint + index * cfg.block_len
    y_r = ys[start:start + cfg.block_len]
    x_t = np.empty(cfg.block_len, dtype=np.float64)
    x_t[0] = xs[start]
    for i in range(1, cfg.block_len):
        nxt = f1(x_t[i - 1]) + key.alpha * (y_r[i - 1] - x_t[i - 1])
        nxt += cfg.amplitude * bit
        if cfg.mod1:
            nxt %= 1.0
        x_t[i] = nxt
    return x_t


def encrypt_message(key: Key, cfg: CipherConfig, bits: np.ndarray) -> np.ndarray:
    """Cifra un mensaje completo bit a bit, siguiendo el flujo de los scripts.

    Devuelve (len(bits), block_len). Es la ruta lenta y literal: sirve de
    referencia contra la cual se valida `block_ciphertexts`.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    needed = cfg.stpoint + cfg.block_len * (len(bits) + 1)
    xs, ys = keystream(key, cfg, needed)
    return np.array(
        [_encrypt_block(key, cfg, xs, ys, j, int(b)) for j, b in enumerate(bits)],
        dtype=np.float64,
    )


def block_ciphertexts(key: Key, cfg: CipherConfig, n_blocks: int) -> np.ndarray:
    """Texto cifrado de cada bloque para m = 0 y m = 1, forma (n_blocks, 2, block_len).

    Que esto baste para describir el cifrado completo no es una suposicion: el
    bloque j arranca en x[stpoint + j*block_len] y su recursion solo consume
    y_r del mismo tramo y su propio bit m_j, de modo que no hay acoplamiento
    entre bloques. `cipher_table` lo verifica contra `encrypt_message`.
    """
    needed = cfg.stpoint + cfg.block_len * (n_blocks + 1)
    xs, ys = keystream(key, cfg, needed)
    out = np.empty((n_blocks, 2, cfg.block_len), dtype=np.float64)
    for j in range(n_blocks):
        for b in (0, 1):
            out[j, b] = _encrypt_block(key, cfg, xs, ys, j, b)
    return out


def bits_from_text(text: str, width: int = 8) -> np.ndarray:
    """Bits del texto claro, `width` bits por caracter, sin perder ceros a la izquierda."""
    bits = []
    for char in text:
        code = ord(char)
        if code >= 1 << width:
            raise ValueError(f"{char!r} no cabe en {width} bits")
        bits.extend((code >> (width - 1 - k)) & 1 for k in range(width))
    return np.array(bits, dtype=np.uint8)


def orbit_median(key: Key, cfg: CipherConfig, n_points: int = 200_000) -> float:
    """Mediana empirica de la orbita libre x_n.

    Es el umbral que hay que usar para cuantizar: con el umbral geometrico 1/2
    la fuente de bits queda sesgada (64 % de unos para el LM y 92 % para el
    GLM), y entonces cualquier distancia de Hamming se mide contra la linea base
    2p(1-p) en lugar de 1/2, que es distinta para cada esquema. Comparar los dos
    mapas con el mismo umbral fijo reproduce exactamente el error de saturacion
    de la metrica de distancia media del informe.
    """
    xs, _ = keystream(key, cfg, n_points)
    return float(np.median(xs[1:]))


def quantize_block(block: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Cuantiza las `block_len` muestras de un bloque a un bit por muestra.

    Con `threshold = orbit_median(...)` la fuente queda balanceada y la linea
    base de decorrelacion es 1/2 para los dos esquemas.
    """
    return (np.asarray(block) >= threshold).astype(np.uint8)


def divergence_profile(blocks: np.ndarray) -> np.ndarray:
    """|x_t(m=1) - x_t(m=0)| promediado sobre bloques, por posicion de muestra.

    Medida de avalancha sin cuantizador: no depende de ninguna eleccion de
    umbral y se compara directamente con exp(lambda_1 * i) mientras no sature.
    """
    return np.mean(np.abs(blocks[:, 1, :] - blocks[:, 0, :]), axis=0)


def extract_bit(block: np.ndarray, how: str, threshold: float = 0.5) -> int:
    """Reduce un bloque de texto cifrado a un solo bit.

    Las definiciones (8) y (9) exigen f : {0,1}^n -> {0,1}^n, de modo que hay
    que extraer exactamente un bit de salida por bit de entrada. La eleccion es
    arbitraria, asi que el analisis se repite con varias y se verifica que el
    resultado no dependa de ella:

      "last_msb"       la ultima muestra contra `threshold`
      "mean_msb"       la media del bloque contra `threshold`
      "mantissa_xor"   XOR de los 32 bits bajos de la mantisa de la ultima muestra
      "first_msb"      la PRIMERA muestra contra `threshold`; es un control,
                       porque x_t[0] = x[stpoint + j*block_len] no depende de m_j
                       y por lo tanto g_j tiene que salir constante
    """
    block = np.asarray(block)
    if how == "last_msb":
        return int(block[-1] >= threshold)
    if how == "mean_msb":
        return int(block.mean() >= threshold)
    if how == "mantissa_xor":
        raw = int(np.array([block[-1]], dtype=np.float64).view(np.uint64)[0])
        return int(bin(raw & 0xFFFFFFFF).count("1") & 1)
    if how == "first_msb":
        return int(block[0] >= threshold)
    raise ValueError(f"extractor desconocido: {how!r}")


def cipher_table_bruteforce(
    key: Key,
    cfg: CipherConfig,
    n: int,
    how: str = "last_msb",
    threshold: float = 0.5,
) -> np.ndarray:
    """Tabla de verdad de f : {0,1}^n -> {0,1}^n cifrando los 2^n textos claros.

    Ruta lenta y sin suposiciones: cada texto claro se cifra por separado con
    `encrypt_message`. Coste 2^n * n * block_len iteraciones del mapa, asi que
    es practicable hasta n ~ 12.
    """
    size = 1 << n
    table = np.zeros(size, dtype=np.int64)
    for x in range(size):
        bits = np.array([(x >> j) & 1 for j in range(n)], dtype=np.uint8)
        blocks = encrypt_message(key, cfg, bits)
        if not np.all(np.isfinite(blocks)):
            raise FloatingPointError(f"el cifrado divergio en x={x}")
        value = 0
        for j in range(n):
            value |= extract_bit(blocks[j], how, threshold) << j
        table[x] = value
    return table


def cipher_table(
    key: Key,
    cfg: CipherConfig,
    n: int,
    how: str = "last_msb",
    blocks: np.ndarray | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Tabla de verdad de f : {0,1}^n -> {0,1}^n, por la ruta de 2n evaluaciones.

    El bit j del texto claro es el de peso 2^j, y el bit j de la salida es el
    extraido del bloque j. Es la f que consumen `differential_probability` y
    `linear_probability`. Equivale a `cipher_table_bruteforce`, cosa que los
    tests comprueban para n = 8 y n = 10 en los dos esquemas.
    """
    if blocks is None:
        blocks = block_ciphertexts(key, cfg, n)
    if not np.all(np.isfinite(blocks)):
        raise FloatingPointError("el cifrado divergio: hay valores no finitos")

    out_bit = np.array(
        [[extract_bit(blocks[j, b], how, threshold) for b in (0, 1)] for j in range(n)],
        dtype=np.int64,
    )
    inputs = np.arange(1 << n, dtype=np.int64)
    table = np.zeros(1 << n, dtype=np.int64)
    for j in range(n):
        sel = (inputs >> j) & 1
        table |= out_bit[j][sel] << j
    return table


def with_mod1(cfg: CipherConfig, enabled: bool) -> CipherConfig:
    """Misma configuracion con el `% 1` activado o desactivado."""
    return replace(cfg, mod1=enabled)
