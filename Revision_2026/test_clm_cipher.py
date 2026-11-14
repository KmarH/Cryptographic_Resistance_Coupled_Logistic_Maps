"""Tests de la reimplementacion de los cifrados.

Dos cosas que hay que comprobar antes de creerle cualquier numero al analisis:

  1. Que la ruta rapida (2n evaluaciones) coincide con cifrar los 2^n textos
     claros uno por uno. Si coinciden, los bloques son independientes; si no,
     la ruta rapida es invalida.
  2. Que para m = 0 el texto cifrado es exactamente el tramo de la orbita libre.
     Es consecuencia de que x_t[0] = x[stpoint + j*block_len] y de que la
     recursion del transmisor es la misma que la del generador cuando A*m = 0.
     Esto significa que todo bit 0 de cualquier mensaje entrega al atacante
     `block_len` puntos limpios del mapa.

Ejecutar:  python -m pytest test_clm_cipher.py -q
"""

import numpy as np
import pytest

from clm_cipher import (
    GLM_CONFIG,
    GLM_KEY,
    LM_CONFIG,
    LM_KEY,
    bits_from_text,
    cipher_table,
    cipher_table_bruteforce,
    encrypt_message,
    keystream,
    with_mod1,
)

SCHEMES = [("lm", LM_KEY, LM_CONFIG), ("glm", GLM_KEY, GLM_CONFIG)]


@pytest.mark.parametrize(("name", "key", "cfg"), SCHEMES)
@pytest.mark.parametrize("how", ["last_msb", "mean_msb", "mantissa_xor"])
def test_fast_table_matches_bruteforce(name, key, cfg, how):
    fast = cipher_table(key, cfg, 8, how)
    slow = cipher_table_bruteforce(key, cfg, 8, how)
    assert np.array_equal(fast, slow)


@pytest.mark.parametrize(("name", "key", "cfg"), SCHEMES)
def test_zero_bit_block_is_the_bare_keystream(name, key, cfg):
    n = 6
    xs, _ = keystream(key, cfg, cfg.stpoint + cfg.block_len * (n + 1))
    blocks = encrypt_message(key, cfg, np.zeros(n, dtype=np.uint8))
    for j in range(n):
        start = cfg.stpoint + j * cfg.block_len
        np.testing.assert_allclose(
            blocks[j], xs[start:start + cfg.block_len], rtol=0, atol=0,
        )


@pytest.mark.parametrize(("name", "key", "cfg"), SCHEMES)
def test_ciphertext_stays_finite(name, key, cfg):
    blocks = encrypt_message(key, cfg, np.ones(16, dtype=np.uint8))
    assert np.all(np.isfinite(blocks))


def test_lm_mod1_is_inactive_at_the_published_keys():
    """El `% 1` que `m1_alter.py` omite y `en.py` aplica no cambia la orbita del LM."""
    n = 16
    plain = bits_from_text("A1")
    off = encrypt_message(LM_KEY, with_mod1(LM_CONFIG, False), plain)
    on = encrypt_message(LM_KEY, with_mod1(LM_CONFIG, True), plain)
    np.testing.assert_allclose(off, on, rtol=0, atol=0)
    assert len(plain) == n


def test_bits_from_text_keeps_leading_zeros():
    """La conversion de los scripts originales pierde bits; esta no."""
    import binascii

    for text, expected in [("A", 8), ("0", 8), ("AA", 16), ("01", 16)]:
        assert len(bits_from_text(text)) == expected
        original = bin(int(binascii.hexlify(text.encode()), 16))[2:]
        assert len(original) < expected  # el defecto que se corrige
