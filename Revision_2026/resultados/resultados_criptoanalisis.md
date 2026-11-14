# Criptoanalisis diferencial y lineal de los mapas logisticos acoplados

Salida de `run_analysis.py`. PAD y PAL se calculan sobre el espacio de entrada completo con las definiciones de Alvarez y Li (2006), ecuaciones (8) y (9) del informe.

## A. PAD y PAL exhaustivas segun las ecuaciones (8) y (9)

Espacio de entrada completo Omega = {0,1}^n, XOR sobre bits, sin tolerancias de punto flotante y sin normalizaciones ajenas al tamano de bloque. El nucleo esta validado contra la S-box de AES, cuyos valores conocidos son PAD = PAL = 2^-6 = 0.015625.

| esquema | n | extractor | PAD | dx | dy | PAL | a | b |
|---|---|---|---|---|---|---|---|---|
| LM | 8 | `last_msb` | 1 | 1 | 0 | 0 | 1 | 1 |
| LM | 8 | `mean_msb` | 1 | 1 | 0 | 0 | 1 | 1 |
| LM | 8 | `mantissa_xor` | 1 | 1 | 0 | 1 | 2 | 2 |
| LM | 8 | `first_msb` | 1 | 1 | 0 | 0 | 1 | 1 |
| LM | 12 | `last_msb` | 1 | 1 | 0 | 1 | 1024 | 1024 |
| LM | 12 | `mean_msb` | 1 | 1 | 0 | 0 | 1 | 1 |
| LM | 12 | `mantissa_xor` | 1 | 1 | 0 | 1 | 2 | 2 |
| LM | 12 | `first_msb` | 1 | 1 | 0 | 0 | 1 | 1 |
| GLM | 8 | `last_msb` | 1 | 1 | 0 | 0 | 1 | 1 |
| GLM | 8 | `mean_msb` | 1 | 1 | 0 | 1 | 32 | 32 |
| GLM | 8 | `mantissa_xor` | 1 | 1 | 1 | 1 | 1 | 1 |
| GLM | 8 | `first_msb` | 1 | 1 | 0 | 0 | 1 | 1 |
| GLM | 12 | `last_msb` | 1 | 1 | 0 | 0 | 1 | 1 |
| GLM | 12 | `mean_msb` | 1 | 1 | 0 | 1 | 32 | 32 |
| GLM | 12 | `mantissa_xor` | 1 | 1 | 1 | 1 | 1 | 1 |
| GLM | 12 | `first_msb` | 1 | 1 | 0 | 0 | 1 | 1 |

Contraste con lo publicado:

| esquema | n | PAD informe | PAD correcta | PAL informe | PAL correcta |
|---|---|---|---|---|---|
| GLM | 8 | 0.0234375 | 1 | 0.000724498 | 0 |
| LM | 8 | 0.015625 | 1 | 0.000724498 | 0 |

PAD vale 1 en los dos esquemas, en todos los tamanos de bloque y con todos los extractores. PAL vale 1 o 0 segun el extractor, es decir el estadistico ni siquiera esta bien definido para este cifrado. Ninguno de los dos valores publicados es reproducible: eran el conteo de coincidencias en punto flotante dividido por 2^n en el caso de PAD, y una constante que solo depende del numero de bits del mensaje en el caso de PAL.

## B. Por que los valores salen degenerados

- **LM**, n = 12: f separable bit a bit, f(x)_j = g_j(x_j): **True**; 7 de 12 bits de salida dependen biyectivamente de su bit de entrada
- **GLM**, n = 12: f separable bit a bit, f(x)_j = g_j(x_j): **True**; 7 de 12 bits de salida dependen biyectivamente de su bit de entrada

Si f es separable bit a bit, f(x) ^ f(x ^ dx) no depende de x sino solo de dx, asi que para cada dx toda la masa del espectro diferencial cae en un unico dy y PAD = 1 exactamente. Si ademas algun g_j es biyectivo, las mascaras a = b = e_j dan una identidad que se cumple para los 2^n textos claros y PAL = 1 exactamente.

La causa esta en la estructura del cifrado, no en la eleccion de cuantizador. El bloque j arranca en x[stpoint + j*block_len] de la orbita libre y su recursion solo consume ese tramo del flujo de clave y su propio bit m_j, de modo que no hay difusion entre bits del texto claro: cambiar un bit del mensaje altera 50 muestras del criptograma y deja todas las demas bit a bit identicas. Esto se verifica cifrando los 2^n textos claros uno por uno (`cipher_table_bruteforce`) y comparando con las 2n evaluaciones de `cipher_table`.

PAD y PAL estan definidas para cifrados de bloque. Aqui no miden la fortaleza del esquema: miden que el esquema no es un cifrado de bloque, sino una modulacion por enmascaramiento caotico. La seccion de criptoanalisis diferencial y lineal del informe debe retirarse o reescribirse con este resultado, y las nociones de seguridad pertinentes son las de las partes C y D junto con la resistencia a estimacion de parametros que el informe ya mide.

Corolario que refuerza el ataque CBPSO: con m_j = 0 el bloque es exactamente el tramo de la orbita libre, porque x_t[0] = x[stpoint + j*block_len] y la recursion del transmisor coincide con la del generador cuando A*m = 0. Todo bit 0 de cualquier mensaje entrega 50 puntos limpios del mapa. El informe supone que el atacante puede fijar m(n) = 0; en realidad no necesita fijar nada.

## C. Sustitutos que si discriminan los dos mapas

Avalancha intrabloque: fraccion de los `block_len` bits cuantizados de un bloque que cambian al invertir su bit de texto claro, restringida al unico bloque que ese bit puede afectar. Valor ideal 0.5. Cuantizacion a la mediana de cada orbita, de modo que la linea base de decorrelacion es 0.5 para los dos esquemas.

| esquema | umbral (mediana) | avalancha media | desv. est. | min | max | lambda_1 | lambda_2 | lambda_cond |
|---|---|---|---|---|---|---|---|---|
| LM | 0.6460 | 0.1463 | 0.1184 | 0.0000 | 0.4600 | +0.1930 | -0.1560 | +0.0607 |
| GLM | 0.7686 | 0.0512 | 0.0231 | 0.0000 | 0.0800 | +0.5185 | -1.0809 | -1.0924 |

lambda_1 y lambda_2 son los exponentes del mapa acoplado 2D, en nats por iteracion, sobre la orbita libre, 2x10^5 iteraciones con reortonormalizacion QR. El informe los menciona como criterio pero no reporta ninguno. Notese que mu = 3.1 y mu1 = 2.9 situan a cada mapa logistico aislado en regimen periodico (periodo 2 y punto fijo respectivamente): el caos del esquema simple lo produce el acoplamiento.

lambda_cond es el exponente condicional del transmisor, con el tramo de flujo de clave y_r fijo. Es el que gobierna la avalancha intrabloque, y explica por que el GLM la tiene mas baja que el LM a pesar de tener un lambda_1 mas de dos veces mayor: el bit del mensaje entra en una recursion forzada cuya tasa de crecimiento es distinta de la del sistema acoplado completo. Es la distincion que hay que hacer explicita en el informe, porque el argumento sobre difusion esta planteado sobre el exponente equivocado.

Perfil de divergencia |x_t(m=1) - x_t(m=0)| promediado sobre bloques, sin cuantizador, en las posiciones de muestra indicadas:

| esquema | i=1 | i=2 | i=5 | i=10 | i=20 | i=30 | i=40 | i=49 |
|---|---|---|---|---|---|---|---|---|
| LM | 1.000e-02 | 9.179e-03 | 1.773e-02 | 3.973e-02 | 1.058e-01 | 7.748e-02 | 1.288e-01 | 9.251e-02 |
| GLM | 1.000e-02 | 1.912e-02 | 3.678e-02 | 2.388e-02 | 3.642e-02 | 1.942e-02 | 2.577e-02 | 2.391e-02 |

La inyeccion vale A = 0.01, valor que el informe nunca declara. El perfil arranca ahi y satura cuando alcanza la escala del atractor; el numero de iteraciones que tarda es la cantidad que mide la difusion real del esquema.

Estadistica de la orbita libre cuantizada. No es el criptograma, que es analogico, pero cualquier implementacion practica tiene que cuantizar, asi que acota lo que se puede esperar como fuente de bits. Con umbral 1/2 en lugar de la mediana, la proporcion de unos es 0.6417 para el LM y 0.9182 para el GLM.

| esquema | proporcion de unos | corr. serial lag-1 | z de rachas | entropia de palabras de 8 bits |
|---|---|---|---|---|
| LM | 0.5000 | -0.7543 | +243.80 | 3.4453 / 8 |
| GLM | 0.5000 | -0.2708 | +33.49 | 6.8057 / 8 |

## D. Sensibilidad a la llave

Distancia de Hamming relativa entre criptogramas cuantizados del mismo mensaje cifrado con la llave original y con un parametro perturbado en delta. Es la nocion de difusion que corresponde a un cifrado de flujo por enmascaramiento, y conecta con el ataque CBPSO: 0.5 a delta pequeno significa que un error minimo en la llave estimada arruina el descifrado. Cuantizacion a la mediana, asi que 0.5 es la linea base de decorrelacion en ambos casos.

| esquema | parametro | delta=0.001 | delta=1e-06 | delta=1e-09 | delta=1e-12 |
|---|---|---|---|---|---|
| LM | mu | 0.6050 | 0.5550 | 0.5062 | 0.3725 |
| LM | mu1 | 0.4988 | 0.4775 | 0.6088 | 0.5425 |
| LM | alpha | 0.4462 | 0.5200 | 0.5800 | 0.4600 |
| GLM | mu | 0.4813 | 0.4825 | 0.4863 | 0.4738 |
| GLM | mu1 | 0.4700 | 0.5050 | 0.5275 | 0.5300 |
| GLM | alpha | 0.5088 | 0.4888 | 0.4963 | 0.4963 |
| GLM | nu | 0.5513 | 0.4825 | 0.4688 | 0.4800 |
| GLM | nu1 | 0.5288 | 0.4900 | 0.5012 | 0.5275 |

