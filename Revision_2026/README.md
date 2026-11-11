# Revisión 2026: código que reproduce las cifras del artículo

Esta carpeta reemplaza, para efectos de reproducibilidad, a los cuadernos de
`Original_Spanish_version/` y `ENG/`. Aquellos quedan como el registro de lo que
se hizo primero; los números del artículo en su versión corregida salen de aquí.

Dependencias: `numpy`, `scipy`, y `pytest` para las pruebas.

```bash
python -m pytest -q                                    # 32 pruebas
python merge_verif.py resultados/evolucion_diferencial # veredicto del GLM
python merge_cbpso.py resultados/cbpso_alto            # el mismo intento con CBPSO
```

Los dos últimos no vuelven a correr nada: leen los checkpoints incluidos y
reconstruyen las cifras del artículo a partir de ellos.

## Qué hay

| archivo | contenido |
|---|---|
| `diffuse.py` | Núcleo de PAD y PAL: exhaustivas, bit a bit, ecuaciones (8) y (9) de Álvarez y Li (2006). Espectro diferencial, espectro de Walsh y prueba de separabilidad bit a bit. |
| `clm_cipher.py` | Reimplementación de los dos cifrados, fiel a `m1_alter.py` y `en.py`, con el ancho de bits corregido, el `% 1` como bandera explícita y `A` como campo de configuración. |
| `maps.py` | Órbitas del mapa simple y del generalizado, escalares y vectorizadas, y la función objetivo del ataque con los cuatro escenarios. |
| `cbpso.py` | CBPSO vectorizado, con cinco correcciones documentadas en el encabezado. |
| `run_analysis.py` | Criptoanálisis diferencial y lineal; escribe `resultados/resultados_criptoanalisis.md`. |
| `run_attack.py` | Ataque con CBPSO sobre los cuatro escenarios, el control con evolución diferencial y las figuras. |
| `run_figures_map.py` | Figuras del atractor y de la señal del transmisor. |
| `verify_glm_break.py` | Ataque de presupuesto alto con evolución diferencial sobre el mapa generalizado, en dos cajas de búsqueda. |
| `cbpso_alto.py` | El mismo intento usando sólo CBPSO, con el enjambre y la duración ampliados. |
| `merge_verif.py`, `merge_cbpso.py` | Fusionan los repartos de cada barrido y dictan el veredicto. |
| `test_*.py` | Pruebas congeladas, escritas antes de la implementación. |

Los `.json` de `resultados/` son los checkpoints de los dos barridos, con la
semilla anotada en cada corrida, de modo que cualquier cifra del artículo se
puede rastrear hasta la corrida que la produjo.

## Qué cambia respecto de los cuadernos originales

### Criptoanálisis diferencial y lineal

**El núcleo está validado contra la literatura, no contra sí mismo.**
`test_diffuse.py` comprueba PAD y PAL con la caja-S de AES, cuyos valores están
publicados: uniformidad diferencial 4, de donde PAD = 4/256 = 2⁻⁶; no
linealidad 112, de donde max|W| = 32 y PAL = ((144−128)/128)² = 2⁻⁶. También
fija el comportamiento de la identidad, del mapa constante y de una permutación
de bits.

**Se calculan las definiciones, no un sustituto.** La versión original contaba
coincidencias aproximadas entre valores en punto flotante con tolerancia 10⁻²,
usaba la resta entera en lugar del XOR, incluía la columna del flujo de clave en
la comparación, y dividía por 2ⁿ un numerador cuya cota es ≈49·n. En PAL,
`umbral = 0` hacía que `abs(corr) >= umbral` fuera siempre verdadera, de modo
que el resultado era una constante que sólo depende del número de bits del
mensaje. Eso explica que los dos esquemas dieran el mismo valor.

**El resultado correcto es que estas métricas no aplican.** Ambos esquemas dan
PAD = 1 para todo tamaño de bloque y todo cuantizador, y PAL que vale 0 o 1
según el cuantizador. La razón es estructural y está verificada por fuerza
bruta: el texto claro se inyecta de forma aditiva en una sola variable de
estado y cada bit se cifra en un tramo propio del flujo de clave, de manera que
la función de cifrado es separable bit a bit, `f(x)_j = g_j(x_j)`.

### Ataque sobre el caos

**Conversión de bits.** La original descartaba los ceros a la izquierda del
primer byte, de modo que los bloques nominales de 8 bits contenían menos. Aquí
el ancho es fijo.

**Inicialización caótica.** La original generaba un escalar por partícula y lo
repetía en todas las dimensiones, lo cual confina el enjambre inicial a la
diagonal del espacio de búsqueda. Aquí se genera por partícula y dimensión. El
mismo error estaba en la búsqueda caótica.

**Rango de la búsqueda caótica.** La original lo escalaba por el valor del mejor
global, `a * global_best[i]`, lo cual anula el rango para los parámetros cuyo
óptimo está cerca de cero. Aquí se escala por el ancho de la cota.

**Métrica de distancia.** La original la medía sobre trayectorias largas, donde
la divergencia exponencial propia del caos la satura: devuelve el mismo valor
para un ataque exitoso y para uno fallido. Aquí se mide sobre los mismos N = 30
puntos que define la función objetivo.

**Escenarios simétricos.** La versión original daba al atacante del mapa simple
unas condiciones iniciales fijas y al del generalizado unas cotas más estrechas
que su propio dominio declarado. Aquí los cuatro escenarios son simétricos, y
las cotas estrechas se conservan sólo como caso de comparación explícito en
`verify_glm_break.py`.

**Cuantización al calcular avalancha y sensibilidad de llave.** Umbral fijo en
0,5 sobre un flujo con 64 % y 92 % de unos mide el sesgo, no la difusión. Aquí
se cuantiza en la mediana de la órbita.

## Qué se midió

Mapa simple acoplado: CBPSO recupera la llave en los 100 ciclos, y la evolución
diferencial de control en la totalidad de las corridas de los dos escenarios, de
manera que la brecha no depende de conocer las condiciones iniciales.

Mapa generalizado acoplado, con el presupuesto original: ni CBPSO ni el control
cruzan el umbral de 10⁻⁶, y se detienen en el mismo orden de magnitud.

Mapa generalizado acoplado, con presupuesto alto:

| ataque | evaluaciones | ciclos equivalentes | éxitos | mejor ECM |
|---|---|---|---|---|
| Evolución diferencial + refinamiento local, cotas estrechas | 1,80×10⁶ | 7,2 | 1 de 12 | 6,64×10⁻²³ |
| Evolución diferencial + refinamiento local, dominio completo | 1,80×10⁶ | 7,2 | 1 de 12 | 1,74×10⁻²⁵ |
| CBPSO ampliado + el mismo refinamiento, dominio completo | 1,80×10⁸ | 717 | 0 de 12 | 1,54×10⁻⁴ |

La llave del mapa generalizado se recupera, y no sólo dentro de las cotas
estrechas del código original sino también en el espacio de llaves declarado.
CBPSO no lo consigue ni con cien veces el presupuesto del ataque que sí lo
logra: mejora 27 veces respecto de los 100 ciclos originales y se detiene tres
órdenes de magnitud por encima del umbral. Lo que rompe al mapa generalizado es
el método de búsqueda, no el tamaño del presupuesto.

## Sobre los dos barridos

Cada barrido se repartió entre dos máquinas, con checkpoint después de cada
corrida; de ahí que los archivos de `resultados/` lleven el nombre de la
arquitectura en que corrieron. Una semilla se corrió a propósito en las dos:
dio `2.871996e-02` en ambas, hasta el último dígito, pese a correr sobre
aarch64 y sobre x86_64 con versiones distintas de Python y las mismas de numpy
y scipy. Eso es lo que autoriza a leer los repartos como una sola muestra, y
`merge_verif.py` vuelve a comprobarlo cada vez que se ejecuta.
