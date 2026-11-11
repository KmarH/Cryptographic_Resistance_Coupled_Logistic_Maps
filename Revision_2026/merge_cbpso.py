"""Fusiona los repartos del ataque con CBPSO escalado y dicta el veredicto.

La pregunta que decide si el articulo necesita introducir un segundo algoritmo
es si CBPSO, con el mismo presupuesto que la evolucion diferencial que si
recupero la llave, tambien la recupera. Se distingue a proposito entre lo que
logra el enjambre por si solo y lo que logra despues del pulido local, porque
son dos afirmaciones distintas y el articulo solo puede hacer la que midio.

Uso:  python merge_cbpso.py cbpso/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from cbpso_alto import CONFIGURACIONES, EXITO, PRESUPUESTO_DE, presupuesto
from maps import SCENARIOS

ORDEN = ("base", "igualada", "diez", "cien")


def cargar(directorio):
    configuraciones: dict[str, dict[int, dict]] = {}
    for archivo in sorted(Path(directorio).glob("*.json")):
        estado = json.loads(archivo.read_text(encoding="utf-8"))
        for nombre, bloque in estado.items():
            destino = configuraciones.setdefault(nombre, {})
            for corrida in bloque["runs"]:
                destino[corrida["seed"]] = corrida
    return configuraciones


def main():
    directorio = sys.argv[1] if len(sys.argv) > 1 else "cbpso"
    escenario = SCENARIOS["GLM-A"]
    verdad = np.array([escenario.truth[k] for k in escenario.free])
    configuraciones = cargar(directorio)
    if not configuraciones:
        raise SystemExit(
            f"no hay ninguna corrida en {directorio}: sin datos no hay veredicto"
        )

    # Un veredicto negativo solo vale si la configuracion mas cara llego a
    # correrse; con el barrido a medias lo unico honesto es decir que falta.
    completas = {n: c for n, c in configuraciones.items() if len(c) >= 12}
    rompe_solo = rompe_con_pulido = None
    for nombre in ORDEN:
        if nombre not in configuraciones:
            continue
        corridas = configuraciones[nombre]
        solos = np.array([c["mse_cbpso"] for c in corridas.values()])
        pulidos = np.array([c["mse_refinado"] for c in corridas.values()])
        segundos = np.array([c["seconds"] for c in corridas.values()])
        mejor = min(corridas.values(), key=lambda c: c["mse_refinado"])
        vector = np.array(mejor["position"])
        error = 100.0 * np.abs(vector - verdad) / np.abs(verdad)
        ajustes = CONFIGURACIONES[nombre]

        exitos_solo = int(np.sum(solos <= EXITO))
        exitos_pulido = int(np.sum(pulidos <= EXITO))
        if exitos_solo and rompe_solo is None:
            rompe_solo = nombre
        if exitos_pulido and rompe_con_pulido is None:
            rompe_con_pulido = nombre

        print(f"\n{nombre.upper()}  {ajustes.n_particles} particulas x "
              f"{ajustes.n_iterations} iteraciones")
        print(f"  evaluaciones        {presupuesto(ajustes):,}"
              f"  ({presupuesto(ajustes) / PRESUPUESTO_DE:.2f} veces el ataque DE)")
        print(f"  semillas            {len(corridas)}")
        print(f"  exitos, CBPSO solo  {exitos_solo}")
        print(f"  exitos, con pulido  {exitos_pulido}")
        print(f"  ECM minimo solo     {solos.min():.6e}")
        print(f"  ECM minimo pulido   {pulidos.min():.6e}")
        print(f"  ECM mediano pulido  {np.median(pulidos):.6e}")
        print(f"  segundos por corrida {segundos.mean():.0f} de media")
        print("  error % del mejor   " + np.array2string(error, precision=3))

    print("\nVEREDICTO")
    if rompe_solo:
        print(f"  CBPSO recupera la llave por si solo desde la configuracion "
              f"'{rompe_solo}'. El articulo no necesita un segundo algoritmo.")
    elif rompe_con_pulido:
        print(f"  CBPSO no cruza el umbral por si solo en ninguna configuracion, "
              f"pero si lo hace desde '{rompe_con_pulido}' cuando su mejor "
              f"resultado se pule con Nelder-Mead. Eso es un paso de refinamiento "
              f"local sobre el mismo ataque, no un optimizador rival, de modo que "
              f"el articulo puede sostenerse en CBPSO declarando ese pulido.")
    elif not completas:
        print("  Ninguna configuracion tiene todavia las 12 semillas, asi que no "
              "hay veredicto: lo medido hasta aqui solo dice que ninguna de las "
              "corridas terminadas cruzo el umbral.")
    else:
        print("  CBPSO no recupera la llave en ninguna configuracion completa, ni "
              "solo ni con pulido local, hasta "
              f"{max(completas, key=lambda n: ORDEN.index(n))}. La recuperacion "
              "observada depende entonces del ataque y no unicamente del "
              "presupuesto, y el articulo tiene que decir con que metodo se "
              "obtuvo.")
        pendientes = [n for n in configuraciones if n not in completas]
        if pendientes:
            print(f"  Siguen a medias: {', '.join(sorted(pendientes))}.")


if __name__ == "__main__":
    main()
