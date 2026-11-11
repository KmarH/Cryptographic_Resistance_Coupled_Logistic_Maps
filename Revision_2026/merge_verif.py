"""Fusiona los repartos de la verificacion y dicta el veredicto.

El barrido se repartio entre dos maquinas y cuatro procesos. Cada uno escribio
su propio archivo con las semillas que le tocaron anotadas en cada resultado,
de modo que la fusion es por numero de semilla y no por orden de llegada. Una
semilla se corrio a proposito en las dos maquinas: si las dos dan el mismo
error cuadratico medio, queda demostrado que ambas corren el mismo experimento
y los dos subconjuntos se pueden leer como una sola muestra.

Uso:  python merge_verif.py checkpoints/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from maps import SCENARIOS

SUCCESS = 1e-6
IGUALES = 1e-12  # tolerancia relativa para declarar identicas dos corridas


def cargar(directorio):
    """Devuelve {mitad: {semilla: {"mse", "position", "fuente"}}}."""
    mitades: dict[str, dict[int, dict]] = {}
    conflictos = []
    cruzadas = []
    for archivo in sorted(Path(directorio).glob("*.json")):
        estado = json.loads(archivo.read_text(encoding="utf-8"))
        fuente = archivo.stem
        for mitad, bloque in estado.items():
            if mitad == "meta":
                continue
            destino = mitades.setdefault(mitad, {})
            for corrida in bloque["runs"]:
                semilla = corrida["seed"]
                nuevo = {
                    "mse": corrida["mse"],
                    "position": corrida["position"],
                    "fuente": fuente,
                }
                previo = destino.get(semilla)
                if previo is None:
                    destino[semilla] = nuevo
                    continue
                relativa = abs(previo["mse"] - nuevo["mse"]) / max(abs(previo["mse"]), 1e-300)
                registro = (mitad, semilla, previo["fuente"], previo["mse"],
                            nuevo["fuente"], nuevo["mse"], relativa)
                (cruzadas if relativa <= IGUALES else conflictos).append(registro)
                if nuevo["mse"] < previo["mse"]:
                    destino[semilla] = nuevo
    return mitades, cruzadas, conflictos


def resumen(corridas, verdad):
    valores = np.array([c["mse"] for c in corridas.values()])
    mejor = min(corridas.values(), key=lambda c: c["mse"])
    posicion = np.array(mejor["position"])
    return {
        "semillas": len(corridas),
        "exitos": int(np.sum(valores <= SUCCESS)),
        "min": float(valores.min()),
        "mediana": float(np.median(valores)),
        "posicion": posicion,
        "error_pct": 100.0 * np.abs(posicion - verdad) / np.abs(verdad),
    }


def main():
    directorio = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    escenario = SCENARIOS["GLM-A"]
    verdad = np.array([escenario.truth[k] for k in escenario.free])

    mitades, cruzadas, conflictos = cargar(directorio)

    print("PRUEBA CRUZADA ENTRE MAQUINAS")
    if not cruzadas and not conflictos:
        print("  no hubo ninguna semilla repetida: la prueba no se pudo hacer")
    for mitad, semilla, fa, va, fb, _vb, _rel in cruzadas:
        print(f"  {mitad} semilla {semilla}: {fa} y {fb} coinciden ({va:.6e})")
    for mitad, semilla, fa, va, fb, vb, rel in conflictos:
        print(f"  AVISO {mitad} semilla {semilla}: {fa}={va:.6e} contra "
              f"{fb}={vb:.6e} (diferencia relativa {rel:.1e})")

    filas = {}
    for mitad in ("estrechas", "completas"):
        if mitad not in mitades:
            continue
        corridas = mitades[mitad]
        faltan = sorted(set(range(12)) - set(corridas))
        filas[mitad] = resumen(corridas, verdad)
        fila = filas[mitad]
        print(f"\nCOTAS {mitad.upper()}")
        print(f"  semillas            {fila['semillas']}/12"
              + (f"  (faltan {faltan})" if faltan else ""))
        print(f"  exitos (ECM<=1e-6)  {fila['exitos']}")
        print(f"  ECM minimo          {fila['min']:.6e}")
        print(f"  ECM mediano         {fila['mediana']:.6e}")
        nombres = ", ".join(escenario.free)
        print(f"  mejor posicion      [{nombres}]")
        print("                      " + np.array2string(fila["posicion"], precision=6))
        print("  error porcentual    " + np.array2string(fila["error_pct"], precision=3))

    print("\nVEREDICTO")
    est = filas.get("estrechas")
    com = filas.get("completas")
    if est and est["exitos"]:
        print("  Con las cotas estrechas del codigo original la llave del GLM se")
        print(f"  recupera: {est['exitos']} de {est['semillas']} corridas cruzan el umbral.")
    elif est:
        print(f"  Con las cotas estrechas ninguna de las {est['semillas']} corridas cruza")
        print(f"  el umbral; la mejor queda en {est['min']:.3e}.")
    if com and com["exitos"]:
        print("  La recuperacion NO depende de esas cotas: tambien ocurre en el")
        print(f"  espacio de llaves declarado ({com['exitos']} de {com['semillas']}).")
    elif com:
        print(f"  En el espacio de llaves declarado ninguna de las {com['semillas']}")
        print(f"  corridas cruza el umbral; la mejor queda en {com['min']:.3e}.")


if __name__ == "__main__":
    main()
