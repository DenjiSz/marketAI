"""
Archivo principal: main.py
Orquesta los módulos `infomarket.py` e `indicadores.py`, ejecuta la auto-gestión
de historial al iniciar (eliminando cualquier dato previo para comenzar desde cero)
y mantiene el bucle periódico de monitoreo de noticias e indicadores técnicos/flujo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import colorama

from indicadores import GestorIndicadores
from infomarket import EMPRESAS, GestorInfoMarket, formatear_fecha

colorama.init(autoreset=True)


# =====================================================================
# AUTO-GESTIÓN DE HISTORIAL
# =====================================================================
ARCHIVOS_HISTORIAL = [
    "historial_noticias.json",
    "historial_indicadores.json",
    "estado_mercado.json",
    "noticias_cache.db",
]

CARPETAS_HISTORIAL = [
    "historial",
    ".market_cache",
]


def autogestionar_historial(
    gestor_noticias: Optional[GestorInfoMarket] = None,
    gestor_indicadores: Optional[GestorIndicadores] = None,
    base_dir: Optional[Path] = None,
) -> int:
    """
    Auto-gestión de historial:
    Verifica si existen datos o archivos de ejecuciones previas, los elimina por completo
    del disco y reinicia los estados en memoria para comenzar desde cero.
    Devuelve la cantidad de archivos/carpetas previos eliminados.
    """
    directorio = base_dir if base_dir is not None else Path(__file__).resolve().parent
    eliminados = 0

    for nombre_archivo in ARCHIVOS_HISTORIAL:
        ruta = directorio / nombre_archivo
        if ruta.exists():
            try:
                ruta.unlink()
                eliminados += 1
            except OSError:
                pass

    for nombre_carpeta in CARPETAS_HISTORIAL:
        ruta_dir = directorio / nombre_carpeta
        if ruta_dir.exists() and ruta_dir.is_dir():
            try:
                shutil.rmtree(ruta_dir, ignore_errors=True)
                eliminados += 1
            except OSError:
                pass

    if gestor_noticias is not None:
        gestor_noticias.limpiar_historial()
    if gestor_indicadores is not None:
        gestor_indicadores.limpiar_historial()

    return eliminados


# =====================================================================
# CICLO PRINCIPAL DE MONITOREO
# =====================================================================
def ejecutar_monitor(
    intervalo_segundos: int = 10,
    max_ciclos: Optional[int] = None,
    empresas: Optional[List[str]] = None,
) -> None:
    """
    Inicializa los gestores de noticias e indicadores, limpia cualquier historial previo
    y ejecuta el bucle de escaneo cada `intervalo_segundos`.
    """
    lista_empresas = empresas if empresas is not None else list(EMPRESAS)

    gestor_noticias = GestorInfoMarket(empresas=lista_empresas)
    gestor_indicadores = GestorIndicadores(
        empresas=lista_empresas,
        gestor_noticias=gestor_noticias,
    )

    # 1. Auto-gestión de historial: eliminar datos previos y comenzar desde cero
    autogestionar_historial(gestor_noticias, gestor_indicadores)

    ciclo = 0
    try:
        while True:
            ciclo += 1
            # Escaneo de noticias (1ra vez: hasta 2 días previos; luego solo recientes sin repetir)
            gestor_noticias.escanear_y_mostrar()

            # Escaneo de indicadores, flujo de volumen (FRVP), cambios abruptos, P/E y S&R
            # (1ra vez: completo con todos los indicadores; luego solo acciones con motivo de compra/venta)
            gestor_indicadores.escanear_y_mostrar()

            if max_ciclos is not None and ciclo >= max_ciclos:
                break

            time.sleep(intervalo_segundos)
    except KeyboardInterrupt:
        ahora_str = formatear_fecha(dt.datetime.now().astimezone())
        print(f"\n{ahora_str} | Monitoreo finalizado por el usuario.")


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor en tiempo real de noticias (infomarket) e indicadores técnicos/flujo (indicadores)."
    )
    parser.add_argument(
        "--intervalo",
        type=int,
        default=10,
        help="Segundos de espera entre cada ciclo de escaneo (por defecto: 10).",
    )
    parser.add_argument(
        "--ciclos",
        type=int,
        default=None,
        help="Número máximo de ciclos a ejecutar (por defecto: infinito en bucle).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parsear_argumentos()
    ejecutar_monitor(
        intervalo_segundos=max(1, args.intervalo),
        max_ciclos=args.ciclos,
    )
