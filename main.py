"""
Archivo principal: main.py
Orquesta la arquitectura de 3 niveles del proyecto:
- Nivel 1A (`infomarket.py`): escaneo en tiempo real de noticias de las 89 empresas, Fed y macroeconomía.
- Nivel 1B (`indicadores.py`): evaluación de 11 indicadores, flujo institucional (FRVP/CMF),
  fundamentales (P/E, S&R) y filtro anti-errático (Kaufman ER, Choppiness CHOP, R², Whipsaw, NATR%).
- Nivel 2 y 3 (`analisis_ai.py`): correlación de eventos (Noticias + Datos + Indicadores) para
  construir el `evento_potencial_para_analisis` con todo el `contexto` histórico de la acción
  y enviarlo al modelo de IA (Gemini 3.8 Flash, Gemini 4 Argon, Gemini 4 Carbon, o Claude Sonnet/Opus).
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import time
from pathlib import Path
from typing import List, Optional

import colorama

from analisis_ai import CATALOGO_MODELOS_AI, GestorEventosAnalisis
from indicadores import GestorIndicadores
from infomarket import EMPRESAS, GestorInfoMarket, formatear_fecha

colorama.init(autoreset=True)


# =====================================================================
# AUTO-GESTIÓN DE HISTORIAL
# =====================================================================
ARCHIVOS_HISTORIAL = [
    "historial_noticias.json",
    "historial_indicadores.json",
    "eventos_potenciales_analisis.json",
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
    gestor_eventos_ai: Optional[GestorEventosAnalisis] = None,
    base_dir: Optional[Path] = None,
) -> int:
    """
    Auto-gestión de historial:
    Verifica si existen datos o archivos de ejecuciones previas, los elimina por completo
    del disco y reinicia los estados en memoria para comenzar desde cero.
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
    if gestor_eventos_ai is not None:
        gestor_eventos_ai.limpiar_historial()

    return eliminados


# =====================================================================
# CICLO PRINCIPAL DE MONITOREO (NIVELES 1, 2 Y 3)
# =====================================================================
def ejecutar_monitor(
    intervalo_segundos: int = 10,
    max_ciclos: Optional[int] = None,
    empresas: Optional[List[str]] = None,
    modelo_ai: str = "flash",
) -> None:
    """
    Inicializa los gestores de noticias, indicadores y análisis AI, limpia cualquier historial
    previo y ejecuta el bucle periódico cada `intervalo_segundos`.
    """
    lista_empresas = empresas if empresas is not None else list(EMPRESAS)

    gestor_noticias = GestorInfoMarket(empresas=lista_empresas)
    gestor_indicadores = GestorIndicadores(
        empresas=lista_empresas,
        gestor_noticias=gestor_noticias,
    )
    gestor_eventos_ai = GestorEventosAnalisis(
        gestor_noticias=gestor_noticias,
        gestor_indicadores=gestor_indicadores,
        modelo_ai=modelo_ai,
    )

    # 1. Auto-gestión de historial: eliminar datos previos y comenzar desde cero
    autogestionar_historial(gestor_noticias, gestor_indicadores, gestor_eventos_ai)

    ciclo = 0
    try:
        while True:
            ciclo += 1
            # NIVEL 1A: Escaneo de noticias (1ra vez: hasta 2 días previos; luego solo recientes sin repetir)
            nuevas_noticias = gestor_noticias.escanear_y_mostrar()

            # NIVEL 1B: Escaneo de indicadores, estabilidad anti-errática, flujo (FRVP/CMF), P/E y S&R
            gestor_indicadores.escanear_y_mostrar()

            # NIVEL 2 Y 3: Correlación (Noticias + Datos + Indicadores) -> evento_potencial_para_analisis
            # + Contexto agrupado de la acción -> Análisis con Modelo de IA
            gestor_eventos_ai.evaluar_y_generar_eventos(nuevas_noticias)

            if max_ciclos is not None and ciclo >= max_ciclos:
                break

            time.sleep(intervalo_segundos)
    except KeyboardInterrupt:
        ahora_str = formatear_fecha(dt.datetime.now().astimezone())
        print(f"\n{ahora_str} | Monitoreo finalizado por el usuario.")


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor bursátil multinivel: Noticias (infomarket) + Indicadores/Estabilidad (indicadores) + IA (analisis_ai)."
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
    parser.add_argument(
        "--modelo-ai",
        type=str,
        default="flash",
        choices=list(CATALOGO_MODELOS_AI.keys()),
        help="Modelo de IA para analizar evento_potencial_para_analisis: flash (gemini-3.8-flash), argon (gemini-4-argon), carbon (gemini-4-carbon), hydrogen (gemini-3.5-flash-lite), sonnet (claude-sonnet-4-6), opus (claude-opus-4-6).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parsear_argumentos()
    ejecutar_monitor(
        intervalo_segundos=max(1, args.intervalo),
        max_ciclos=args.ciclos,
        modelo_ai=args.modelo_ai,
    )
