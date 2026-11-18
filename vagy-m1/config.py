# -*- coding: utf-8 -*-
"""
config.py
=========
Configuración centralizada del proyecto VAGY-M1:
- Paleta de colores ANSI / TrueColor y utilidades de formato de terminal en Windows.
- Utilidades seguras de consola que NO alteran `ENABLE_VIRTUAL_TERMINAL_INPUT` ni bloquean
  pestañas de Windows Terminal / PowerShell al ejecutar ni al salir.
- Listas oficiales de acciones (Lista 1 y Lista 2), equivalencias Yahoo Finance y mapa sectorial
  para correlación de noticias interrelacionadas (Semiconductores/IA/RAM, Óptica, Finanzas, Salud, etc.).
- Catálogo ampliado de 10 modelos IA con sus niveles oficiales de esfuerzo (`thinking_level` / `effort`):
  Flash 3.6/3.7/3.8, Argon, Hydrogen, Carbon, Pro 3.1, Sonnet, Opus y Fable.
- Detección dinámica de régimen horario bursátil (Pre-market apertura, Apertura, Cierre vs Media sesión/Cerrado).
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Almacena los modos Win32 originales de la consola para restaurarlos intactos al salir
_MODO_STDIN_ORIGINAL: Optional[int] = None
_MODO_STDOUT_ORIGINAL: Optional[int] = None
_PATRON_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


# ==============================================================================
# 1. ACTIVACIÓN DE SOPORTE ANSI / VT100 Y GESTIÓN SEGURA DE CONSOLA EN WINDOWS
# ==============================================================================

def activar_ansi_windows() -> None:
    """
    Habilita procesamiento VT100 / TrueColor 24-bit y UTF-8 guardando los modos Win32
    originales de stdin/stdout para restaurarlos al cerrar sin dañar PowerShell ni Windows Terminal.
    """
    global _MODO_STDIN_ORIGINAL, _MODO_STDOUT_ORIGINAL
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)

            handle_in = kernel32.GetStdHandle(-10)
            mode_in = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle_in, ctypes.byref(mode_in)):
                if _MODO_STDIN_ORIGINAL is None:
                    _MODO_STDIN_ORIGINAL = int(mode_in.value)

            handle_out = kernel32.GetStdHandle(-11)
            mode_out = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle_out, ctypes.byref(mode_out)):
                if _MODO_STDOUT_ORIGINAL is None:
                    _MODO_STDOUT_ORIGINAL = int(mode_out.value)
                # ENABLE_PROCESSED_OUTPUT (0x0001) | ENABLE_WRAP_AT_EOL (0x0002) | ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)
                kernel32.SetConsoleMode(handle_out, mode_out.value | 0x0004 | 0x0001 | 0x0002)
        except Exception:
            pass

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def desactivar_quickedit_consola() -> None:
    """
    Desactiva únicamente QuickEdit Mode (0x0040) manteniendo ENABLE_EXTENDED_FLAGS (0x0080)
    para evitar pausas accidentales por clic sin tocar `ENABLE_VIRTUAL_TERMINAL_INPUT` (0x0200),
    `ENABLE_LINE_INPUT` ni `ENABLE_ECHO_INPUT`.
    """
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle_in = kernel32.GetStdHandle(-10)
        mode_in = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle_in, ctypes.byref(mode_in)):
            nuevo = (mode_in.value | 0x0080) & ~0x0040
            kernel32.SetConsoleMode(handle_in, nuevo)
    except Exception:
        pass


def configurar_consola_tui_interactiva() -> bool:
    """
    Verifica si `STD_INPUT_HANDLE` corresponde a una consola interactiva real en Windows
    y desactiva únicamente QuickEdit conservando intactos `ENABLE_PROCESSED_INPUT` (0x0001)
    y `ENABLE_VIRTUAL_TERMINAL_INPUT` (0x0200) para jamás corromper ConPTY / PowerShell PSReadLine.
    """
    if os.name != "nt":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        handle_in = kernel32.GetStdHandle(-10)
        mode_in = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle_in, ctypes.byref(mode_in)):
            return False
        # No desactivar 0x0200 ni LINE/ECHO: usamos msvcrt/lectura segura sobre el buffer actual
        # y únicamente garantizamos PROCESSED_INPUT (Ctrl+C rápido) y no-QuickEdit.
        nuevo = (mode_in.value | 0x0001 | 0x0080) & ~0x0040
        kernel32.SetConsoleMode(handle_in, nuevo)
        return True
    except Exception:
        return False


def restaurar_consola_windows(resetear_vt: bool = True) -> None:
    """
    Restaura íntegramente los modos Win32 originales de entrada/salida, vacía el buffer de
    entrada pendiente (para evitar teclas fantasmas al volver a PowerShell) y restablece
    auto-wrap, cursor visible y atributos de color sin emitir `\x1b[r` (evitando regresar el cursor a 1;1).
    """
    if resetear_vt:
        try:
            sys.stdout.write("\x1b[?7h\x1b[?25h\x1b[0m")
            sys.stdout.flush()
        except Exception:
            pass

    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle_in = kernel32.GetStdHandle(-10)
        if _MODO_STDIN_ORIGINAL is not None:
            kernel32.SetConsoleMode(handle_in, _MODO_STDIN_ORIGINAL)
        else:
            mode_in = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle_in, ctypes.byref(mode_in)):
                # Garantizar modos estándar saludables si no se guardó copia previa
                sano = mode_in.value | 0x0001 | 0x0002 | 0x0004
                kernel32.SetConsoleMode(handle_in, sano)
        try:
            kernel32.FlushConsoleInputBuffer(handle_in)
        except Exception:
            pass

        if _MODO_STDOUT_ORIGINAL is not None:
            handle_out = kernel32.GetStdHandle(-11)
            kernel32.SetConsoleMode(handle_out, _MODO_STDOUT_ORIGINAL | 0x0004 | 0x0001 | 0x0002)
    except Exception:
        pass


def leer_portapapeles_windows() -> str:
    """Lee texto Unicode del portapapeles de Windows (soporte 64-bit seguro para Ctrl+V)."""
    if os.name != "nt":
        return ""
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.OpenClipboard.argtypes = [ctypes.c_void_p]
        user32.OpenClipboard.restype = ctypes.c_int
        user32.GetClipboardData.argtypes = [ctypes.c_uint]
        user32.GetClipboardData.restype = ctypes.c_void_p
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = ctypes.c_int
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.restype = ctypes.c_int

        if not user32.OpenClipboard(None):
            return ""
        try:
            h_data = user32.GetClipboardData(13)  # CF_UNICODETEXT = 13
            if not h_data:
                return ""
            ptr = kernel32.GlobalLock(h_data)
            if not ptr:
                return ""
            try:
                texto = ctypes.wstring_at(ptr)
                return texto.replace("\r", " ").replace("\n", " ").strip()
            finally:
                kernel32.GlobalUnlock(h_data)
        finally:
            user32.CloseClipboard()
    except Exception:
        return ""


def longitud_visible_ansi(texto: str) -> int:
    """Devuelve la cantidad real de columnas visibles ignorando códigos de escape ANSI."""
    return len(_PATRON_ANSI.sub("", texto))


def recortar_ansi_visible(texto: str, max_cols: int) -> str:
    """
    Trunca una cadena con códigos de color ANSI a un máximo exacto de `max_cols`
    columnas visibles sin romper secuencias de escape ni provocar saltos de línea.
    """
    if max_cols <= 0:
        return ""
    if longitud_visible_ansi(texto) <= max_cols:
        return texto

    limite_chars = max(1, max_cols - 1)
    partes: List[str] = []
    visibles = 0
    pos = 0
    n = len(texto)

    while pos < n and visibles < limite_chars:
        m = _PATRON_ANSI.match(texto, pos)
        if m:
            partes.append(m.group(0))
            pos = m.end()
        else:
            partes.append(texto[pos])
            visibles += 1
            pos += 1

    partes.append("…\x1b[0m")
    return "".join(partes)


# ==============================================================================
# 2. SECUENCIAS DE COLOR ANSI Y TRUECOLOR (24-BIT)
# ==============================================================================

class Colores:
    """Códigos ANSI estándar y TrueColor para resaltado exacto en las 4 terminales."""

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"

    # Colores específicos requeridos por especificación
    MORADO = "\x1b[38;2;195;110;255m"         # Ticker principal IA + Leyenda Supervisión AI
    AZUL = "\x1b[38;2;75;165;255m"            # Ticker noticias + Leyenda Supervisión Noticias
    CIAN = "\x1b[38;2;80;220;235m"            # Encabezados e información de sistema
    BLANCO = "\x1b[38;2;240;240;245m"         # Texto general limpio
    GRIS = "\x1b[38;2;145;150;165m"           # Separadores y etiquetas secundarias

    # Escala semántica de evaluación e indicadores
    VERDE_FUERTE = "\x1b[1;38;2;0;255;115m"   # Compra fuerte (verde intenso brillante)
    VERDE = "\x1b[38;2;65;205;95m"            # Compra / Alcista / Ganancia / Noticia positiva
    AMARILLO = "\x1b[38;2;245;215;65m"        # Neutral / Advertencia de latencia
    NARANJA = "\x1b[38;2;255;145;35m"         # Venta (naranja real TrueColor)
    ROJO = "\x1b[1;38;2;255;65;65m"           # Venta fuerte / Bajista / Pérdida / Noticia negativa

    # Resaltados exactos de leyendas de supervisión solicitados
    BG_LEYENDA_AI = "\x1b[48;2;115;38;175;38;2;255;255;255m"         # Fondo morado + texto blanco legible
    BG_LEYENDA_NOTICIAS = "\x1b[48;2;26;98;195;38;2;255;255;255m"    # Fondo azul + texto blanco legible

    # Fondos sutiles para barras de supervisión en terminal principal
    BG_SUPERVISOR_AI = "\x1b[48;2;24;22;40m"
    BG_SUPERVISION_NEWS = "\x1b[48;2;18;30;44m"
    BG_PROMPT = "\x1b[48;2;32;26;44m"


# ==============================================================================
# 3. FUNCIONES DE FORMATO EXACTO REQUERIDAS POR EL USUARIO
# ==============================================================================

def formatear_fecha_hora_ampm(dt: datetime | None = None) -> str:
    """Devuelve fecha y hora con segundos en formato AM/PM: YYYY-MM-DD hh:mm:ss AM/PM."""
    objetivo = dt or datetime.now()
    return objetivo.strftime("%Y-%m-%d %I:%M:%S %p")


def formatear_hora_ampm(dt: datetime | None = None) -> str:
    """Devuelve únicamente la hora con segundos en formato AM/PM: hh:mm:ss AM/PM."""
    objetivo = dt or datetime.now()
    return objetivo.strftime("%I:%M:%S %p")


def ticker_morado(ticker: str) -> str:
    """
    Formatea [ticker] resaltando SOLO el ticker en morado, dejando los corchetes sin resaltar.
    Usado en el encabezado de predicción de la terminal principal.
    """
    return f"{Colores.RESET}[{Colores.MORADO}{Colores.BOLD}{ticker}{Colores.RESET}]"


def ticker_azul(ticker: str) -> str:
    """
    Formatea [ticker] resaltando SOLO el ticker en azul, dejando los corchetes sin resaltar.
    Usado en las líneas de noticias tanto de la terminal principal como de /mkinfo.
    """
    return f"{Colores.RESET}[{Colores.AZUL}{Colores.BOLD}{ticker}{Colores.RESET}]"


def ticker_cartera_pyg(ticker: str, pyg_pct: float) -> str:
    """
    Formatea [ticker] para la cartera virtual (/car-v):
    verde si hay ganancias (> 0), rojo si hay pérdidas (< 0), amarillo si es 0.00%,
    manteniendo los corchetes sin resaltar.
    """
    if pyg_pct > 0.0001:
        color = Colores.VERDE_FUERTE
    elif pyg_pct < -0.0001:
        color = Colores.ROJO
    else:
        color = Colores.AMARILLO
    return f"{Colores.RESET}[{color}{ticker}{Colores.RESET}]"


def leyenda_supervision_ai(compacta: bool = False) -> str:
    """Devuelve la leyenda `Supervisión AI` resaltada en morado."""
    etiqueta = "Supervisión AI" if not compacta else "Sup. AI"
    return f"{Colores.BG_LEYENDA_AI}{Colores.BOLD} {etiqueta} {Colores.RESET}"


def leyenda_supervision_noticias(compacta: bool = False) -> str:
    """Devuelve la leyenda `Supervisión Noticias` resaltada en azul."""
    etiqueta = "Supervisión Noticias" if not compacta else "Sup. Noticias"
    return f"{Colores.BG_LEYENDA_NOTICIAS}{Colores.BOLD} {etiqueta} {Colores.RESET}"


def colorear_evaluacion(evaluacion: str) -> str:
    """
    Aplica el color exacto de texto para la evaluación de indicadores:
    - COMPRA FUERTE -> verde fuerte
    - COMPRA        -> verde
    - NEUTRAL       -> amarillo
    - VENTA         -> naranja
    - VENTA FUERTE  -> rojo
    """
    ev_upper = evaluacion.strip().upper()
    mapa = {
        "COMPRA FUERTE": Colores.VERDE_FUERTE,
        "COMPRA": Colores.VERDE,
        "NEUTRAL": Colores.AMARILLO,
        "VENTA": Colores.NARANJA,
        "VENTA FUERTE": Colores.ROJO,
    }
    color = mapa.get(ev_upper, Colores.BLANCO)
    return f"{color}{evaluacion}{Colores.RESET}"


def colorear_nombre_indicador(nombre: str, alarma: str | None) -> str:
    """
    Resalta el nombre del indicador si disparó un evento/alarma:
    verde si es alarma de COMPRA, rojo si es alarma de VENTA.
    """
    if alarma == "COMPRA":
        return f"{Colores.VERDE_FUERTE}{nombre}{Colores.RESET}"
    if alarma == "VENTA":
        return f"{Colores.ROJO}{nombre}{Colores.RESET}"
    return f"{Colores.BLANCO}{nombre}{Colores.RESET}"


def colorear_plataforma_noticia(plataforma: str, sentimiento: str) -> str:
    """
    Resalta la plataforma en verde si la noticia es positiva y en rojo si es negativa
    (usado al listar noticias analizadas en la terminal principal).
    """
    sent = sentimiento.strip().upper()
    if sent == "POSITIVA":
        return f"{Colores.VERDE}{plataforma}{Colores.RESET}"
    if sent == "NEGATIVA":
        return f"{Colores.ROJO}{plataforma}{Colores.RESET}"
    return f"{Colores.AMARILLO}{plataforma}{Colores.RESET}"


# ==============================================================================
# 4. LISTAS DE ACCIONES, YFINANCE Y GRAFO SECTORIAL INTERRELACIONADO
# ==============================================================================

LISTA_ACCIONES_1: List[str] = [
    "PANW", "LITE", "CLS", "STRL", "CIEN", "AGX", "MU", "WDC", "VIAV", "FIX",
    "SNDK", "NVDA", "STX", "PWR", "LINC", "POWL", "HWM", "ENLT", "EZPW", "RBC",
    "GLW", "CAT", "TTMI", "VRT", "CECO", "ALM", "CRS", "FTI", "TSEM", "MOD",
    "FTAI", "IESC", "COHR", "JBL", "GTX", "KLAC", "CW", "FEIM", "IBKR", "CPRX",
    "ESLT", "LRCX", "ADEA", "ASX", "COKE", "SNEX", "CASY", "BWAY", "BTSG", "ENVA",
    "BLX", "MUFG", "WELL", "APG", "TSM", "FCFS", "SMFG", "ESE", "WMT", "AAMI",
    "HLT", "AIT", "CDNL", "PLPC", "PL", "NVT", "STLD", "AXSM", "WWD", "SEI",
    "CAH", "ODC", "CCJ", "ATI", "MFG", "MCK", "ANAB", "TKO", "MIRM", "KNSA",
    "UBS", "AEIS", "R", "NGS", "GS", "NOK", "ADI", "ITRN", "AER", "AMD", "GOOGL",
]

LISTA_ACCIONES_2: List[str] = [
    "GOOGL", "ASML", "LLY", "AVGO", "V", "NVDA", "ADI", "AMAT", "KLAC", "GEV",
    "GLW", "CSCO", "ABBV", "CEG", "JMP", "AAPL", "BRKB", "NU", "SKHY", "LRCX",
    "MPWR", "TSEM", "CIEN", "CRWD", "CRM", "IBKR", "AMGN", "GIL", "MRK", "TXN",
    "SONY", "SNDK", "PLTR", "CRDO", "ALAB", "TER", "NOW", "TM", "BX", "ROK",
    "DDOG", "SNOW", "COHR", "LITE", "MRVL", "NBIS", "MP", "FLEX", "QCOM",
]

# Equivalencias de símbolos para descarga en Yahoo Finance sin alterar el ticker mostrado al usuario
MAPA_SIMBOLOS_YFINANCE: Dict[str, str] = {
    "BRKB": "BRK-B",
}


def simbolo_yfinance(ticker_usuario: str) -> str:
    """Traduce el ticker de la lista al símbolo exacto requerido por Yahoo Finance."""
    return MAPA_SIMBOLOS_YFINANCE.get(ticker_usuario.upper(), ticker_usuario.upper())


def ticker_desde_yfinance(simbolo_yf: str) -> str:
    """Revierte un símbolo de Yahoo Finance al nombre original en la lista del usuario."""
    inverso = {v: k for k, v in MAPA_SIMBOLOS_YFINANCE.items()}
    return inverso.get(simbolo_yf.upper(), simbolo_yf.upper())


# Clústeres sectoriales e interrelación de cadenas de suministro (para análisis en lote y noticias interrelacionadas)
MAPA_SECTORES_TICKERS: Dict[str, List[str]] = {
    "SEMICONDUCTORES_IA": [
        "NVDA", "AMD", "AVGO", "TSM", "ASML", "LRCX", "KLAC", "AMAT", "TSEM",
        "MRVL", "CRDO", "ALAB", "MPWR", "ADI", "TXN", "QCOM", "TER", "ASX",
        "AEIS", "FEIM", "ADEA", "NBIS", "PLTR", "CLS", "JBL", "FLEX",
    ],
    "MEMORIA_ALMACENAMIENTO": [
        "MU", "WDC", "SNDK", "SKHY", "STX", "LRCX", "AMAT", "KLAC", "TSM", "NVDA",
    ],
    "OPTICA_RED_DATACENTER": [
        "LITE", "COHR", "CIEN", "VIAV", "GLW", "CSCO", "VRT", "PWR", "GEV",
        "CEG", "MOD", "TTMI", "CLS", "JBL", "FLEX", "NOK", "STRL", "FIX",
        "POWL", "IESC", "PLPC", "NVT", "ENLT",
    ],
    "TECNOLOGIA": [
        "GOOGL", "AAPL", "PANW", "CRWD", "CRM", "NOW", "PLTR", "DDOG", "SNOW",
        "NBIS", "SONY", "ADEA", "PL", "NOK", "ITRN", "CSCO",
    ],
    "FINANCIERO": [
        "GS", "UBS", "IBKR", "V", "BX", "BRKB", "NU", "MUFG", "SMFG", "MFG",
        "JMP", "ENVA", "EZPW", "FCFS", "BLX", "SNEX", "AAMI",
    ],
    "SALUD": [
        "LLY", "ABBV", "AMGN", "MRK", "CAH", "MCK", "CPRX", "ANAB", "MIRM",
        "KNSA", "AXSM", "BTSG", "WELL", "BWAY", "CDNL",
    ],
    "DEFENSA_AERO_INDUSTRIAL": [
        "HWM", "CW", "ESLT", "PL", "CAT", "ROK", "FTAI", "AER", "CRS", "ATI",
        "WWD", "R", "APG", "NVT", "STLD", "POWL", "PWR", "STRL", "FIX", "AGX",
        "CECO", "GTX", "IESC", "MOD", "RBC", "AIT", "ESE", "GEV",
    ],
    "ENERGIA_MATERIAS": [
        "CCJ", "FTI", "SEI", "NGS", "CEG", "ENLT", "ALM", "MP", "STLD", "ATI",
        "CRS", "ODC",
    ],
    "CONSUMO": [
        "WMT", "CASY", "HLT", "COKE", "TM", "GIL", "SONY", "TKO", "LINC", "ODC",
    ],
}

# Índice inverso rápido: ticker -> lista de sectores a los que pertenece
SECTORES_POR_TICKER: Dict[str, List[str]] = {}
for _sec, _tks in MAPA_SECTORES_TICKERS.items():
    for _tk in _tks:
        SECTORES_POR_TICKER.setdefault(_tk, []).append(_sec)


def obtener_sectores_ticker(ticker: str) -> List[str]:
    """Devuelve los sectores/cadenas de suministro vinculados a un ticker."""
    return list(SECTORES_POR_TICKER.get(ticker.strip().upper(), ["GENERAL"]))


def obtener_tickers_relacionados_por_sectores(
    sectores: List[str],
    universe: Optional[List[str]] = None,
) -> List[str]:
    """Devuelve acciones de la lista activa que pertenecen a los sectores especificados."""
    universe_set = {t.upper() for t in universe} if universe else None
    vistos = set()
    resultado: List[str] = []
    for sec in sectores:
        for tk in MAPA_SECTORES_TICKERS.get(sec, []):
            if tk in vistos:
                continue
            if universe_set is None or tk in universe_set:
                vistos.add(tk)
                resultado.append(tk)
    return resultado


# ==============================================================================
# 5. CATÁLOGO DE MODELOS DE IA SOPORTADOS Y NIVELES DE ESFUERZO
# ==============================================================================

MODELOS_IA_DISPONIBLES: Dict[str, Dict[str, Any]] = {
    "flash3.6": {
        "nombre_corto": "flash3.6",
        "nombre_mostrar": "Flash 3.6",
        "proveedor": "gemini",
        "model_id": "gemini-3.6-flash",
        "fallback_ids": ["gemini-3.8-flash"],
        "niveles_esfuerzo": ["low", "medium", "high"],
        "esfuerzo_default": "medium",
        "descripcion": "Gemini 3.6 Flash — Veloz y eficiente (effort: low, medium, high)",
    },
    "flash3.7": {
        "nombre_corto": "flash3.7",
        "nombre_mostrar": "Flash 3.7",
        "proveedor": "gemini",
        "model_id": "gemini-3.7-flash",
        "fallback_ids": ["gemini-3.8-flash"],
        "niveles_esfuerzo": ["low", "medium", "high"],
        "esfuerzo_default": "medium",
        "descripcion": "Gemini 3.7 Flash — Precisión agéntica mejorada (effort: low, medium, high)",
    },
    "flash3.8": {
        "nombre_corto": "flash3.8",
        "nombre_mostrar": "Flash 3.8",
        "proveedor": "gemini",
        "model_id": "gemini-3.8-flash",
        "fallback_ids": [],
        "niveles_esfuerzo": ["low", "medium", "high"],
        "esfuerzo_default": "medium",
        "descripcion": "Gemini 3.8 Flash — Alta velocidad y balance (effort: low, medium, high)",
    },
    "argon": {
        "nombre_corto": "argon",
        "nombre_mostrar": "Argon",
        "proveedor": "gemini",
        "model_id": "gemini-4-argon",
        "fallback_ids": ["gemini-3.8-flash"],
        "niveles_esfuerzo": ["low", "medium", "high"],
        "esfuerzo_default": "medium",
        "descripcion": "Gemini 4 Argon — Razonamiento frontera eficiente (effort: low, medium, high)",
    },
    "hydrogen": {
        "nombre_corto": "hydrogen",
        "nombre_mostrar": "Hydrogen",
        "proveedor": "gemini",
        "model_id": "gemini-4-hydrogen",
        "fallback_ids": ["gemini-4-argon"],
        "niveles_esfuerzo": ["low", "medium", "high", "xhigh"],
        "esfuerzo_default": "high",
        "descripcion": "Gemini 4 Hydrogen — Rápido con modo profundo (effort: low, medium, high, xhigh)",
    },
    "carbon": {
        "nombre_corto": "carbon",
        "nombre_mostrar": "Carbon",
        "proveedor": "gemini",
        "model_id": "gemini-4-carbon",
        "fallback_ids": ["gemini-4-argon"],
        "niveles_esfuerzo": ["low", "medium", "high"],
        "esfuerzo_default": "medium",
        "descripcion": "Gemini 4 Carbon — Máxima profundidad analítica (effort: low, medium, high)",
    },
    "pro3.1": {
        "nombre_corto": "pro3.1",
        "nombre_mostrar": "Pro 3.1",
        "proveedor": "gemini",
        "model_id": "gemini-3.1-pro-preview",
        "fallback_ids": [],
        "niveles_esfuerzo": ["low", "high"],
        "esfuerzo_default": "high",
        "descripcion": "Gemini 3.1 Pro — Razonamiento complejo (effort: low, high)",
    },
    "sonnet": {
        "nombre_corto": "sonnet",
        "nombre_mostrar": "Sonnet",
        "proveedor": "anthropic",
        "model_id": "claude-sonnet-5-5",
        "fallback_ids": ["claude-sonnet-4-6", "claude-3-7-sonnet-latest"],
        "niveles_esfuerzo": ["low", "medium", "high", "xhigh", "max"],
        "esfuerzo_default": "high",
        "descripcion": "Claude Sonnet 5.5 — Balance financiero rápido (effort: low, medium, high, xhigh, max)",
    },
    "opus": {
        "nombre_corto": "opus",
        "nombre_mostrar": "Opus",
        "proveedor": "anthropic",
        "model_id": "claude-opus-5-5",
        "fallback_ids": ["claude-opus-4-6", "claude-3-opus-latest"],
        "niveles_esfuerzo": ["low", "medium", "high", "xhigh", "max"],
        "esfuerzo_default": "medium",
        "descripcion": "Claude Opus 5.5 — Análisis profundo multi-cartera (effort: low, medium, high, xhigh, max)",
    },
    "fable": {
        "nombre_corto": "fable",
        "nombre_mostrar": "Fable",
        "proveedor": "anthropic",
        "model_id": "claude-fable-5-2",
        "fallback_ids": ["claude-opus-5-5", "claude-opus-4-6"],
        "niveles_esfuerzo": ["low", "medium", "high", "xhigh", "max"],
        "esfuerzo_default": "high",
        "descripcion": "Claude Fable 5.2 — Modelo frontera insignia (effort: low, medium, high, xhigh, max)",
    },
}

ALIAS_MODELOS_IA: Dict[str, str] = {
    "flash": "flash3.8",
    "flash36": "flash3.6",
    "flash 3.6": "flash3.6",
    "3.6": "flash3.6",
    "flash37": "flash3.7",
    "flash 3.7": "flash3.7",
    "3.7": "flash3.7",
    "flash38": "flash3.8",
    "flash 3.8": "flash3.8",
    "3.8": "flash3.8",
    "pro": "pro3.1",
    "pro 3.1": "pro3.1",
    "3.1": "pro3.1",
}


def resolver_clave_modelo(entrada: str) -> Optional[str]:
    """Resuelve alias o clave de modelo a una clave canónica de `MODELOS_IA_DISPONIBLES`."""
    norm = re.sub(r"\s+", " ", (entrada or "").strip().lower())
    if not norm:
        return "flash3.8"
    if norm in MODELOS_IA_DISPONIBLES:
        return norm
    if norm in ALIAS_MODELOS_IA:
        return ALIAS_MODELOS_IA[norm]
    claves = list(MODELOS_IA_DISPONIBLES.keys())
    if norm.isdigit():
        idx = int(norm) - 1
        if 0 <= idx < len(claves):
            return claves[idx]
    return None


def normalizar_esfuerzo_modelo(modelo_clave: str, esfuerzo: Optional[str]) -> str:
    """Devuelve un nivel de esfuerzo válido para el modelo dado (o su valor por defecto)."""
    clave = resolver_clave_modelo(modelo_clave) or "flash3.8"
    meta = MODELOS_IA_DISPONIBLES[clave]
    permitidos: List[str] = list(meta["niveles_esfuerzo"])
    if not esfuerzo:
        return str(meta["esfuerzo_default"])
    e = esfuerzo.strip().lower()
    if e in permitidos:
        return e
    # Aceptar `medium` explícito si el usuario lo pasa en Pro 3.1 aunque el menú muestre low/high
    if clave == "pro3.1" and e == "medium":
        return "medium"
    return str(meta["esfuerzo_default"])


# ==============================================================================
# 6. RÉGIMEN HORARIO DE MERCADO (WALL STREET ET) PARA CONTROL DE CONSULTAS
# ==============================================================================

def _ahora_nueva_york(ahora_et: Optional[datetime] = None) -> datetime:
    if ahora_et is not None:
        return ahora_et
    try:
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # Fallback aproximado EST (UTC-5) si no hubiera base tzdata
        return datetime.now(timezone(timedelta(hours=-5)))


def obtener_regimen_horario_mercado(ahora_et: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Evalúa la hora actual de Nueva York (NYSE/NASDAQ) para ajustar la sensibilidad
    del Smart Gatekeeper:
    - Más PERMISIVO en `PRE_MARKET_APERTURA` (08:30-09:30 ET), `APERTURA_MERCADO` (09:30-10:45 ET)
      y `CIERRE_MERCADO` (14:45-16:00 ET).
    - ESTRICTO en `SESION_MEDIA` (10:45-14:45 ET) y `EXTENDIDO` (04:00-08:30 / 16:00-20:00 ET).
    - MUY ESTRICTO en `MERCADO_CERRADO` (noches / fines de semana) para evitar neutrales inútiles.
    """
    dt_ny = _ahora_nueva_york(ahora_et)
    dia_sem = dt_ny.weekday()  # 0=Lunes .. 6=Domingo
    hora_local = dt_ny.time()

    if dia_sem >= 5:
        return {
            "regimen": "MERCADO_CERRADO",
            "etiqueta": "CERRADO (FIN SEMANA)",
            "es_permisivo": False,
            "umbral_confluencia": 82.0,
            "ventana_lote_seg": 5.5,
            "cooldown_neutral_seg": 600.0,
            "enfriamiento_base_seg": 300.0,
            "hora_et": dt_ny.strftime("%I:%M %p ET"),
        }

    if dt_time(8, 30) <= hora_local < dt_time(9, 30):
        return {
            "regimen": "PRE_MARKET_APERTURA",
            "etiqueta": "PRE-MARKET APERTURA (PERMISIVO)",
            "es_permisivo": True,
            "umbral_confluencia": 52.0,
            "ventana_lote_seg": 2.5,
            "cooldown_neutral_seg": 240.0,
            "enfriamiento_base_seg": 110.0,
            "hora_et": dt_ny.strftime("%I:%M %p ET"),
        }

    if dt_time(9, 30) <= hora_local < dt_time(10, 45):
        return {
            "regimen": "APERTURA_MERCADO",
            "etiqueta": "APERTURA SESIÓN (PERMISIVO)",
            "es_permisivo": True,
            "umbral_confluencia": 54.0,
            "ventana_lote_seg": 3.0,
            "cooldown_neutral_seg": 270.0,
            "enfriamiento_base_seg": 120.0,
            "hora_et": dt_ny.strftime("%I:%M %p ET"),
        }

    if dt_time(10, 45) <= hora_local < dt_time(14, 45):
        return {
            "regimen": "SESION_MEDIA",
            "etiqueta": "MEDIA SESIÓN (ESTRICTO)",
            "es_permisivo": False,
            "umbral_confluencia": 68.0,
            "ventana_lote_seg": 4.5,
            "cooldown_neutral_seg": 480.0,
            "enfriamiento_base_seg": 210.0,
            "hora_et": dt_ny.strftime("%I:%M %p ET"),
        }

    if dt_time(14, 45) <= hora_local < dt_time(16, 0):
        return {
            "regimen": "CIERRE_MERCADO",
            "etiqueta": "PRE-CIERRE SESIÓN (PERMISIVO)",
            "es_permisivo": True,
            "umbral_confluencia": 54.0,
            "ventana_lote_seg": 2.8,
            "cooldown_neutral_seg": 260.0,
            "enfriamiento_base_seg": 120.0,
            "hora_et": dt_ny.strftime("%I:%M %p ET"),
        }

    if (dt_time(4, 0) <= hora_local < dt_time(8, 30)) or (dt_time(16, 0) <= hora_local < dt_time(20, 0)):
        return {
            "regimen": "HORARIO_EXTENDIDO",
            "etiqueta": "EXTENDIDO (SELECTIVO)",
            "es_permisivo": False,
            "umbral_confluencia": 74.0,
            "ventana_lote_seg": 5.0,
            "cooldown_neutral_seg": 540.0,
            "enfriamiento_base_seg": 240.0,
            "hora_et": dt_ny.strftime("%I:%M %p ET"),
        }

    return {
        "regimen": "MERCADO_CERRADO",
        "etiqueta": "NOCTURNO/CERRADO (MUY ESTRICTO)",
        "es_permisivo": False,
        "umbral_confluencia": 80.0,
        "ventana_lote_seg": 5.5,
        "cooldown_neutral_seg": 600.0,
        "enfriamiento_base_seg": 300.0,
        "hora_et": dt_ny.strftime("%I:%M %p ET"),
    }


# ==============================================================================
# 7. PARÁMETROS OPERATIVOS, UMBRALES Y CONSULTAS MÚLTIPLES EN LOTE
# ==============================================================================

NOMBRE_DIR_HYSTMP = "hystmp"
NOMBRE_DIR_SESION_ACTIVA = ".sesion_activa"

# Intervalos de actualización (segundos)
INTERVALO_CICLO_NOTICIAS_SEG = 25          # Sondeo continuo de fuentes RSS y Yahoo News
INTERVALO_CICLO_INDICADORES_SEG = 35       # Descarga vectorizada por lotes de precios/indicadores
UMBRAL_LATENCIA_NOTICIAS_ADVERTENCIA = 180 # 3 min sin noticias nuevas -> Estado VIGILANCIA
UMBRAL_LATENCIA_NOTICIAS_ANORMAL = 420     # 7 min sin noticias nuevas -> Estado ANORMAL

# Umbrales de filtrado técnico e IA
UMBRAL_POTENCIAL_DATOS_TERMINAL = 62.0     # Score mínimo (0-100) para mostrar en terminal /datos
UMBRAL_CERTEZA_EVENTO_TECNICO = 60.0       # Certeza mínima técnica (%) para considerar evento fuerte
UMBRAL_CERTEZA_AI_OPERACION = 65.0         # Certeza mínima IA (%) para ejecutar compra/venta
UMBRAL_CERTEZA_AI_ROTACION_PERDIDA = 84.0  # Certeza mínima IA (%) para liquidar posición en pérdida por rotación

# Mitigación de consultas innecesarias y método de consulta múltiple
COOLDOWN_CONSULTA_TICKER_SEG = 150         # Cooldown base estándar entre análisis del mismo ticker
COOLDOWN_NEUTRAL_EXTENDIDO_SEG = 450       # Cooldown ampliado si el modelo dictaminó NEUTRAL
MAX_CONSULTAS_AI_POR_MINUTO = 8            # Tope global de llamadas API por minuto (en lotes multi-ticker)
MAX_ASIGNACION_POR_POSICION_PCT = 25.0     # Máximo % de la cartera en una sola posición estándar

MIN_TICKERS_LOTE_AI = 2                    # Agrupa al menos 2 empresas cuando existan señales próximas
MAX_TICKERS_LOTE_AI = 6                    # Máximo de empresas evaluadas simultáneamente en una consulta IA
VENTANA_AGRUPACION_LOTE_SEG = 3.5          # Ventana para acumular candidatos interrelacionados antes de enviar lote
