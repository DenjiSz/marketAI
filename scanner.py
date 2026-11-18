# -*- coding: utf-8 -*-
"""
scanner.py
==========
Escáner cuantitativo multi-empresa (evolución de `indicadores.py`).
Descarga datos financieros/fundamentales, flujo de volumen real, reporte de resultados
y calcula vectorizadamente 17 indicadores técnicos, institucionales y de acción del precio:

- Datos financieros de la empresa: Precio actual, cambio %, P/E (Trailing y Forward),
  Soporte 1/2, Resistencia 1/2, Capitalización de Mercado, Rango 52 semanas, ATR, Volumen
  y **Reporte de Resultados** (fecha próximo reporte, ingresos esperados vs reportados, EPS y sorpresa).
- 17 Indicadores cuantitativos y patrones swing/estructurales:
  1.  Oscilador (contrarian) | RSI(14) | cortes 20/35/65/80
  2.  Oscilador (contrarian) | Stoch(14,3,3) | cortes 10/20/80/90
  3.  Oscilador (contrarian) | Bollinger %B(20,2) | cortes 0/0.2/0.8/1
  4.  Oscilador (contrarian) | MFI(14), con volumen | cortes 15/30/70/85
  5.  Tendencia | EMA 9/21 | spread dividido por ATR
  6.  Tendencia | MACD histograma | dividido por ATR
  7.  Tendencia | VWAP de sesión | z-score de la distancia %
  8.  Tendencia | ADX/DI | DI+ - DI-, vale 0 si ADX < 20
  9.  Supertrend(10,2.5)/ATR
  10. CMF(20) Flujo Inst.
  11. Volumen Real | RVOL(20) Impulso Direccional
  12. Volumen Real | OBV Z-Score(20) Acumulación
  13. Estructural | Rango S/R & Ruptura (BOS)
  14. Acción de Precio | Ruptura Rango Apertura (ORB)
  15. Patrón Chartista | Bandera (Bull/Bear Flag)
  16. Reversión Swing | Rebote por Sobreventa / Rechazo Sobrecompra
  17. Marco Diario | Tendencia SMA 20/50 & Compresión Squeeze

Filtra eventos con nivel de certeza/potencial y solo emite a la terminal secundaria `/d`
y al acumulador multi-pilar de IA aquellos activos que alcanzan alto potencial operativo.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dt_time
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    INTERVALO_CICLO_INDICADORES_SEG,
    UMBRAL_CERTEZA_EVENTO_TECNICO,
    UMBRAL_POTENCIAL_DATOS_TERMINAL,
    formatear_fecha_hora_ampm,
    simbolo_yfinance,
)
from historial import GestorHistorial

# Silenciar trazas internas de yfinance/pandas para no ensuciar nunca la interfaz TUI
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. ESTRUCTURAS DE DATOS DE INDICADORES, RESULTADOS Y SNAPSHOT FINANCIERO
# ==============================================================================

@dataclass
class ResultadoIndicador:
    """Resultado individual de un indicador técnico/estructural con su evaluación y alarma."""

    categoria: str          # Ej. "Oscilador (contrarian)", "Tendencia", "Flujo Inst.", "Patrón Chartista", etc.
    nombre: str             # Ej. "RSI(14)"
    valor: float            # Valor numérico principal calculado
    texto_valor: str        # Representación legible con unidades o cortes
    evaluacion: str         # "COMPRA FUERTE" | "COMPRA" | "NEUTRAL" | "VENTA" | "VENTA FUERTE"
    puntaje: float          # Escala normalizada [-2.0, +2.0] para confluencia
    alarma: Optional[str]   # "COMPRA" | "VENTA" | None (resalta nombre en verde o rojo)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatosReporteResultados:
    """Datos de earnings/resultados trimestrales extraídos para análisis conjunto con `analisis.py`."""

    ticker: str
    fecha_proximo_reporte: Optional[str] = None      # YYYY-MM-DD
    dias_hasta_reporte: Optional[int] = None         # Días calendario hasta próximo reporte
    fecha_ultimo_reporte: Optional[str] = None       # YYYY-MM-DD
    ingresos_esperados: Optional[float] = None       # USD estimados trimestre actual/próximo
    ingresos_reportados: Optional[float] = None      # USD reales último trimestre reportado
    ingresos_crecimiento_yoy_pct: Optional[float] = None
    ingresos_sorpresa_pct: Optional[float] = None
    eps_esperado: Optional[float] = None
    eps_reportado: Optional[float] = None
    eps_sorpresa_pct: Optional[float] = None
    evaluacion_reporte: str = "PENDIENTE"            # "POSITIVO" | "MIXTO" | "NEGATIVO" | "PROXIMO_INMINENTE" | "PENDIENTE"
    resumen_reporte: str = ""
    actualizado_epoch: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatosFinancieros:
    """Datos fundamentales, niveles estructurales clave y reporte de resultados de una acción."""

    ticker: str
    precio_actual: float
    cambio_diario_pct: float
    pe_ratio: Optional[float]
    pe_forward: Optional[float]
    soporte_1: float
    soporte_2: float
    resistencia_1: float
    resistencia_2: float
    vwap_sesion: float
    atr_14: float
    volumen_barra: float
    volumen_promedio_20: float
    cap_mercado_b: Optional[float]   # En miles de millones (Billions USD)
    max_52_semanas: Optional[float]
    min_52_semanas: Optional[float]
    beta: Optional[float]
    reporte_resultados: Optional[DatosReporteResultados] = None

    def to_dict(self) -> Dict[str, Any]:
        datos = asdict(self)
        if self.reporte_resultados is not None:
            datos["reporte_resultados"] = self.reporte_resultados.to_dict()
        return datos


@dataclass
class AnalisisTecnicoTicker:
    """Paquete completo financiero + 17 indicadores + patrones + evaluación de certeza/potencial."""

    ticker: str
    epoch: float
    fecha_hora_ampm: str
    datos_financieros: DatosFinancieros
    indicadores: List[ResultadoIndicador]
    evaluacion_general: str          # "COMPRA FUERTE" | "COMPRA" | "NEUTRAL" | "VENTA" | "VENTA FUERTE"
    direccion_potencial: str         # "ALCISTA" | "BAJISTA" | "NEUTRAL"
    potencial_score: float           # 0.0 - 100.0
    certeza_pct: float               # 0.0 - 100.0
    alarmas_compra: List[str] = field(default_factory=list)
    alarmas_venta: List[str] = field(default_factory=list)
    patrones_detectados: List[str] = field(default_factory=list)
    es_evento_alto_potencial: bool = False
    motivo_evento: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "epoch": self.epoch,
            "fecha_hora_ampm": self.fecha_hora_ampm,
            "datos_financieros": self.datos_financieros.to_dict(),
            "indicadores": [ind.to_dict() for ind in self.indicadores],
            "evaluacion_general": self.evaluacion_general,
            "direccion_potencial": self.direccion_potencial,
            "potencial_score": self.potencial_score,
            "certeza_pct": self.certeza_pct,
            "alarmas_compra": self.alarmas_compra,
            "alarmas_venta": self.alarmas_venta,
            "patrones_detectados": self.patrones_detectados,
            "es_evento_alto_potencial": self.es_evento_alto_potencial,
            "motivo_evento": self.motivo_evento,
        }


# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS Y DETECCIÓN DE PATRONES (NUMPY / PANDAS)
# ==============================================================================

def _safe_float(val: object, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _safe_opt_float(val: object) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


def formatear_monto_compacto(valor: Optional[float]) -> str:
    """Formatea ingresos en $B / $M de forma legible."""
    if valor is None:
        return "N/D"
    abs_v = abs(valor)
    if abs_v >= 1e9:
        return f"${valor / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${valor / 1e6:.1f}M"
    return f"${valor:,.0f}"


def calcular_atr(high: pd.Series, low: pd.Series, close: pd.Series, periodo: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / periodo, min_periods=periodo, adjust=False).mean()


def calcular_rsi(close: pd.Series, periodo: int = 14) -> float:
    """RSI(14) clásico con suavizado exponencial de Wilder."""
    delta = close.diff()
    ganancia = delta.clip(lower=0.0)
    perdida = (-delta).clip(lower=0.0)
    avg_gain = ganancia.ewm(alpha=1.0 / periodo, min_periods=periodo, adjust=False).mean()
    avg_loss = perdida.ewm(alpha=1.0 / periodo, min_periods=periodo, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    return _safe_float(rsi.iloc[-1], 50.0)


def calcular_rsi_serie(close: pd.Series, periodo: int = 14) -> pd.Series:
    """Serie histórica de RSI(14) para detección de rebotes y divergencias."""
    delta = close.diff()
    ganancia = delta.clip(lower=0.0)
    perdida = (-delta).clip(lower=0.0)
    avg_gain = ganancia.ewm(alpha=1.0 / periodo, min_periods=periodo, adjust=False).mean()
    avg_loss = perdida.ewm(alpha=1.0 / periodo, min_periods=periodo, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def calcular_stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_len: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> Tuple[float, float]:
    """Stochastic Oscillator (14, 3, 3) -> devuelve (%K suavizado, %D)."""
    lowest_low = low.rolling(window=k_len, min_periods=max(5, k_len // 2)).min()
    highest_high = high.rolling(window=k_len, min_periods=max(5, k_len // 2)).max()
    rango = (highest_high - lowest_low).replace(0.0, np.nan)
    raw_k = 100.0 * (close - lowest_low) / rango
    k = raw_k.rolling(window=k_smooth, min_periods=1).mean().fillna(50.0)
    d = k.rolling(window=d_smooth, min_periods=1).mean().fillna(50.0)
    return _safe_float(k.iloc[-1], 50.0), _safe_float(d.iloc[-1], 50.0)


def calcular_bollinger_pct_b(close: pd.Series, periodo: int = 20, mult: float = 2.0) -> float:
    """Bollinger %B(20, 2) = (Close - BandaInferior) / (BandaSuperior - BandaInferior)."""
    pct_b = calcular_bollinger_pct_b_serie(close, periodo=periodo, mult=mult)
    return _safe_float(pct_b.iloc[-1], 0.5)


def calcular_bollinger_pct_b_serie(close: pd.Series, periodo: int = 20, mult: float = 2.0) -> pd.Series:
    sma = close.rolling(window=periodo, min_periods=max(5, periodo // 2)).mean()
    std = close.rolling(window=periodo, min_periods=max(5, periodo // 2)).std(ddof=0)
    upper = sma + mult * std
    lower = sma - mult * std
    ancho = (upper - lower).replace(0.0, np.nan)
    return ((close - lower) / ancho).fillna(0.5)


def calcular_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, periodo: int = 14) -> float:
    """Money Flow Index MFI(14) ponderado por volumen."""
    tp = (high + low + close) / 3.0
    rmf = tp * volume.clip(lower=0.0)
    diff_tp = tp.diff()
    pos_mf = rmf.where(diff_tp > 0.0, 0.0).rolling(window=periodo, min_periods=max(5, periodo // 2)).sum()
    neg_mf = rmf.where(diff_tp < 0.0, 0.0).rolling(window=periodo, min_periods=max(5, periodo // 2)).sum()
    mfr = pos_mf / neg_mf.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    mfi = mfi.fillna(50.0)
    return _safe_float(mfi.iloc[-1], 50.0)


def calcular_adx_di(high: pd.Series, low: pd.Series, close: pd.Series, periodo: int = 14) -> Tuple[float, float, float, float]:
    """
    Calcula ADX(14), DI+(14), DI-(14) y el valor reglado:
    `DI+ - DI-`, que vale exactamente `0.0` si `ADX < 20`.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=close.index,
    )

    atr = calcular_atr(high, low, close, periodo=periodo).replace(0.0, np.nan)
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / periodo, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / periodo, adjust=False).mean() / atr)

    suma_di = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / suma_di
    adx = dx.ewm(alpha=1.0 / periodo, adjust=False).mean().fillna(0.0)

    adx_val = _safe_float(adx.iloc[-1], 0.0)
    di_pos_val = _safe_float(plus_di.iloc[-1], 0.0)
    di_neg_val = _safe_float(minus_di.iloc[-1], 0.0)

    spread_reglado = 0.0 if adx_val < 20.0 else (di_pos_val - di_neg_val)
    return adx_val, di_pos_val, di_neg_val, spread_reglado


def calcular_supertrend_norm(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_ref: float,
    periodo: int = 10,
    multiplicador: float = 2.5,
) -> Tuple[float, int, bool]:
    """
    Calcula Supertrend(10, 2.5) y devuelve:
    `(distancia_dividida_por_ATR, direccion (+1 alcista / -1 bajista), hubo_cambio_reciente)`.
    """
    atr_st = calcular_atr(high, low, close, periodo=periodo).bfill().fillna(max(atr_ref, 0.01))
    hl2 = (high + low) / 2.0
    basic_upper = (hl2 + multiplicador * atr_st).to_numpy(dtype=float)
    basic_lower = (hl2 - multiplicador * atr_st).to_numpy(dtype=float)
    c_arr = close.to_numpy(dtype=float)

    n = len(c_arr)
    if n < 3:
        return 0.0, 1, False

    final_upper = np.copy(basic_upper)
    final_lower = np.copy(basic_lower)
    direccion = np.ones(n, dtype=int)
    st_line = np.copy(final_lower)

    for i in range(1, n):
        if basic_upper[i] < final_upper[i - 1] or c_arr[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower[i] > final_lower[i - 1] or c_arr[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if direccion[i - 1] == -1 and c_arr[i] > final_upper[i]:
            direccion[i] = 1
        elif direccion[i - 1] == 1 and c_arr[i] < final_lower[i]:
            direccion[i] = -1
        else:
            direccion[i] = direccion[i - 1]

        st_line[i] = final_lower[i] if direccion[i] == 1 else final_upper[i]

    denom = max(atr_ref, 1e-6)
    st_div_atr = (c_arr[-1] - st_line[-1]) / denom
    cambio_reciente = bool(direccion[-1] != direccion[-2])
    return _safe_float(st_div_atr, 0.0), int(direccion[-1]), cambio_reciente


def calcular_vwap_sesion_zscore(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calcula el VWAP de la sesión intradía más reciente y el Z-Score de la distancia porcentual
    `((Close - VWAP) / VWAP) * 100`.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].clip(lower=0.0)

    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        fechas = df.index.date
        mask_hoy = fechas == fechas[-1]
        if int(mask_hoy.sum()) >= 3:
            tp_ses = tp[mask_hoy]
            vol_ses = vol[mask_hoy]
            close_ses = df["Close"][mask_hoy]
        else:
            tp_ses, vol_ses, close_ses = tp.tail(26), vol.tail(26), df["Close"].tail(26)
    else:
        tp_ses, vol_ses, close_ses = tp.tail(26), vol.tail(26), df["Close"].tail(26)

    cum_vol = vol_ses.cumsum().replace(0.0, np.nan)
    vwap_serie = ((tp_ses * vol_ses).cumsum() / cum_vol).bfill()
    vwap_actual = _safe_float(vwap_serie.iloc[-1], _safe_float(df["Close"].iloc[-1]))

    dist_pct = ((close_ses - vwap_serie) / vwap_serie.replace(0.0, np.nan)) * 100.0
    dist_pct = dist_pct.fillna(0.0)
    if len(dist_pct) >= 4:
        media = _safe_float(dist_pct.mean(), 0.0)
        desv = _safe_float(dist_pct.std(ddof=0), 0.0)
        zscore = (_safe_float(dist_pct.iloc[-1]) - media) / desv if desv > 1e-5 else _safe_float(dist_pct.iloc[-1])
    else:
        zscore = _safe_float(dist_pct.iloc[-1], 0.0)

    return vwap_actual, _safe_float(zscore, 0.0)


def calcular_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, periodo: int = 20) -> float:
    """Chaikin Money Flow CMF(20) — mide acumulación/distribución institucional."""
    rango = (high - low).replace(0.0, np.nan)
    mf_mult = (((close - low) - (high - close)) / rango).fillna(0.0)
    mf_vol = mf_mult * volume.clip(lower=0.0)
    vol_sum = volume.clip(lower=0.0).rolling(window=periodo, min_periods=max(5, periodo // 2)).sum().replace(0.0, np.nan)
    cmf = (mf_vol.rolling(window=periodo, min_periods=max(5, periodo // 2)).sum() / vol_sum).fillna(0.0)
    return _safe_float(cmf.iloc[-1], 0.0)


def calcular_ruptura_orb(
    df_intra: pd.DataFrame,
    precio_actual: float,
    atr_14: float,
    rvol: float,
) -> Tuple[float, str, str, float, Optional[str]]:
    """
    Detecta Ruptura de Rango de Apertura (Opening Range Breakout - ORB):
    Usa los primeros 30 minutos de la sesión regular (09:30-10:00 ET en velas 15m -> 2 barras)
    o las 2 primeras velas de la fecha más reciente.
    Devuelve: `(valor_atr_spread, evaluacion, texto, puntaje, alarma)`.
    """
    if df_intra is None or len(df_intra) < 4:
        return 0.0, "NEUTRAL", "ORB N/D (pocas velas)", 0.0, None

    try:
        sub = df_intra.copy()
        if isinstance(sub.index, pd.DatetimeIndex):
            idx = sub.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            idx_ny = idx.tz_convert(ZoneInfo("America/New_York"))
            ultima_fecha = idx_ny[-1].date()
            mask_dia = idx_ny.date == ultima_fecha
            sub_dia = sub[mask_dia]
            idx_dia = idx_ny[mask_dia]

            # Barras regulares desde 09:30 ET
            horas = idx_dia.time
            mask_reg = (horas >= dt_time(9, 30)) & (horas < dt_time(16, 0))
            sub_reg = sub_dia[mask_reg]
            idx_reg = idx_dia[mask_reg]

            if len(sub_reg) >= 2:
                # Primeros 30 min (09:30 a 09:59 ET)
                mask_orb = idx_reg.time < dt_time(10, 0)
                sub_orb = sub_reg[mask_orb] if int(mask_orb.sum()) >= 1 else sub_reg.iloc[:2]
                fuente = sub_reg
            else:
                sub_orb = sub_dia.iloc[:2]
                fuente = sub_dia
        else:
            sub_orb = sub.iloc[-6:-4]
            fuente = sub
    except Exception:
        sub_orb = df_intra.iloc[-min(len(df_intra), 8):-min(len(df_intra), 6)]
        fuente = df_intra

    if len(sub_orb) < 1 or len(fuente) < 2:
        return 0.0, "NEUTRAL", "ORB en formación", 0.0, None

    orb_high = _safe_float(sub_orb["High"].max(), precio_actual)
    orb_low = _safe_float(sub_orb["Low"].min(), precio_actual)
    spread_hi = (precio_actual - orb_high) / max(atr_14, 1e-4)
    spread_lo = (orb_low - precio_actual) / max(atr_14, 1e-4)

    # Si aún estamos dentro del rango inicial (primeras 2 barras de apertura)
    if len(fuente) <= 2:
        return 0.0, "NEUTRAL", f"ORB formando [${orb_low:.2f}-${orb_high:.2f}]", 0.0, None

    if spread_hi >= 0.25 and rvol >= 1.25:
        return round(spread_hi, 3), "COMPRA FUERTE", f"ORB Alcista +{spread_hi:.2f} ATR > ${orb_high:.2f} (RVOL {rvol:.2f}x)", 2.0, "COMPRA"
    if spread_hi > 0.05:
        return round(spread_hi, 3), "COMPRA", f"Sobre ORB High ${orb_high:.2f} (+{spread_hi:.2f} ATR)", 1.0, None
    if spread_lo >= 0.25 and rvol >= 1.25:
        return round(-spread_lo, 3), "VENTA FUERTE", f"ORB Bajista -{spread_lo:.2f} ATR < ${orb_low:.2f} (RVOL {rvol:.2f}x)", -2.0, "VENTA"
    if spread_lo > 0.05:
        return round(-spread_lo, 3), "VENTA", f"Bajo ORB Low ${orb_low:.2f} (-{spread_lo:.2f} ATR)", -1.0, None
    return 0.0, "NEUTRAL", f"Dentro ORB [${orb_low:.2f}-${orb_high:.2f}]", 0.0, None


def detectar_patron_bandera(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    atr_14: float,
    rvol: float,
) -> Tuple[float, str, str, float, Optional[str]]:
    """
    Detecta Patrón Bandera (Bull Flag / Bear Flag):
    1. Mástil previo fuerte en barras [-18:-4] con desplazamiento >= 2.0 * ATR.
    2. Consolidación estrecha en últimas 3-5 barras (retroceso <= 45% del mástil).
    3. Ruptura o proximidad al borde de continuación con flujo acorde.
    """
    if len(close) < 20:
        return 0.0, "NEUTRAL", "Bandera N/D", 0.0, None

    c_arr = close.to_numpy(dtype=float)
    h_arr = high.to_numpy(dtype=float)
    l_arr = low.to_numpy(dtype=float)
    v_arr = volume.to_numpy(dtype=float)
    denom = max(atr_14, 1e-4)

    # Analizar ventana de mástil + bandera (16 velas recientes)
    c_mast = c_arr[-18:-4]
    idx_min = int(np.argmin(c_mast))
    idx_max = int(np.argmax(c_mast))
    rango_mast_atr = (float(np.max(c_mast)) - float(np.min(c_mast))) / denom

    c_flag = c_arr[-5:-1]
    h_flag = h_arr[-5:-1]
    l_flag = l_arr[-5:-1]
    techo_flag = float(np.max(h_flag))
    piso_flag = float(np.min(l_flag))
    ancho_flag = max(techo_flag - piso_flag, 1e-6)
    vol_mast = float(np.mean(v_arr[-14:-4])) if np.mean(v_arr[-14:-4]) > 0 else 1.0
    vol_flag = float(np.mean(v_arr[-5:-1]))
    compresion_vol = vol_flag <= (vol_mast * 1.05)

    precio = float(c_arr[-1])

    # Bull Flag: mínimo ocurre antes del máximo dentro del mástil
    if rango_mast_atr >= 2.0 and idx_max > idx_min:
        altura_mast = float(np.max(c_mast)) - float(np.min(c_mast))
        retroceso = (float(np.max(c_mast)) - piso_flag) / max(altura_mast, 1e-6)
        es_compacta = ancho_flag <= (1.6 * denom) and retroceso <= 0.48
        if es_compacta:
            if precio > techo_flag and (rvol >= 1.15 or compresion_vol):
                return 2.0, "COMPRA FUERTE", f"Ruptura Bull Flag > ${techo_flag:.2f} (mástil {rango_mast_atr:.1f} ATR)", 2.0, "COMPRA"
            if precio >= piso_flag and compresion_vol:
                return 1.0, "COMPRA", f"Bull Flag en consolidación [${piso_flag:.2f}-${techo_flag:.2f}]", 1.1, None

    # Bear Flag: máximo ocurre antes del mínimo dentro del mástil
    if rango_mast_atr >= 2.0 and idx_min > idx_max:
        altura_mast = float(np.max(c_mast)) - float(np.min(c_mast))
        retroceso = (techo_flag - float(np.min(c_mast))) / max(altura_mast, 1e-6)
        es_compacta = ancho_flag <= (1.6 * denom) and retroceso <= 0.48
        if es_compacta:
            if precio < piso_flag and (rvol >= 1.15 or compresion_vol):
                return -2.0, "VENTA FUERTE", f"Ruptura Bear Flag < ${piso_flag:.2f} (mástil {rango_mast_atr:.1f} ATR)", -2.0, "VENTA"
            if precio <= techo_flag and compresion_vol:
                return -1.0, "VENTA", f"Bear Flag en consolidación [${piso_flag:.2f}-${techo_flag:.2f}]", -1.1, None

    return 0.0, "NEUTRAL", "Sin bandera definida", 0.0, None


def detectar_rebote_sobreventa(
    df_intra: pd.DataFrame,
    rsi_serie: pd.Series,
    bb_pct_b_serie: pd.Series,
    soporte_1: float,
    resistencia_1: float,
    cmf_val: float,
    rvol: float,
) -> Tuple[float, str, str, float, Optional[str]]:
    """
    Detecta Rebotes confirmados por Sobreventa (en soporte o RSI/BB extremos girando al alza)
    y simétricamente Rechazos por Sobrecompra.
    Evita 'atrapar cuchillos': exige confirmación de giro de vela + momentum/flujo.
    """
    if len(df_intra) < 8:
        return 0.0, "NEUTRAL", "Rebote N/D", 0.0, None

    c_actual = _safe_float(df_intra["Close"].iloc[-1])
    o_actual = _safe_float(df_intra["Open"].iloc[-1])
    rsi_actual = _safe_float(rsi_serie.iloc[-1], 50.0)
    rsi_prev_min = _safe_float(rsi_serie.iloc[-6:-1].min(), 50.0)
    bb_prev_min = _safe_float(bb_pct_b_serie.iloc[-6:-1].min(), 0.5)
    cerca_s1 = c_actual <= (soporte_1 * 1.02)

    vela_verde = c_actual > o_actual
    giro_alcista = rsi_actual > (rsi_prev_min + 2.5) and _safe_float(df_intra["Close"].iloc[-1]) > _safe_float(df_intra["Close"].iloc[-2])

    # Rebote por sobreventa confirmado
    sobreventa_prev = (rsi_prev_min <= 33.0) or (bb_prev_min <= 0.12)
    if sobreventa_prev and giro_alcista and vela_verde:
        if (cerca_s1 or cmf_val >= 0.0 or rvol >= 1.2) and rsi_prev_min <= 28.0:
            return 2.0, "COMPRA FUERTE", f"Rebote Sobreventa confirmado (RSI min {rsi_prev_min:.1f}->{rsi_actual:.1f}, RVOL {rvol:.2f}x)", 1.9, "COMPRA"
        return 1.0, "COMPRA", f"Rebote desde sobreventa (RSI min {rsi_prev_min:.1f}->{rsi_actual:.1f})", 1.0, None

    # Rechazo por sobrecompra confirmado
    rsi_prev_max = _safe_float(rsi_serie.iloc[-6:-1].max(), 50.0)
    bb_prev_max = _safe_float(bb_pct_b_serie.iloc[-6:-1].max(), 0.5)
    cerca_r1 = c_actual >= (resistencia_1 * 0.98)
    vela_roja = c_actual < o_actual
    giro_bajista = rsi_actual < (rsi_prev_max - 2.5) and _safe_float(df_intra["Close"].iloc[-1]) < _safe_float(df_intra["Close"].iloc[-2])

    sobrecompra_prev = (rsi_prev_max >= 69.0) or (bb_prev_max >= 0.88)
    if sobrecompra_prev and giro_bajista and vela_roja:
        if (cerca_r1 or cmf_val <= 0.0 or rvol >= 1.2) and rsi_prev_max >= 73.0:
            return -2.0, "VENTA FUERTE", f"Rechazo Sobrecompra (RSI max {rsi_prev_max:.1f}->{rsi_actual:.1f}, RVOL {rvol:.2f}x)", -1.9, "VENTA"
        return -1.0, "VENTA", f"Giro desde sobrecompra (RSI max {rsi_prev_max:.1f}->{rsi_actual:.1f})", -1.0, None

    return 0.0, "NEUTRAL", f"Sin rebote/rechazo extremo (RSI={rsi_actual:.1f})", 0.0, None


def evaluar_tendencia_diaria_squeeze(
    df_diario: Optional[pd.DataFrame],
    precio_actual: float,
) -> Tuple[float, str, str, float, Optional[str]]:
    """
    Evalúa el marco diario (SMA 20/50 + compresión de volatilidad BB vs Keltner Squeeze)
    para alinear operaciones swing y filtrar ruido intradía.
    """
    if df_diario is None or len(df_diario) < 25:
        return 0.0, "NEUTRAL", "Tendencia diaria insuficiente", 0.0, None

    d_close = df_diario["Close"].astype(float)
    d_high = df_diario["High"].astype(float)
    d_low = df_diario["Low"].astype(float)
    sma20 = _safe_float(d_close.rolling(20).mean().iloc[-1], precio_actual)
    sma50 = _safe_float(d_close.rolling(min(50, len(d_close))).mean().iloc[-1], sma20)

    # Detección de Squeeze (ancho BB estrecho frente a ATR diario)
    std20 = _safe_float(d_close.rolling(20).std(ddof=0).iloc[-1], 0.0)
    atr_d = _safe_float(calcular_atr(d_high, d_low, d_close, 14).iloc[-1], precio_actual * 0.015)
    squeeze_activo = (2.0 * std20) < (1.55 * max(atr_d, 1e-4))
    expansion_hoy = _safe_float(d_close.iloc[-1] - d_close.iloc[-2], 0.0)

    if precio_actual > sma20 > sma50:
        if squeeze_activo and expansion_hoy > 0:
            return 2.0, "COMPRA FUERTE", f"Tendencia Diaria Alcista + Squeeze Release (SMA20 ${sma20:.2f} > SMA50 ${sma50:.2f})", 2.0, "COMPRA"
        return 1.0, "COMPRA", f"Tendencia Diaria Alcista (P>${sma20:.2f}>SMA50 ${sma50:.2f})", 1.0, None

    if precio_actual < sma20 < sma50:
        if squeeze_activo and expansion_hoy < 0:
            return -2.0, "VENTA FUERTE", f"Tendencia Diaria Bajista + Squeeze Breakdown (SMA20 ${sma20:.2f} < SMA50 ${sma50:.2f})", -2.0, "VENTA"
        return -1.0, "VENTA", f"Tendencia Diaria Bajista (P<${sma20:.2f}<SMA50 ${sma50:.2f})", -1.0, None

    if precio_actual > sma20:
        return 0.5, "COMPRA", f"Sobre SMA20 diaria (${sma20:.2f})", 0.5, None
    if precio_actual < sma20:
        return -0.5, "VENTA", f"Bajo SMA20 diaria (${sma20:.2f})", -0.5, None
    return 0.0, "NEUTRAL", f"Consolidación en SMA20 (${sma20:.2f})", 0.0, None


# ==============================================================================
# 3. GESTOR PRINCIPAL DEL ESCÁNER (`ScannerMercado`)
# ==============================================================================

class ScannerMercado:
    """
    Descarga velas intradía y diarias por lotes, fundamentales, fechas de próximos
    reportes e ingresos esperados/reportados, y calcula 17 indicadores técnicos/patrones.
    """

    def __init__(
        self,
        tickers: List[str],
        modo_operacion: str,
        historial: GestorHistorial,
        callback_evento_alto_potencial: Optional[Callable[[AnalisisTecnicoTicker], None]] = None,
    ) -> None:
        self.tickers = [t.strip().upper() for t in tickers if t.strip()]
        self.modo_operacion = modo_operacion.strip().lower()  # "spot" o "short y long"
        self.historial = historial
        self.callback_evento = callback_evento_alto_potencial

        self._lock = threading.RLock()
        self._snapshots_actuales: Dict[str, AnalisisTecnicoTicker] = {}
        self._historial_ticker: Dict[str, List[AnalisisTecnicoTicker]] = {t: [] for t in self.tickers}
        self._cache_fundamentales: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_reportes: Dict[str, DatosReporteResultados] = {}
        self._ultimo_evento_emitido: Dict[str, Tuple[float, str]] = {}

        self._detener = threading.Event()
        self._hilo_ciclo: Optional[threading.Thread] = None
        self._hilo_fundamentales: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # 1. CACHÉ ASÍNCRONO DE FUNDAMENTALES Y REPORTES DE RESULTADOS
    # ------------------------------------------------------------------

    def _obtener_fundamentales_cacheados(self, ticker: str) -> Dict[str, Any]:
        with self._lock:
            entrada = self._cache_fundamentales.get(ticker)
            rep = self._cache_reportes.get(ticker)
        base = dict(entrada[1]) if entrada is not None else {
            "pe_ratio": None,
            "pe_forward": None,
            "cap_mercado_b": None,
            "max_52s": None,
            "min_52s": None,
            "beta": None,
        }
        base["reporte_resultados"] = rep
        return base

    @staticmethod
    def _extraer_datos_reporte_ticker(tk_usr: str, tk_obj: yf.Ticker, info: Dict[str, Any]) -> DatosReporteResultados:
        """
        Extrae fecha del próximo reporte, último reporte, ingresos esperados,
        ingresos reales reportados, EPS esperado/reportado y evaluación conjunta base.
        """
        ahora_date = datetime.now().date()
        fecha_prox: Optional[str] = None
        dias_hasta: Optional[int] = None
        fecha_ult: Optional[str] = None
        eps_esp: Optional[float] = None
        eps_rep: Optional[float] = None
        eps_sorp: Optional[float] = None
        ing_esp: Optional[float] = None
        ing_rep: Optional[float] = None
        ing_yoy: Optional[float] = None

        # 1. earnings_dates distingue con precisión filas futuras (Reported EPS == NaN) vs pasadas
        try:
            ed = tk_obj.earnings_dates
            if ed is not None and not ed.empty:
                ed = ed.sort_index(ascending=False)
                futuras: List[Tuple[datetime.date, Optional[float]]] = []
                pasadas: List[Tuple[datetime.date, Optional[float], Optional[float], Optional[float]]] = []
                for idx_ts, row in ed.iterrows():
                    try:
                        d_row = pd.Timestamp(idx_ts).date()
                    except Exception:
                        continue
                    e_est = _safe_opt_float(row.get("EPS Estimate"))
                    e_rep = _safe_opt_float(row.get("Reported EPS"))
                    e_srp = _safe_opt_float(row.get("Surprise(%)"))
                    if e_rep is None and d_row >= ahora_date:
                        futuras.append((d_row, e_est))
                    elif e_rep is not None:
                        pasadas.append((d_row, e_est, e_rep, e_srp))

                if futuras:
                    futuras.sort(key=lambda x: x[0])
                    fecha_prox = futuras[0][0].isoformat()
                    dias_hasta = (futuras[0][0] - ahora_date).days
                    if futuras[0][1] is not None:
                        eps_esp = round(futuras[0][1], 3)

                if pasadas:
                    pasadas.sort(key=lambda x: x[0], reverse=True)
                    d_ult, e_u_est, e_u_rep, e_u_srp = pasadas[0]
                    fecha_ult = d_ult.isoformat()
                    eps_rep = round(e_u_rep, 3) if e_u_rep is not None else None
                    if eps_esp is None and e_u_est is not None:
                        eps_esp = round(e_u_est, 3)
                    if e_u_srp is not None:
                        eps_sorp = round(e_u_srp, 2)
                    elif e_u_est is not None and abs(e_u_est) > 1e-6 and e_u_rep is not None:
                        eps_sorp = round(((e_u_rep - e_u_est) / abs(e_u_est)) * 100.0, 2)
        except Exception:
            pass

        # 2. calendar aporta Revenue Average y fallback de próxima fecha/EPS esperado
        try:
            cal = tk_obj.calendar or {}
            if isinstance(cal, dict):
                ing_esp = _safe_opt_float(cal.get("Revenue Average"))
                if eps_esp is None:
                    val_eps_cal = _safe_opt_float(cal.get("Earnings Average"))
                    if val_eps_cal is not None:
                        eps_esp = round(val_eps_cal, 3)
                if fecha_prox is None:
                    edates = cal.get("Earnings Date") or []
                    if isinstance(edates, list) and edates:
                        d_cal = pd.Timestamp(edates[0]).date()
                        if d_cal >= ahora_date:
                            fecha_prox = d_cal.isoformat()
                            dias_hasta = (d_cal - ahora_date).days
                        elif fecha_ult is None:
                            fecha_ult = d_cal.isoformat()
        except Exception:
            pass

        # 3. revenue_estimate fallback para ingresos esperados del trimestre actual ('0q')
        if ing_esp is None:
            try:
                re_df = tk_obj.revenue_estimate
                if re_df is not None and not re_df.empty and "avg" in re_df.columns:
                    if "0q" in re_df.index:
                        ing_esp = _safe_opt_float(re_df.loc["0q", "avg"])
                    else:
                        ing_esp = _safe_opt_float(re_df["avg"].iloc[0])
            except Exception:
                pass

        # 4. quarterly_income_stmt aporta los ingresos reales más recientes reportados y YoY
        try:
            qis = tk_obj.quarterly_income_stmt
            if qis is not None and not qis.empty and "Total Revenue" in qis.index:
                serie_rev = qis.loc["Total Revenue"].dropna()
                if len(serie_rev) >= 1:
                    ing_rep = _safe_opt_float(serie_rev.iloc[0])
                    if fecha_ult is None:
                        try:
                            fecha_ult = pd.Timestamp(serie_rev.index[0]).date().isoformat()
                        except Exception:
                            pass
                    if len(serie_rev) >= 5:
                        rev_yoy_base = _safe_opt_float(serie_rev.iloc[4])
                        if ing_rep is not None and rev_yoy_base is not None and abs(rev_yoy_base) > 1e-6:
                            ing_yoy = round(((ing_rep - rev_yoy_base) / abs(rev_yoy_base)) * 100.0, 2)
        except Exception:
            pass

        if ing_yoy is None:
            rev_g = _safe_opt_float(info.get("revenueGrowth"))
            if rev_g is not None:
                ing_yoy = round(rev_g * 100.0, 2)
        if ing_rep is None:
            ing_rep = _safe_opt_float(info.get("totalRevenue"))

        # Sorpresa relativa entre ingresos reales últimos vs consenso esperado actual/reciente (si aplica)
        ing_sorp: Optional[float] = None
        if ing_rep is not None and ing_esp is not None and ing_esp > 0 and dias_hasta is None:
            ing_sorp = round(((ing_rep - ing_esp) / ing_esp) * 100.0, 2)

        # Evaluación sintética del reporte
        eval_rep = "PENDIENTE"
        if dias_hasta is not None and 0 <= dias_hasta <= 10:
            eval_rep = "PROXIMO_INMINENTE"
        elif eps_sorp is not None:
            if eps_sorp >= 4.0 and (ing_yoy is None or ing_yoy >= 0.0):
                eval_rep = "POSITIVO"
            elif eps_sorp <= -4.0 or (ing_yoy is not None and ing_yoy <= -6.0):
                eval_rep = "NEGATIVO"
            else:
                eval_rep = "MIXTO"
        elif ing_yoy is not None:
            eval_rep = "POSITIVO" if ing_yoy >= 8.0 else ("NEGATIVO" if ing_yoy <= -5.0 else "MIXTO")

        partes_res: List[str] = []
        if fecha_prox:
            d_txt = f" ({dias_hasta}d)" if dias_hasta is not None else ""
            partes_res.append(f"Próx. reporte: {fecha_prox}{d_txt}")
        if ing_esp is not None:
            partes_res.append(f"Ing. esp: {formatear_monto_compacto(ing_esp)}")
        if ing_rep is not None:
            yoy_txt = f" ({ing_yoy:+.1f}% YoY)" if ing_yoy is not None else ""
            partes_res.append(f"Ing. rep: {formatear_monto_compacto(ing_rep)}{yoy_txt}")
        if eps_rep is not None:
            srp_txt = f" ({eps_sorp:+.1f}% sorp)" if eps_sorp is not None else ""
            partes_res.append(f"EPS rep: {eps_rep:.2f}{srp_txt}")
        elif eps_esp is not None:
            partes_res.append(f"EPS esp: {eps_esp:.2f}")

        return DatosReporteResultados(
            ticker=tk_usr,
            fecha_proximo_reporte=fecha_prox,
            dias_hasta_reporte=dias_hasta,
            fecha_ultimo_reporte=fecha_ult,
            ingresos_esperados=ing_esp,
            ingresos_reportados=ing_rep,
            ingresos_crecimiento_yoy_pct=ing_yoy,
            ingresos_sorpresa_pct=ing_sorp,
            eps_esperado=eps_esp,
            eps_reportado=eps_rep,
            eps_sorpresa_pct=eps_sorp,
            evaluacion_reporte=eval_rep,
            resumen_reporte=" | ".join(partes_res) if partes_res else "Sin calendario/estimados publicados",
            actualizado_epoch=time.time(),
        )

    def _refrescar_fundamentales_worker(self) -> None:
        """Enriquece P/E, Market Cap, Beta y Reportes de Resultados en paralelo sin bloquear velas."""
        def _consultar_uno(tk_usr: str) -> Tuple[str, Dict[str, Any], Optional[DatosReporteResultados]]:
            try:
                sym = simbolo_yfinance(tk_usr)
                tk_obj = yf.Ticker(sym)
                info = tk_obj.info or {}
                pe = info.get("trailingPE")
                fpe = info.get("forwardPE")
                mcap = info.get("marketCap")
                hi52 = info.get("fiftyTwoWeekHigh")
                lo52 = info.get("fiftyTwoWeekLow")
                beta = info.get("beta")
                datos_f = {
                    "pe_ratio": round(float(pe), 2) if pe is not None else None,
                    "pe_forward": round(float(fpe), 2) if fpe is not None else None,
                    "cap_mercado_b": round(float(mcap) / 1e9, 2) if mcap is not None else None,
                    "max_52s": round(float(hi52), 2) if hi52 is not None else None,
                    "min_52s": round(float(lo52), 2) if lo52 is not None else None,
                    "beta": round(float(beta), 2) if beta is not None else None,
                }
                rep = self._extraer_datos_reporte_ticker(tk_usr, tk_obj, info)
                return tk_usr, datos_f, rep
            except Exception:
                return tk_usr, {
                    "pe_ratio": None,
                    "pe_forward": None,
                    "cap_mercado_b": None,
                    "max_52s": None,
                    "min_52s": None,
                    "beta": None,
                }, None

        while not self._detener.is_set():
            ahora = time.time()
            pendientes = []
            with self._lock:
                for t in self.tickers:
                    cache = self._cache_fundamentales.get(t)
                    if cache is None or (ahora - cache[0]) > 1800:
                        pendientes.append(t)

            # Procesar en bloques moderados de 12 para no saturar Yahoo
            for i in range(0, len(pendientes), 12):
                if self._detener.is_set():
                    return
                sub = pendientes[i : i + 12]
                pool = ThreadPoolExecutor(max_workers=6)
                try:
                    futuros_f = [pool.submit(_consultar_uno, tk_s) for tk_s in sub]
                    for fut_f in futuros_f:
                        if self._detener.is_set():
                            break
                        tk_res, datos_f, rep_res = fut_f.result()
                        with self._lock:
                            self._cache_fundamentales[tk_res] = (time.time(), datos_f)
                            if rep_res is not None:
                                self._cache_reportes[tk_res] = rep_res
                                snap_act = self._snapshots_actuales.get(tk_res)
                                if snap_act is not None:
                                    snap_act.datos_financieros.reporte_resultados = rep_res
                finally:
                    pool.shutdown(wait=not self._detener.is_set(), cancel_futures=True)
                self._detener.wait(timeout=0.8)

            self._detener.wait(timeout=60.0)

    def obtener_o_consultar_reporte_puntual(self, ticker: str) -> Optional[DatosReporteResultados]:
        """Devuelve el reporte de resultados en caché o lo descarga puntualmente para análisis conjunto."""
        tk = ticker.strip().upper()
        with self._lock:
            rep = self._cache_reportes.get(tk)
            if rep is not None and (time.time() - rep.actualizado_epoch) < 1800:
                return rep
        try:
            sym = simbolo_yfinance(tk)
            tk_obj = yf.Ticker(sym)
            info = tk_obj.info or {}
            rep_nuevo = self._extraer_datos_reporte_ticker(tk, tk_obj, info)
            with self._lock:
                self._cache_reportes[tk] = rep_nuevo
                if tk in self._snapshots_actuales:
                    self._snapshots_actuales[tk].datos_financieros.reporte_resultados = rep_nuevo
            return rep_nuevo
        except Exception:
            return rep

    def obtener_reporte_resultados(self, ticker: str) -> Optional[DatosReporteResultados]:
        tk = ticker.strip().upper()
        with self._lock:
            return self._cache_reportes.get(tk)

    def obtener_reportes_resultados_todos(self) -> Dict[str, DatosReporteResultados]:
        with self._lock:
            return dict(self._cache_reportes)

    # ------------------------------------------------------------------
    # 2. EXTRACCIÓN SEGURA DE DATAFRAME POR TICKER DESDE LOTE YFINANCE
    # ------------------------------------------------------------------

    @staticmethod
    def _extraer_sub_df(df_lote: pd.DataFrame, simbolo_yf: str, total_simbolos: int) -> Optional[pd.DataFrame]:
        """Extrae limpiamente las columnas Open, High, Low, Close, Volume de `yf.download`."""
        if df_lote is None or df_lote.empty:
            return None
        try:
            if isinstance(df_lote.columns, pd.MultiIndex):
                niveles_0 = set(df_lote.columns.get_level_values(0))
                niveles_1 = set(df_lote.columns.get_level_values(1))
                if simbolo_yf in niveles_0:
                    sub = df_lote[simbolo_yf].copy()
                elif simbolo_yf in niveles_1:
                    sub = df_lote.xs(simbolo_yf, axis=1, level=1).copy()
                else:
                    return None
            else:
                if total_simbolos == 1:
                    sub = df_lote.copy()
                else:
                    return None

            cols_req = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in sub.columns for c in cols_req):
                return None
            sub = sub[cols_req].dropna(subset=["High", "Low", "Close"])
            if len(sub) < 22:
                return None
            return sub
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 3. EVALUACIÓN COMPLETA DE LOS 17 INDICADORES + PATRONES SWING
    # ------------------------------------------------------------------

    def _evaluar_ticker_dataframe(
        self,
        ticker: str,
        df_intra: pd.DataFrame,
        df_diario: Optional[pd.DataFrame],
    ) -> AnalisisTecnicoTicker:
        """Calcula datos financieros, niveles S/R, los 17 indicadores/patrones y el score de certeza."""
        high = df_intra["High"].astype(float)
        low = df_intra["Low"].astype(float)
        close = df_intra["Close"].astype(float)
        volume = df_intra["Volume"].fillna(0.0).astype(float)

        precio_actual = _safe_float(close.iloc[-1])
        atr_serie = calcular_atr(high, low, close, periodo=14)
        atr_14 = max(_safe_float(atr_serie.iloc[-1], precio_actual * 0.01), 1e-4)

        # Cambio porcentual diario y niveles estructurales Soporte/Resistencia (Pivotes + Fractales)
        if df_diario is not None and len(df_diario) >= 5:
            d_high = df_diario["High"].astype(float)
            d_low = df_diario["Low"].astype(float)
            d_close = df_diario["Close"].astype(float)
            cierre_previo = _safe_float(d_close.iloc[-2], _safe_float(close.iloc[0], precio_actual))
            cambio_pct = ((precio_actual - cierre_previo) / max(cierre_previo, 1e-6)) * 100.0

            h_prev = _safe_float(d_high.iloc[-2], _safe_float(high.max()))
            l_prev = _safe_float(d_low.iloc[-2], _safe_float(low.min()))
            pivote = (h_prev + l_prev + cierre_previo) / 3.0
            r1_piv = 2.0 * pivote - l_prev
            s1_piv = 2.0 * pivote - h_prev
            swing_hi = _safe_float(d_high.tail(20).max(), r1_piv)
            swing_lo = _safe_float(d_low.tail(20).min(), s1_piv)

            soporte_1 = round(min(precio_actual * 0.995, max(s1_piv, swing_lo)), 2)
            soporte_2 = round(min(soporte_1 * 0.985, swing_lo), 2)
            resistencia_1 = round(max(precio_actual * 1.005, min(r1_piv, swing_hi)), 2)
            resistencia_2 = round(max(resistencia_1 * 1.015, swing_hi), 2)
            max_52_calc = round(_safe_float(d_high.max(), precio_actual), 2)
            min_52_calc = round(_safe_float(d_low.min(), precio_actual), 2)
        else:
            cierre_ini = _safe_float(close.iloc[-min(len(close), 26)], precio_actual)
            cambio_pct = ((precio_actual - cierre_ini) / max(cierre_ini, 1e-6)) * 100.0
            soporte_1 = round(_safe_float(low.tail(30).min(), precio_actual - 1.5 * atr_14), 2)
            soporte_2 = round(soporte_1 - 1.5 * atr_14, 2)
            resistencia_1 = round(_safe_float(high.tail(30).max(), precio_actual + 1.5 * atr_14), 2)
            resistencia_2 = round(resistencia_1 + 1.5 * atr_14, 2)
            max_52_calc = None
            min_52_calc = None

        vwap_sesion, vwap_z = calcular_vwap_sesion_zscore(df_intra)
        vol_barra = _safe_float(volume.iloc[-1], 0.0)
        vol_prom_20 = max(_safe_float(volume.tail(20).mean(), 1.0), 1.0)

        fund = self._obtener_fundamentales_cacheados(ticker)
        datos_fin = DatosFinancieros(
            ticker=ticker,
            precio_actual=round(precio_actual, 2),
            cambio_diario_pct=round(cambio_pct, 2),
            pe_ratio=fund["pe_ratio"],
            pe_forward=fund["pe_forward"],
            soporte_1=soporte_1,
            soporte_2=soporte_2,
            resistencia_1=resistencia_1,
            resistencia_2=resistencia_2,
            vwap_sesion=round(vwap_sesion, 2),
            atr_14=round(atr_14, 3),
            volumen_barra=round(vol_barra, 0),
            volumen_promedio_20=round(vol_prom_20, 0),
            cap_mercado_b=fund["cap_mercado_b"],
            max_52_semanas=fund["max_52s"] or max_52_calc,
            min_52_semanas=fund["min_52s"] or min_52_calc,
            beta=fund["beta"],
            reporte_resultados=fund.get("reporte_resultados"),
        )

        indicadores: List[ResultadoIndicador] = []
        patrones_detectados: List[str] = []

        # --- 1. Oscilador (contrarian) | RSI(14) | cortes 20/35/65/80 ---
        rsi_serie = calcular_rsi_serie(close, 14)
        rsi_val = _safe_float(rsi_serie.iloc[-1], 50.0)
        if rsi_val <= 20.0:
            ev, pts, alm = "COMPRA FUERTE", 2.0, "COMPRA"
        elif rsi_val <= 35.0:
            ev, pts, alm = "COMPRA", 1.0, None
        elif rsi_val < 65.0:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif rsi_val < 80.0:
            ev, pts, alm = "VENTA", -1.0, None
        else:
            ev, pts, alm = "VENTA FUERTE", -2.0, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Oscilador (contrarian)", "RSI(14)", round(rsi_val, 2),
            f"{rsi_val:.2f} [20/35/65/80]", ev, pts, alm,
        ))

        # --- 2. Oscilador (contrarian) | Stoch(14,3,3) | cortes 10/20/80/90 ---
        stoch_k, stoch_d = calcular_stoch(high, low, close, 14, 3, 3)
        if stoch_k <= 10.0:
            ev, pts, alm = "COMPRA FUERTE", 2.0, "COMPRA"
        elif stoch_k <= 20.0:
            ev, pts, alm = "COMPRA", 1.0, "COMPRA" if stoch_k > stoch_d else None
        elif stoch_k < 80.0:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif stoch_k < 90.0:
            ev, pts, alm = "VENTA", -1.0, "VENTA" if stoch_k < stoch_d else None
        else:
            ev, pts, alm = "VENTA FUERTE", -2.0, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Oscilador (contrarian)", "Stoch(14,3,3)", round(stoch_k, 2),
            f"%K={stoch_k:.1f} %D={stoch_d:.1f} [10/20/80/90]", ev, pts, alm,
        ))

        # --- 3. Oscilador (contrarian) | Bollinger %B(20,2) | cortes 0/0.2/0.8/1 ---
        bb_serie = calcular_bollinger_pct_b_serie(close, 20, 2.0)
        bb_pct_b = _safe_float(bb_serie.iloc[-1], 0.5)
        if bb_pct_b <= 0.0:
            ev, pts, alm = "COMPRA FUERTE", 2.0, "COMPRA"
        elif bb_pct_b <= 0.2:
            ev, pts, alm = "COMPRA", 1.0, None
        elif bb_pct_b < 0.8:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif bb_pct_b < 1.0:
            ev, pts, alm = "VENTA", -1.0, None
        else:
            ev, pts, alm = "VENTA FUERTE", -2.0, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Oscilador (contrarian)", "Bollinger %B(20,2)", round(bb_pct_b, 3),
            f"{bb_pct_b:.3f} [0/0.2/0.8/1]", ev, pts, alm,
        ))

        # --- 4. Oscilador (contrarian) | MFI(14), con volumen | cortes 15/30/70/85 ---
        mfi_val = calcular_mfi(high, low, close, volume, 14)
        if mfi_val <= 15.0:
            ev, pts, alm = "COMPRA FUERTE", 2.0, "COMPRA"
        elif mfi_val <= 30.0:
            ev, pts, alm = "COMPRA", 1.0, None
        elif mfi_val < 70.0:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif mfi_val < 85.0:
            ev, pts, alm = "VENTA", -1.0, None
        else:
            ev, pts, alm = "VENTA FUERTE", -2.0, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Oscilador (contrarian)", "MFI(14) vol", round(mfi_val, 2),
            f"{mfi_val:.2f} [15/30/70/85]", ev, pts, alm,
        ))

        # --- 5. Tendencia | EMA 9/21 | spread dividido por ATR ---
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema_spread_atr = _safe_float((ema9.iloc[-1] - ema21.iloc[-1]) / atr_14, 0.0)
        cruce_ema_alcista = bool(ema9.iloc[-1] > ema21.iloc[-1] and ema9.iloc[-2] <= ema21.iloc[-2])
        cruce_ema_bajista = bool(ema9.iloc[-1] < ema21.iloc[-1] and ema9.iloc[-2] >= ema21.iloc[-2])
        if ema_spread_atr >= 1.0 or cruce_ema_alcista:
            ev = "COMPRA FUERTE" if ema_spread_atr >= 1.0 else "COMPRA"
            pts = 2.0 if ema_spread_atr >= 1.0 else 1.25
            alm = "COMPRA"
        elif ema_spread_atr >= 0.25:
            ev, pts, alm = "COMPRA", 1.0, None
        elif ema_spread_atr > -0.25:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif ema_spread_atr > -1.0 and not cruce_ema_bajista:
            ev, pts, alm = "VENTA", -1.0, None
        else:
            ev = "VENTA FUERTE" if ema_spread_atr <= -1.0 else "VENTA"
            pts = -2.0 if ema_spread_atr <= -1.0 else -1.25
            alm = "VENTA"
        indicadores.append(ResultadoIndicador(
            "Tendencia", "EMA 9/21 / ATR", round(ema_spread_atr, 3),
            f"{ema_spread_atr:+.3f} ATR", ev, pts, alm,
        ))

        # --- 6. Tendencia | MACD histograma | dividido por ATR ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_sig
        macd_atr = _safe_float(macd_hist.iloc[-1] / atr_14, 0.0)
        if macd_atr >= 0.35:
            ev, pts, alm = "COMPRA FUERTE", 2.0, "COMPRA"
        elif macd_atr >= 0.08:
            ev, pts, alm = "COMPRA", 1.0, None
        elif macd_atr > -0.08:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif macd_atr > -0.35:
            ev, pts, alm = "VENTA", -1.0, None
        else:
            ev, pts, alm = "VENTA FUERTE", -2.0, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Tendencia", "MACD Hist / ATR", round(macd_atr, 3),
            f"{macd_atr:+.3f} ATR", ev, pts, alm,
        ))

        # --- 7. Tendencia | VWAP de sesión | z-score de la distancia % ---
        if vwap_z >= 1.5:
            ev, pts, alm = "COMPRA FUERTE", 1.8, "COMPRA"
        elif vwap_z >= 0.45:
            ev, pts, alm = "COMPRA", 1.0, None
        elif vwap_z > -0.45:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif vwap_z > -1.5:
            ev, pts, alm = "VENTA", -1.0, None
        else:
            ev, pts, alm = "VENTA FUERTE", -1.8, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Tendencia", "VWAP Sesión Z-Score", round(vwap_z, 3),
            f"Z={vwap_z:+.2f} (VWAP=${vwap_sesion:.2f})", ev, pts, alm,
        ))

        # --- 8. Tendencia | ADX/DI | DI+ - DI-, vale 0 si ADX < 20 ---
        adx_val, _di_pos, _di_neg, spread_di = calcular_adx_di(high, low, close, 14)
        if spread_di >= 15.0:
            ev, pts, alm = "COMPRA FUERTE", 2.0, "COMPRA"
        elif spread_di >= 5.0:
            ev, pts, alm = "COMPRA", 1.0, None
        elif spread_di > -5.0:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif spread_di > -15.0:
            ev, pts, alm = "VENTA", -1.0, None
        else:
            ev, pts, alm = "VENTA FUERTE", -2.0, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Tendencia", "ADX/DI (DI+-DI-)", round(spread_di, 2),
            f"{spread_di:+.2f} (ADX={adx_val:.1f})", ev, pts, alm,
        ))

        # --- 9. Supertrend(10,2.5)/ATR ---
        st_atr, st_dir, st_flip = calcular_supertrend_norm(high, low, close, atr_14, 10, 2.5)
        if st_dir == 1 and (st_flip or st_atr >= 1.2):
            ev, pts, alm = "COMPRA FUERTE", 2.0, "COMPRA"
        elif st_dir == 1:
            ev, pts, alm = "COMPRA", 1.0, None
        elif st_dir == -1 and (st_flip or st_atr <= -1.2):
            ev, pts, alm = "VENTA FUERTE", -2.0, "VENTA"
        else:
            ev, pts, alm = "VENTA", -1.0, None
        indicadores.append(ResultadoIndicador(
            "Tendencia", "Supertrend(10,2.5)/ATR", round(st_atr, 3),
            f"{st_atr:+.3f} ({'ALCISTA' if st_dir == 1 else 'BAJISTA'})", ev, pts, alm,
        ))

        # --- 10. CMF(20) Flujo Inst. ---
        cmf_val = calcular_cmf(high, low, close, volume, 20)
        if cmf_val >= 0.20:
            ev, pts, alm = "COMPRA FUERTE", 2.0, "COMPRA"
        elif cmf_val >= 0.06:
            ev, pts, alm = "COMPRA", 1.0, None
        elif cmf_val > -0.06:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif cmf_val > -0.20:
            ev, pts, alm = "VENTA", -1.0, None
        else:
            ev, pts, alm = "VENTA FUERTE", -2.0, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Flujo Inst.", "CMF(20) Flujo Inst.", round(cmf_val, 3),
            f"{cmf_val:+.3f}", ev, pts, alm,
        ))

        # --- 11. Volumen Real | RVOL(20) Impulso Direccional ---
        rvol = vol_barra / vol_prom_20
        vela_alcista = precio_actual >= _safe_float(df_intra["Open"].iloc[-1], precio_actual)
        if rvol >= 1.8 and vela_alcista:
            ev, pts, alm = "COMPRA FUERTE", 1.8, "COMPRA"
        elif rvol >= 1.25 and vela_alcista:
            ev, pts, alm = "COMPRA", 0.9, None
        elif rvol >= 1.8 and not vela_alcista:
            ev, pts, alm = "VENTA FUERTE", -1.8, "VENTA"
        elif rvol >= 1.25 and not vela_alcista:
            ev, pts, alm = "VENTA", -0.9, None
        else:
            ev, pts, alm = "NEUTRAL", 0.0, None
        indicadores.append(ResultadoIndicador(
            "Volumen Real", "RVOL(20) Real", round(rvol, 2),
            f"{rvol:.2f}x prom ({'comprador' if vela_alcista else 'vendedor'})", ev, pts, alm,
        ))

        # --- 12. Volumen Real | OBV Z-Score(20) ---
        signo_cierre = np.sign(close.diff().fillna(0.0))
        obv = (signo_cierre * volume).cumsum()
        obv_win = obv.tail(20)
        obv_std = _safe_float(obv_win.std(ddof=0), 0.0)
        obv_z = (_safe_float(obv.iloc[-1]) - _safe_float(obv_win.mean())) / obv_std if obv_std > 1e-5 else 0.0
        if obv_z >= 1.6:
            ev, pts, alm = "COMPRA FUERTE", 1.5, "COMPRA"
        elif obv_z >= 0.6:
            ev, pts, alm = "COMPRA", 0.8, None
        elif obv_z > -0.6:
            ev, pts, alm = "NEUTRAL", 0.0, None
        elif obv_z > -1.6:
            ev, pts, alm = "VENTA", -0.8, None
        else:
            ev, pts, alm = "VENTA FUERTE", -1.5, "VENTA"
        indicadores.append(ResultadoIndicador(
            "Volumen Real", "OBV Z-Score(20)", round(obv_z, 2),
            f"Z={obv_z:+.2f}", ev, pts, alm,
        ))

        # --- 13. Estructural | Rango Soporte/Resistencia & BOS ---
        rango_sr = max(resistencia_1 - soporte_1, atr_14)
        pos_sr = (precio_actual - soporte_1) / rango_sr
        if precio_actual > resistencia_1 and rvol >= 1.2:
            ev, pts, alm = "COMPRA FUERTE", 1.8, "COMPRA"
            txt_sr = f"BOS Alcista > R1(${resistencia_1:.2f})"
            patrones_detectados.append("Ruptura BOS Alcista")
        elif pos_sr <= 0.18:
            ev, pts, alm = "COMPRA", 1.1, "COMPRA"
            txt_sr = f"Zona Soporte S1(${soporte_1:.2f})"
        elif precio_actual < soporte_1 and rvol >= 1.2:
            ev, pts, alm = "VENTA FUERTE", -1.8, "VENTA"
            txt_sr = f"BOS Bajista < S1(${soporte_1:.2f})"
            patrones_detectados.append("Ruptura BOS Bajista")
        elif pos_sr >= 0.85:
            ev, pts, alm = "VENTA", -1.0, None
            txt_sr = f"Zona Resistencia R1(${resistencia_1:.2f})"
        else:
            ev, pts, alm = "NEUTRAL", 0.0, None
            txt_sr = f"Rango {pos_sr*100:.0f}% [S1=${soporte_1:.2f} / R1=${resistencia_1:.2f}]"
        indicadores.append(ResultadoIndicador(
            "Estructural", "Estructura S/R & BOS", round(pos_sr, 2),
            txt_sr, ev, pts, alm,
        ))

        # --- 14. Acción de Precio | Ruptura de Rango de Apertura (ORB) ---
        v_orb, ev_orb, txt_orb, pts_orb, alm_orb = calcular_ruptura_orb(df_intra, precio_actual, atr_14, rvol)
        indicadores.append(ResultadoIndicador(
            "Acción Precio", "Ruptura Rango Apertura (ORB)", round(v_orb, 3),
            txt_orb, ev_orb, pts_orb, alm_orb,
        ))
        if alm_orb == "COMPRA":
            patrones_detectados.append("ORB Alcista")
        elif alm_orb == "VENTA":
            patrones_detectados.append("ORB Bajista")

        # --- 15. Patrón Chartista | Bandera (Bull/Bear Flag) ---
        v_flg, ev_flg, txt_flg, pts_flg, alm_flg = detectar_patron_bandera(high, low, close, volume, atr_14, rvol)
        indicadores.append(ResultadoIndicador(
            "Patrón Chartista", "Patrón Bandera (Flag)", round(v_flg, 2),
            txt_flg, ev_flg, pts_flg, alm_flg,
        ))
        if ev_flg == "COMPRA FUERTE":
            patrones_detectados.append("Bandera Alcista (Bull Flag)")
        elif ev_flg == "COMPRA":
            patrones_detectados.append("Bull Flag Formando")
        elif ev_flg == "VENTA FUERTE":
            patrones_detectados.append("Bandera Bajista (Bear Flag)")
        elif ev_flg == "VENTA":
            patrones_detectados.append("Bear Flag Formando")

        # --- 16. Reversión Swing | Rebote Sobreventa / Rechazo Sobrecompra ---
        v_reb, ev_reb, txt_reb, pts_reb, alm_reb = detectar_rebote_sobreventa(
            df_intra, rsi_serie, bb_serie, soporte_1, resistencia_1, cmf_val, rvol
        )
        indicadores.append(ResultadoIndicador(
            "Reversión Swing", "Rebote Sobreventa/Sobrecompra", round(v_reb, 2),
            txt_reb, ev_reb, pts_reb, alm_reb,
        ))
        if alm_reb == "COMPRA":
            patrones_detectados.append("Rebote Sobreventa Confirmado")
        elif alm_reb == "VENTA":
            patrones_detectados.append("Rechazo Sobrecompra")

        # --- 17. Marco Diario | Tendencia SMA 20/50 & Squeeze ---
        v_td, ev_td, txt_td, pts_td, alm_td = evaluar_tendencia_diaria_squeeze(df_diario, precio_actual)
        indicadores.append(ResultadoIndicador(
            "Marco Diario", "Tendencia Diaria SMA20/50 + Squeeze", round(v_td, 2),
            txt_td, ev_td, pts_td, alm_td,
        ))
        if alm_td == "COMPRA":
            patrones_detectados.append("Squeeze Release Alcista Diario")
        elif alm_td == "VENTA":
            patrones_detectados.append("Squeeze Breakdown Bajista Diario")

        # --- SÍNTESIS DE CONFLUENCIA, EVALUACIÓN GENERAL Y CERTEZA ---
        alarmas_compra = [ind.nombre for ind in indicadores if ind.alarma == "COMPRA"]
        alarmas_venta = [ind.nombre for ind in indicadores if ind.alarma == "VENTA"]

        suma_puntaje = sum(ind.puntaje for ind in indicadores)
        max_teorico = 32.0
        indice_neto = suma_puntaje / max_teorico  # [-1.0, +1.0]

        if indice_neto >= 0.28:
            eval_general = "COMPRA FUERTE"
            dir_potencial = "ALCISTA"
        elif indice_neto >= 0.11:
            eval_general = "COMPRA"
            dir_potencial = "ALCISTA"
        elif indice_neto <= -0.28:
            eval_general = "VENTA FUERTE"
            dir_potencial = "BAJISTA"
        elif indice_neto <= -0.11:
            eval_general = "VENTA"
            dir_potencial = "BAJISTA"
        else:
            eval_general = "NEUTRAL"
            dir_potencial = "NEUTRAL"

        n_alarmas_favor = len(alarmas_compra) if dir_potencial == "ALCISTA" else (len(alarmas_venta) if dir_potencial == "BAJISTA" else 0)
        n_alarmas_contra = len(alarmas_venta) if dir_potencial == "ALCISTA" else (len(alarmas_compra) if dir_potencial == "BAJISTA" else 0)
        respaldo_institucional = 8.0 if ((dir_potencial == "ALCISTA" and cmf_val > 0.05) or (dir_potencial == "BAJISTA" and cmf_val < -0.05)) else 0.0
        respaldo_rvol = min(10.0, max(0.0, (rvol - 1.0) * 8.0))
        bonus_patrones = min(12.0, len(patrones_detectados) * 5.0)

        potencial_score = min(
            100.0,
            abs(indice_neto) * 115.0 + n_alarmas_favor * 8.5 + respaldo_institucional + respaldo_rvol + bonus_patrones - n_alarmas_contra * 7.0,
        )
        potencial_score = round(max(0.0, potencial_score), 1)

        certeza_pct = min(
            98.0,
            40.0 + abs(indice_neto) * 65.0 + n_alarmas_favor * 5.8 + respaldo_institucional * 0.8 + bonus_patrones * 0.5 - n_alarmas_contra * 8.5,
        ) if dir_potencial != "NEUTRAL" else 28.0
        certeza_pct = round(max(15.0, certeza_pct), 1)

        operacion_compatible = (
            dir_potencial == "ALCISTA"
            or (dir_potencial == "BAJISTA" and "short" in self.modo_operacion)
        )
        es_alto_potencial = bool(
            operacion_compatible
            and potencial_score >= UMBRAL_POTENCIAL_DATOS_TERMINAL
            and certeza_pct >= UMBRAL_CERTEZA_EVENTO_TECNICO
            and n_alarmas_favor >= 2
        )

        motivo = ""
        if es_alto_potencial:
            lista_alm = ", ".join((alarmas_compra if dir_potencial == "ALCISTA" else alarmas_venta)[:4])
            pat_txt = f" | Patrones: {', '.join(patrones_detectados[:2])}" if patrones_detectados else ""
            motivo = f"{eval_general} ({certeza_pct:.0f}% certeza) | Alarmas: {lista_alm}{pat_txt}"

        return AnalisisTecnicoTicker(
            ticker=ticker,
            epoch=time.time(),
            fecha_hora_ampm=formatear_fecha_hora_ampm(),
            datos_financieros=datos_fin,
            indicadores=indicadores,
            evaluacion_general=eval_general,
            direccion_potencial=dir_potencial,
            potencial_score=potencial_score,
            certeza_pct=certeza_pct,
            alarmas_compra=alarmas_compra,
            alarmas_venta=alarmas_venta,
            patrones_detectados=patrones_detectados,
            es_evento_alto_potencial=es_alto_potencial,
            motivo_evento=motivo,
        )

    # ------------------------------------------------------------------
    # 4. EJECUCIÓN VECTORIZADA POR LOTES Y CICLO CONTINUO
    # ------------------------------------------------------------------

    def actualizar_todos_los_tickers(self) -> Dict[str, AnalisisTecnicoTicker]:
        """
        Descarga velas intradía (15m, 5d) y diarias (1d, 3mo) en lotes vectorizados rápidos,
        actualiza los 17 indicadores de toda la lista y emite eventos de alto potencial a `/d`.
        """
        mapa_yf = {t: simbolo_yfinance(t) for t in self.tickers}
        simbolos_yf = list(dict.fromkeys(mapa_yf.values()))

        ahora = time.time()
        if not hasattr(self, "_cache_df_diario") or (ahora - getattr(self, "_ts_df_diario", 0.0)) > 900:
            try:
                self._cache_df_diario = yf.download(
                    simbolos_yf,
                    period="3mo",
                    interval="1d",
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    auto_adjust=True,
                )
                self._ts_df_diario = ahora
            except Exception:
                self._cache_df_diario = pd.DataFrame()

        resultados_ciclo: Dict[str, AnalisisTecnicoTicker] = {}
        paso = 35
        for idx in range(0, len(self.tickers), paso):
            sub_tickers = self.tickers[idx : idx + paso]
            sub_yf = [mapa_yf[t] for t in sub_tickers]
            try:
                df_intra_lote = yf.download(
                    sub_yf,
                    period="5d",
                    interval="15m",
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    auto_adjust=True,
                )
            except Exception:
                continue

            for tk in sub_tickers:
                sym = mapa_yf[tk]
                df_tk_intra = self._extraer_sub_df(df_intra_lote, sym, len(sub_yf))
                if df_tk_intra is None:
                    continue
                df_tk_diario = self._extraer_sub_df(
                    getattr(self, "_cache_df_diario", pd.DataFrame()),
                    sym,
                    len(simbolos_yf),
                )
                try:
                    analisis = self._evaluar_ticker_dataframe(tk, df_tk_intra, df_tk_diario)
                    resultados_ciclo[tk] = analisis
                except Exception:
                    continue

        eventos_a_notificar: List[AnalisisTecnicoTicker] = []
        with self._lock:
            for tk, an in resultados_ciclo.items():
                self._snapshots_actuales[tk] = an
                hist_lista = self._historial_ticker.setdefault(tk, [])
                hist_lista.append(an)
                if len(hist_lista) > 6:
                    del hist_lista[:-6]

                if an.es_evento_alto_potencial:
                    prev = self._ultimo_evento_emitido.get(tk)
                    if prev is None or prev[1] != an.evaluacion_general or (ahora - prev[0]) >= 300:
                        self._ultimo_evento_emitido[tk] = (ahora, an.evaluacion_general)
                        eventos_a_notificar.append(an)

        for ev_tk in eventos_a_notificar:
            self.historial.registrar_evento_datos_stream(ev_tk.to_dict())
            if self.callback_evento is not None:
                try:
                    self.callback_evento(ev_tk)
                except Exception:
                    pass

        return resultados_ciclo

    def consultar_ticker_puntual(self, ticker: str) -> Optional[AnalisisTecnicoTicker]:
        """Obtiene el snapshot más reciente o fuerza cálculo puntual inmediato."""
        tk = ticker.strip().upper()
        with self._lock:
            existente = self._snapshots_actuales.get(tk)
            if existente is not None and (time.time() - existente.epoch) < 45.0:
                return existente

        sym = simbolo_yfinance(tk)
        try:
            df_intra = yf.download(sym, period="5d", interval="15m", progress=False, auto_adjust=True)
            df_sub = self._extraer_sub_df(df_intra, sym, 1)
            if df_sub is None:
                return existente
            df_diario = self._extraer_sub_df(getattr(self, "_cache_df_diario", pd.DataFrame()), sym, len(self.tickers))
            analisis = self._evaluar_ticker_dataframe(tk, df_sub, df_diario)
            with self._lock:
                self._snapshots_actuales[tk] = analisis
                hist_lista = self._historial_ticker.setdefault(tk, [])
                hist_lista.append(analisis)
                if len(hist_lista) > 6:
                    del hist_lista[:-6]
            if analisis.es_evento_alto_potencial:
                self.historial.registrar_evento_datos_stream(analisis.to_dict())
            return analisis
        except Exception:
            return existente

    def obtener_snapshot_e_historial(self, ticker: str) -> Tuple[Optional[AnalisisTecnicoTicker], List[AnalisisTecnicoTicker]]:
        """Devuelve `(snapshot_actual, snapshots_previos)` para ampliar contexto de `analisis_ai.py`."""
        tk = ticker.strip().upper()
        with self._lock:
            actual = self._snapshots_actuales.get(tk)
            previos = list(self._historial_ticker.get(tk, []))[:-1]
            return actual, previos

    def obtener_snapshots_todos(self) -> Dict[str, AnalisisTecnicoTicker]:
        """Devuelve una copia del mapa actual de snapshots técnicos para evaluación multi-ticker y sectorial."""
        with self._lock:
            return dict(self._snapshots_actuales)

    def obtener_mapa_precios_y_pe(self) -> Dict[str, Dict[str, Any]]:
        """Entrega precios actuales, P/E y S/R de todos los activos para valoración en vivo de `cartera.py`."""
        with self._lock:
            return {
                tk: {
                    "precio": snap.datos_financieros.precio_actual,
                    "pe": snap.datos_financieros.pe_ratio,
                    "soporte": snap.datos_financieros.soporte_1,
                    "resistencia": snap.datos_financieros.resistencia_1,
                    "evaluacion": snap.evaluacion_general,
                    "direccion": snap.direccion_potencial,
                    "certeza": snap.certeza_pct,
                }
                for tk, snap in self._snapshots_actuales.items()
            }

    def iniciar_monitoreo_continuo(self, callback_ciclo_precios: Optional[Callable[[Dict[str, Dict]], None]] = None) -> None:
        """Lanza el hilo continuo de cálculo cuantitativo y el hilo asíncrono de fundamentales/reportes."""
        self._detener.clear()

        if self._hilo_fundamentales is None or not self._hilo_fundamentales.is_alive():
            self._hilo_fundamentales = threading.Thread(
                target=self._refrescar_fundamentales_worker,
                name="FundamentalesReportesThread",
                daemon=True,
            )
            self._hilo_fundamentales.start()

        def _bucle() -> None:
            while not self._detener.is_set():
                try:
                    self.actualizar_todos_los_tickers()
                    if callback_ciclo_precios is not None:
                        callback_ciclo_precios(self.obtener_mapa_precios_y_pe())
                except Exception:
                    pass
                self._detener.wait(timeout=INTERVALO_CICLO_INDICADORES_SEG)

        if self._hilo_ciclo is None or not self._hilo_ciclo.is_alive():
            self._hilo_ciclo = threading.Thread(
                target=_bucle,
                name="ScannerCuantitativoThread",
                daemon=True,
            )
            self._hilo_ciclo.start()

    def detener(self) -> None:
        self._detener.set()


# Alias de compatibilidad hacia atrás
GestorIndicadores = ScannerMercado
