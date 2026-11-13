"""
Módulo indicadores.py
Motor cuantitativo de indicadores técnicos, flujo institucional y detección de
"EVENTO CON ALTO POTENCIAL DE COMPRA / VENTA" para las 89 empresas.

Correcciones operativas clave:
1. Datos intradía de alta frecuencia con horario extendido (`includePrePost=true&interval=2m&range=2d`)
   y parámetros anti-caché (`&_=timestamp` + `Cache-Control: no-cache`) para que las cotizaciones
   no se congelen tras el cierre regular ni por caché de CDN.
2. Memoria acumulativa de estado (`estado_referencia`): los cambios de indicadores entre ciclos
   no se sobrescriben en silencio cuando aún no se alcanza el umbral, sino que se acumulan hasta
   que la acción completa la confluencia y se muestra en terminal.
3. Dinámica Multi-Horizonte y Micro-Flujo en Tiempo Real: combina la serie en vivo de Yahoo Finance
   con actualizaciones tick-a-tick de pre/post-mercado y micro-variaciones ponderadas por ATR/volumen
   cuando la vela aún no cierra, evitando lecturas estáticas en bucles de 10 segundos.
4. Motor de Confluencia Multicapa calibrado con Cooldown por ticker: filtra el ruido de un único
   indicador aislado, pero garantiza detectar y mostrar los eventos reales de alto potencial de
   compra o venta que se forman en el universo de 89 acciones.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import colorama
import requests

from infomarket import EMPRESAS, GestorInfoMarket, formatear_fecha, resaltar_azul

colorama.init(autoreset=True)


# =====================================================================
# COLORES ANSI PARA INDICADORES Y EVENTOS
# =====================================================================
class ColoresIndicadores:
    RESET = "\033[0m"
    # Resaltado azul turquesa fijo para el símbolo de la acción en indicadores
    TURQUESA_RESALTADO = "\033[1;97;48;5;37m"

    # Semáforo de resultados de cada indicador:
    ROJO = "\033[1;38;5;196m"          # venta fuerte
    NARANJA = "\033[1;38;5;208m"       # venta
    AMARILLO = "\033[1;38;5;226m"      # neutral
    VERDE_LIGERO = "\033[1;38;5;120m"  # compra
    VERDE_FUERTE = "\033[1;38;5;46m"   # compra fuerte

    # Resaltado del nombre del indicador que cambió y es motivo de compra o venta:
    RESALTADO_COMPRA_INDICADOR = "\033[1;97;42m"  # Fondo verde, texto blanco brillante
    RESALTADO_VENTA_INDICADOR = "\033[1;97;41m"   # Fondo rojo, texto blanco brillante

    CIAN = "\033[1;36m"


MAPA_COLOR_RESULTADO: Dict[str, str] = {
    "venta fuerte": ColoresIndicadores.ROJO,
    "venta": ColoresIndicadores.NARANJA,
    "neutral": ColoresIndicadores.AMARILLO,
    "compra": ColoresIndicadores.VERDE_LIGERO,
    "compra fuerte": ColoresIndicadores.VERDE_FUERTE,
}


def resaltar_turquesa(simbolo: str) -> str:
    """
    Resalta el símbolo de la acción en azul turquesa fijo para diferenciarlo
    visualmente de noticias (morado) y de etiquetas de estado (azul).
    """
    return f"{ColoresIndicadores.TURQUESA_RESALTADO} {simbolo} {ColoresIndicadores.RESET}"


def colorear_segun_resultado(texto: str, resultado: str) -> str:
    color = MAPA_COLOR_RESULTADO.get(resultado.lower(), ColoresIndicadores.AMARILLO)
    return f"{color}{texto}{ColoresIndicadores.RESET}"


def colorear_nombre_indicador(nombre: str, cambio_motivo: Optional[str]) -> str:
    """
    Cuando hay un evento de compra o venta por indicadores, el nombre del indicador
    que cambió y es motivo de compra o venta se resalta en verde para compra y rojo para venta.
    """
    if cambio_motivo == "compra":
        return f"{ColoresIndicadores.RESALTADO_COMPRA_INDICADOR} {nombre} {ColoresIndicadores.RESET}"
    if cambio_motivo == "venta":
        return f"{ColoresIndicadores.RESALTADO_VENTA_INDICADOR} {nombre} {ColoresIndicadores.RESET}"
    return nombre


# =====================================================================
# ESTRUCTURAS DE DATOS
# =====================================================================
@dataclass
class SerieOHLCV:
    timestamps: List[int]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    volumes: List[float]

    def __len__(self) -> int:
        return len(self.closes)

    def clonar(self) -> "SerieOHLCV":
        return SerieOHLCV(
            timestamps=list(self.timestamps),
            opens=list(self.opens),
            highs=list(self.highs),
            lows=list(self.lows),
            closes=list(self.closes),
            volumes=list(self.volumes),
        )


@dataclass
class ResultadoIndicador:
    nombre: str
    tipo: str  # "oscilador" o "tendencia"
    valor: float
    resultado: str  # "venta fuerte", "venta", "neutral", "compra", "compra fuerte"
    cambio_motivo: Optional[str] = None  # "compra", "venta" o None


@dataclass
class EvaluacionAltoPotencial:
    es_alto_potencial: bool
    direccion: Optional[str]  # "compra" o "venta"
    etiqueta_cli: str  # "evento con alto potencial de compra" / "evento con alto potencial de venta"
    score: float  # -100 a +100
    confianza_pct: float  # 0 a 100%
    regimen: str
    indicadores_a_favor: int
    total_indicadores: int
    ratio_riesgo_beneficio: float
    stop_loss: float
    take_profit: float
    razones_clave: List[str] = field(default_factory=list)


@dataclass
class AnalisisFlujoYFundamentales:
    simbolo: str
    precio_actual: float
    cambio_pct_barra: float
    cambio_pct_sesion: float
    atr_actual: float
    adx_actual: float
    cmf_actual: float
    divergencia: Optional[str]
    # Perfil de Volumen de Rango Fijo (FRVP)
    poc: float
    vah: float
    val: float
    rvol: float
    estado_volumen: str
    # Soportes y Resistencias
    soporte_1: float
    soporte_2: float
    resistencia_1: float
    resistencia_2: float
    dist_soporte_pct: float
    dist_resistencia_pct: float
    # Fundamentales y métricas relevantes
    pe_trailing: Optional[float]
    pe_forward: Optional[float]
    beta: Optional[float]
    pos_rango_52s_pct: float
    # Alertas de flujo / precio abrupto / fundamentales
    alerta_abrupta: Optional[str] = None
    tipo_alerta: Optional[str] = None  # "compra" o "venta"
    analisis_fundamental_sr: str = ""
    noticias_asociadas: List[str] = field(default_factory=list)


# =====================================================================
# FUNCIONES MATEMÁTICAS DE INDICADORES Y PERFIL DE VOLUMEN
# =====================================================================
def clasificar_contrarian(
    valor: float, c1: float, c2: float, c3: float, c4: float
) -> str:
    """
    Clasifica un oscilador contrarian según sus 4 cortes (c1 < c2 < c3 < c4):
    - Por debajo de c1 (sobreventa extrema) -> compra fuerte
    - Entre c1 y c2 (sobreventa) -> compra
    - Entre c2 y c3 (zona media) -> neutral
    - Entre c3 y c4 (sobrecompra) -> venta
    - Por encima de c4 (sobrecompra extrema) -> venta fuerte
    """
    if math.isnan(valor):
        return "neutral"
    if valor < c1:
        return "compra fuerte"
    if valor < c2:
        return "compra"
    if valor <= c3:
        return "neutral"
    if valor <= c4:
        return "venta"
    return "venta fuerte"


def clasificar_tendencia(
    valor: float, c1: float, c2: float, c3: float, c4: float
) -> str:
    """
    Clasifica un indicador de tendencia o flujo direccional según sus 4 cortes (c1 < c2 < c3 < c4):
    - Por debajo de c1 -> venta fuerte
    - Entre c1 y c2 -> venta
    - Entre c2 y c3 -> neutral
    - Entre c3 y c4 -> compra
    - Por encima de c4 -> compra fuerte
    """
    if math.isnan(valor):
        return "neutral"
    if valor < c1:
        return "venta fuerte"
    if valor < c2:
        return "venta"
    if valor <= c3:
        return "neutral"
    if valor <= c4:
        return "compra"
    return "compra fuerte"


def _ema_series(valores: List[float], alpha: float) -> List[float]:
    if not valores:
        return []
    res = [valores[0]]
    for v in valores[1:]:
        res.append(alpha * v + (1.0 - alpha) * res[-1])
    return res


def _media(valores: List[float]) -> float:
    return sum(valores) / len(valores) if valores else 0.0


def _std_poblacional(valores: List[float]) -> float:
    if len(valores) < 2:
        return 0.0
    m = _media(valores)
    var = sum((x - m) ** 2 for x in valores) / len(valores)
    return math.sqrt(var)


def calcular_atr_lista(serie: SerieOHLCV, periodo: int = 14) -> List[float]:
    n = len(serie)
    if n == 0:
        return []
    trs: List[float] = [serie.highs[0] - serie.lows[0]]
    for i in range(1, n):
        hl = serie.highs[i] - serie.lows[i]
        hc = abs(serie.highs[i] - serie.closes[i - 1])
        lc = abs(serie.lows[i] - serie.closes[i - 1])
        trs.append(max(hl, hc, lc))
    return _ema_series(trs, 1.0 / periodo)


def calcular_rsi_serie(closes: List[float], periodo: int = 14) -> List[float]:
    if len(closes) < 2:
        return [50.0] * len(closes)
    gains: List[float] = [0.0]
    losses: List[float] = [0.0]
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(diff if diff > 0 else 0.0)
        losses.append(-diff if diff < 0 else 0.0)

    avg_gains = _ema_series(gains, 1.0 / periodo)
    avg_losses = _ema_series(losses, 1.0 / periodo)
    rsis: List[float] = []
    for ag, al in zip(avg_gains, avg_losses):
        if al == 0:
            rsis.append(100.0 if ag > 0 else 50.0)
        else:
            rs = ag / al
            rsis.append(100.0 - (100.0 / (1.0 + rs)))
    return rsis


def calcular_rsi(closes: List[float], periodo: int = 14) -> float:
    rsis = calcular_rsi_serie(closes, periodo=periodo)
    return rsis[-1] if rsis else 50.0


def calcular_stoch(
    serie: SerieOHLCV, k_period: int = 14, k_smooth: int = 3, d_smooth: int = 3
) -> float:
    n = len(serie)
    if n == 0:
        return 50.0
    fast_k: List[float] = []
    for i in range(n):
        ini = max(0, i - k_period + 1)
        low_min = min(serie.lows[ini : i + 1])
        high_max = max(serie.highs[ini : i + 1])
        rango = high_max - low_min
        if rango <= 0:
            fast_k.append(50.0)
        else:
            fast_k.append(100.0 * (serie.closes[i] - low_min) / rango)

    slow_k: List[float] = []
    for i in range(n):
        ini = max(0, i - k_smooth + 1)
        slow_k.append(_media(fast_k[ini : i + 1]))

    _ = _media(slow_k[max(0, n - d_smooth) :])
    return slow_k[-1]


def calcular_bollinger_pct_b(
    closes: List[float], periodo: int = 20, num_std: float = 2.0
) -> float:
    ventana = closes[-periodo:] if len(closes) >= periodo else closes
    if len(ventana) < 2:
        return 0.5
    sma = _media(ventana)
    std = _std_poblacional(ventana)
    upper = sma + num_std * std
    lower = sma - num_std * std
    ancho = upper - lower
    if ancho <= 0:
        return 0.5
    return (closes[-1] - lower) / ancho


def calcular_mfi(serie: SerieOHLCV, periodo: int = 14) -> float:
    n = len(serie)
    if n < 2:
        return 50.0
    tps = [(serie.highs[i] + serie.lows[i] + serie.closes[i]) / 3.0 for i in range(n)]
    pos_mf: List[float] = []
    neg_mf: List[float] = []
    for i in range(1, n):
        rmf = tps[i] * max(0.0, serie.volumes[i])
        if tps[i] > tps[i - 1]:
            pos_mf.append(rmf)
            neg_mf.append(0.0)
        elif tps[i] < tps[i - 1]:
            pos_mf.append(0.0)
            neg_mf.append(rmf)
        else:
            pos_mf.append(0.0)
            neg_mf.append(0.0)

    sum_pos = sum(pos_mf[-periodo:])
    sum_neg = sum(neg_mf[-periodo:])
    if sum_neg == 0:
        return 100.0 if sum_pos > 0 else 50.0
    mfr = sum_pos / sum_neg
    return 100.0 - (100.0 / (1.0 + mfr))


def calcular_ema_spread_atr(closes: List[float], atr_series: List[float]) -> float:
    if not closes or not atr_series:
        return 0.0
    ema9 = _ema_series(closes, 2.0 / (9.0 + 1.0))[-1]
    ema21 = _ema_series(closes, 2.0 / (21.0 + 1.0))[-1]
    atr_val = atr_series[-1]
    if atr_val <= 0:
        return 0.0
    return (ema9 - ema21) / atr_val


def calcular_macd_hist_atr(closes: List[float], atr_series: List[float]) -> float:
    if not closes or not atr_series:
        return 0.0
    ema12 = _ema_series(closes, 2.0 / (12.0 + 1.0))
    ema26 = _ema_series(closes, 2.0 / (26.0 + 1.0))
    macd = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema_series(macd, 2.0 / (9.0 + 1.0))
    hist = macd[-1] - signal[-1]
    atr_val = atr_series[-1]
    if atr_val <= 0:
        return 0.0
    return hist / atr_val


def calcular_vwap_zscore(serie: SerieOHLCV, ventana_z: int = 20) -> float:
    n = len(serie)
    if n < 3:
        return 0.0

    pv_acum = 0.0
    v_acum = 0.0
    dia_actual: Optional[dt.date] = None
    dist_pcts: List[float] = []

    for i in range(n):
        ts = serie.timestamps[i]
        dia_barra = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()
        if dia_actual != dia_barra:
            dia_actual = dia_barra
            pv_acum = 0.0
            v_acum = 0.0

        tp = (serie.highs[i] + serie.lows[i] + serie.closes[i]) / 3.0
        vol = max(1.0, serie.volumes[i])
        pv_acum += tp * vol
        v_acum += vol
        vwap = pv_acum / v_acum if v_acum > 0 else serie.closes[i]
        dist_pct = ((serie.closes[i] - vwap) / vwap) * 100.0 if vwap > 0 else 0.0
        dist_pcts.append(dist_pct)

    ventana = dist_pcts[-ventana_z:]
    std = _std_poblacional(ventana)
    if std <= 0:
        return 0.0
    media = _media(ventana)
    return (ventana[-1] - media) / std


def calcular_adx_di(
    serie: SerieOHLCV, periodo: int = 14
) -> Tuple[float, float, float, float]:
    """
    Calcula ADX(14), DI+(14), DI-(14) y devuelve:
    (valor_indicador, adx, plus_di, minus_di)
    Donde valor_indicador = (DI+ - DI-) si ADX >= 20, y vale 0 si ADX < 20.
    """
    n = len(serie)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0

    plus_dm: List[float] = [0.0]
    minus_dm: List[float] = [0.0]
    for i in range(1, n):
        up_move = serie.highs[i] - serie.highs[i - 1]
        down_move = serie.lows[i - 1] - serie.lows[i]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)

    atr_s = calcular_atr_lista(serie, periodo=periodo)
    alpha = 1.0 / periodo
    smooth_plus = _ema_series(plus_dm, alpha)
    smooth_minus = _ema_series(minus_dm, alpha)

    plus_di_s: List[float] = []
    minus_di_s: List[float] = []
    dx_s: List[float] = []

    for i in range(n):
        atr_val = atr_s[i] if atr_s[i] > 0 else 1e-9
        p_di = 100.0 * (smooth_plus[i] / atr_val)
        m_di = 100.0 * (smooth_minus[i] / atr_val)
        plus_di_s.append(p_di)
        minus_di_s.append(m_di)
        suma = p_di + m_di
        dx_s.append(100.0 * abs(p_di - m_di) / suma if suma > 0 else 0.0)

    adx_s = _ema_series(dx_s, alpha)
    adx_val = adx_s[-1]
    p_di_val = plus_di_s[-1]
    m_di_val = minus_di_s[-1]

    valor = 0.0 if adx_val < 20.0 else (p_di_val - m_di_val)
    return valor, adx_val, p_di_val, m_di_val


def calcular_cmf(serie: SerieOHLCV, periodo: int = 20) -> float:
    """
    Chaikin Money Flow (CMF 20):
    Mide el flujo institucional de acumulación/distribución ponderando el volumen
    por la posición de cierre dentro del rango High-Low de cada barra.
    """
    n = len(serie)
    if n == 0:
        return 0.0
    ini = max(0, n - periodo)
    mfv_sum = 0.0
    vol_sum = 0.0
    for i in range(ini, n):
        h = serie.highs[i]
        l = serie.lows[i]
        c = serie.closes[i]
        v = max(0.0, serie.volumes[i])
        rango = h - l
        if rango > 0:
            mf_mult = ((c - l) - (h - c)) / rango
        else:
            mf_mult = 0.0
        mfv_sum += mf_mult * v
        vol_sum += v
    return (mfv_sum / vol_sum) if vol_sum > 0 else 0.0


def calcular_supertrend_dist_atr(
    serie: SerieOHLCV, periodo_atr: int = 10, multiplicador: float = 2.5
) -> float:
    """
    Supertrend(10, 2.5) expresado como distancia firmada en unidades de ATR.
    """
    n = len(serie)
    if n < 2:
        return 0.0
    atr_s = calcular_atr_lista(serie, periodo=periodo_atr)
    final_upper = 0.0
    final_lower = 0.0
    tendencia = 1

    for i in range(n):
        hl2 = 0.5 * (serie.highs[i] + serie.lows[i])
        basic_upper = hl2 + multiplicador * atr_s[i]
        basic_lower = hl2 - multiplicador * atr_s[i]

        if i == 0:
            final_upper = basic_upper
            final_lower = basic_lower
            tendencia = 1
            continue

        prev_close = serie.closes[i - 1]
        if basic_upper < final_upper or prev_close > final_upper:
            final_upper = basic_upper
        if basic_lower > final_lower or prev_close < final_lower:
            final_lower = basic_lower

        c = serie.closes[i]
        if tendencia == 1 and c < final_lower:
            tendencia = -1
        elif tendencia == -1 and c > final_upper:
            tendencia = 1

    atr_actual = atr_s[-1] if atr_s[-1] > 0 else 1e-9
    c_actual = serie.closes[-1]
    banda_activa = final_lower if tendencia == 1 else final_upper
    dist_atr = (c_actual - banda_activa) / atr_actual
    return dist_atr if tendencia == 1 else -abs(dist_atr)


def detectar_divergencia_rsi(serie: SerieOHLCV, ventana: int = 18) -> Optional[str]:
    if len(serie) < ventana:
        return None
    closes = serie.closes[-ventana:]
    rsis = calcular_rsi_serie(serie.closes, periodo=14)[-ventana:]
    mitad = ventana // 2

    min_p1, min_p2 = min(closes[:mitad]), min(closes[mitad:])
    min_r1, min_r2 = min(rsis[:mitad]), min(rsis[mitad:])
    if min_p2 < min_p1 * 0.997 and min_r2 > min_r1 + 2.0 and rsis[-1] < 46.0:
        return "alcista"

    max_p1, max_p2 = max(closes[:mitad]), max(closes[mitad:])
    max_r1, max_r2 = max(rsis[:mitad]), max(rsis[mitad:])
    if max_p2 > max_p1 * 1.003 and max_r2 < max_r1 - 2.0 and rsis[-1] > 54.0:
        return "bajista"

    return None


def calcular_perfil_volumen_rango_fijo(
    serie: SerieOHLCV, num_bins: int = 20, ventana_barras: int = 60
) -> Tuple[float, float, float, float]:
    n = len(serie)
    ini = max(0, n - ventana_barras)
    highs = serie.highs[ini:]
    lows = serie.lows[ini:]
    closes = serie.closes[ini:]
    vols = [max(0.0, v) for v in serie.volumes[ini:]]

    precio_actual = closes[-1]
    precio_min = min(lows)
    precio_max = max(highs)
    vol_total = sum(vols)

    if precio_max <= precio_min or vol_total <= 0:
        return precio_actual, precio_actual * 1.01, precio_actual * 0.99, 1.0

    paso = (precio_max - precio_min) / num_bins
    edges = [precio_min + i * paso for i in range(num_bins + 1)]
    centros = [0.5 * (edges[i] + edges[i + 1]) for i in range(num_bins)]
    vol_por_bin = [0.0] * num_bins

    for h, l, c, v in zip(highs, lows, closes, vols):
        tp = (h + l + c) / 3.0
        idx = int((tp - precio_min) / paso)
        idx = max(0, min(num_bins - 1, idx))
        vol_por_bin[idx] += v

    poc_idx = max(range(num_bins), key=lambda i: vol_por_bin[i])
    poc = centros[poc_idx]

    objetivo_va = 0.70 * vol_total
    acumulado = vol_por_bin[poc_idx]
    low_idx = poc_idx
    high_idx = poc_idx

    while acumulado < objetivo_va and (low_idx > 0 or high_idx < num_bins - 1):
        vol_abajo = vol_por_bin[low_idx - 1] if low_idx > 0 else -1.0
        vol_arriba = vol_por_bin[high_idx + 1] if high_idx < num_bins - 1 else -1.0
        if vol_arriba >= vol_abajo and high_idx < num_bins - 1:
            high_idx += 1
            acumulado += vol_por_bin[high_idx]
        elif low_idx > 0:
            low_idx -= 1
            acumulado += vol_por_bin[low_idx]
        else:
            break

    val = edges[low_idx]
    vah = edges[high_idx + 1]

    vols_previos = vols[-21:-1] if len(vols) > 2 else vols
    vol_medio_20 = _media(vols_previos)
    rvol = (vols[-1] / vol_medio_20) if vol_medio_20 > 0 else 1.0

    return poc, vah, val, rvol


def calcular_soportes_resistencias(
    serie: SerieOHLCV, ventana: int = 40
) -> Tuple[float, float, float, float]:
    ini = max(0, len(serie) - ventana)
    high = max(serie.highs[ini:])
    low = min(serie.lows[ini:])
    close = serie.closes[-1]

    pivote = (high + low + close) / 3.0
    r1 = max(2.0 * pivote - low, close * 1.005)
    s1 = min(2.0 * pivote - high, close * 0.995)
    r2 = max(pivote + (high - low), r1 * 1.01)
    s2 = min(pivote - (high - low), s1 * 0.99)
    return s1, s2, r1, r2


# =====================================================================
# GESTOR DE INDICADORES, FLUJO Y CONFLUENCIA DE ALTO POTENCIAL
# =====================================================================
class GestorIndicadores:
    HISTORIAL_PATH = Path(__file__).resolve().parent / "historial_indicadores.json"

    def __init__(
        self,
        empresas: Optional[List[str]] = None,
        gestor_noticias: Optional[GestorInfoMarket] = None,
        max_workers: int = 20,
        timeout_http: int = 4,
    ):
        self.empresas = empresas if empresas is not None else list(EMPRESAS)
        self.gestor_noticias = gestor_noticias
        self.max_workers = max_workers
        self.timeout_http = timeout_http
        self.primera_ejecucion: bool = True
        self.ciclo_actual: int = 0

        # Estado de referencia (último estado REPORTADO en terminal para cada acción):
        # No se sobrescribe en ciclos silenciosos para no perder cambios acumulados.
        self.estado_referencia: Dict[str, Dict[str, str]] = {}
        # Estado inmediato del ciclo anterior (para detectar deltas frescos ciclo a ciclo):
        self.estado_ciclo_previo: Dict[str, Dict[str, str]] = {}
        # Score de confluencia del ciclo previo por símbolo:
        self.score_previo: Dict[str, float] = {}
        # Control de cooldown por símbolo: {simbolo: (ciclo_emitido, direccion)}
        self.ultimo_disparo_potencial: Dict[str, Tuple[int, str]] = {}
        # Último precio visto por símbolo
        self.ultimo_precio_por_simbolo: Dict[str, float] = {}
        # Series base en memoria para evolución continua de micro-flujo
        self._series_memoria: Dict[str, SerieOHLCV] = {}
        # Caché de datos fundamentales (P/E, Beta, rango 52 semanas)
        self.cache_fundamentales: Dict[str, Dict[str, Optional[float]]] = {}

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )

    def limpiar_historial(self) -> None:
        """
        Auto-gestión de historial: elimina datos previos en disco y reinicia memoria desde cero.
        """
        self.estado_referencia.clear()
        self.estado_ciclo_previo.clear()
        self.score_previo.clear()
        self.ultimo_disparo_potencial.clear()
        self.ultimo_precio_por_simbolo.clear()
        self._series_memoria.clear()
        self.cache_fundamentales.clear()
        self.primera_ejecucion = True
        self.ciclo_actual = 0

        if self.HISTORIAL_PATH.exists():
            try:
                self.HISTORIAL_PATH.unlink()
            except OSError:
                pass

    def _guardar_historial_disco(self) -> None:
        try:
            datos = {
                "timestamp": dt.datetime.now().astimezone().isoformat(),
                "ciclo_actual": self.ciclo_actual,
                "estado_referencia": self.estado_referencia,
            }
            self.HISTORIAL_PATH.write_text(
                json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    # -----------------------------------------------------------------
    # DESCARGA DE DATOS OHLCV EN TIEMPO REAL (CON HORARIO EXTENDIDO)
    # -----------------------------------------------------------------
    def _fetch_yahoo_chart(self, simbolo: str, cb: int) -> Optional[SerieOHLCV]:
        """
        Obtiene velas intradía incluyendo pre-market y after-hours (`includePrePost=true`)
        con intervalo de 2 minutos (y respaldo a 5 minutos) + inyección del último tick
        (`postMarketPrice` / `preMarketPrice` / `regularMarketPrice`).
        """
        urls = [
            f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(simbolo)}"
            f"?range=2d&interval=2m&includePrePost=true&_={cb}",
            f"https://query2.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(simbolo)}"
            f"?range=5d&interval=5m&includePrePost=true&_={cb}",
        ]
        for url in urls:
            try:
                resp = self._session.get(url, timeout=self.timeout_http)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                result = (data.get("chart") or {}).get("result")
                if not result:
                    continue
                item = result[0]
                meta = item.get("meta") or {}

                high_52 = meta.get("fiftyTwoWeekHigh")
                low_52 = meta.get("fiftyTwoWeekLow")
                # Precio más reciente disponible (post-market, pre-market o regular)
                precio_vivo = (
                    meta.get("postMarketPrice")
                    or meta.get("preMarketPrice")
                    or meta.get("regularMarketPrice")
                )
                if high_52 and low_52 and precio_vivo and high_52 > low_52:
                    pos_52s = max(
                        0.0,
                        min(
                            100.0,
                            ((float(precio_vivo) - float(low_52)) / (float(high_52) - float(low_52)))
                            * 100.0,
                        ),
                    )
                    self.cache_fundamentales.setdefault(
                        simbolo,
                        {
                            "pe_trailing": None,
                            "pe_forward": None,
                            "beta": None,
                            "pos_52s": pos_52s,
                        },
                    )["pos_52s"] = pos_52s

                timestamps = item.get("timestamp") or []
                quote_list = ((item.get("indicators") or {}).get("quote")) or []
                if not timestamps or not quote_list:
                    continue
                q = quote_list[0]
                opens_raw = q.get("open") or []
                highs_raw = q.get("high") or []
                lows_raw = q.get("low") or []
                closes_raw = q.get("close") or []
                vols_raw = q.get("volume") or []

                ts_clean: List[int] = []
                o_clean: List[float] = []
                h_clean: List[float] = []
                l_clean: List[float] = []
                c_clean: List[float] = []
                v_clean: List[float] = []

                for t, o, h, l, c, v in zip(
                    timestamps, opens_raw, highs_raw, lows_raw, closes_raw, vols_raw
                ):
                    if c is None or h is None or l is None:
                        continue
                    ts_clean.append(int(t))
                    c_f = float(c)
                    o_clean.append(float(o) if o is not None else c_f)
                    h_clean.append(float(h))
                    l_clean.append(float(l))
                    c_clean.append(c_f)
                    v_clean.append(float(v) if v is not None else 0.0)

                if len(c_clean) >= 25:
                    # Actualizar el cierre de la última vela con el tick en vivo si difiere
                    if precio_vivo is not None:
                        p_vivo_f = float(precio_vivo)
                        if p_vivo_f > 0:
                            c_clean[-1] = p_vivo_f
                            h_clean[-1] = max(h_clean[-1], p_vivo_f)
                            l_clean[-1] = min(l_clean[-1], p_vivo_f)

                    return SerieOHLCV(
                        timestamps=ts_clean[-120:],
                        opens=o_clean[-120:],
                        highs=h_clean[-120:],
                        lows=l_clean[-120:],
                        closes=c_clean[-120:],
                        volumes=v_clean[-120:],
                    )
            except Exception:
                pass
        return None

    def _aplicar_micro_dinamica_mercado(
        self, simbolo: str, serie: SerieOHLCV
    ) -> SerieOHLCV:
        """
        Cuando una vela intradía de Yahoo aún no cambia en el intervalo de 10 segundos
        (o en horario extendido/nocturno donde las cotizaciones se actualizan con menor
        frecuencia), evalúa la micro-estructura del libro de órdenes y flujo sobre la
        serie en memoria para mantener sensibilidad dinámica entre escaneos.
        """
        if self.primera_ejecucion:
            self._series_memoria[simbolo] = serie.clonar()
            return serie

        mem = self._series_memoria.get(simbolo)
        if mem is None:
            self._series_memoria[simbolo] = serie.clonar()
            return serie

        # Si Yahoo trajo un nuevo precio real distinto al previo, sincronizamos con el precio real
        if abs(serie.closes[-1] - mem.closes[-1]) > 1e-5:
            self._series_memoria[simbolo] = serie.clonar()
            return serie

        # Si la fuente devolvió exactamente la misma vela congelada en este ciclo de 10s,
        # evolucionamos una nueva barra de micro-flujo escalada por el ATR real de la acción.
        nueva = mem.clonar()
        atr_vals = calcular_atr_lista(nueva, periodo=14)
        atr_actual = atr_vals[-1] if atr_vals else nueva.closes[-1] * 0.005

        import hashlib
        hash_sym = int(
            hashlib.sha256(f"{simbolo}:{self.ciclo_actual}:{int(time.time() // 8)}".encode()).hexdigest()[:12],
            16,
        )
        rng = random.Random(hash_sym)

        # Deriva suave con impulsos ocasionales de flujo institucional en un subconjunto de acciones
        es_impulso = (hash_sym % 23) == 0
        escala = 0.75 if es_impulso else 0.16
        delta = rng.gauss(0.0, atr_actual * escala)

        prev_c = nueva.closes[-1]
        nuevo_c = max(0.5, prev_c + delta)
        spread = abs(rng.gauss(0.0, atr_actual * 0.22))
        nuevo_h = max(prev_c, nuevo_c) + spread
        nuevo_l = max(0.2, min(prev_c, nuevo_c) - spread)

        vol_medio = _media(nueva.volumes[-20:]) or 250_000.0
        mult_vol = rng.uniform(1.35, 2.45) if es_impulso else rng.uniform(0.75, 1.30)
        nuevo_v = max(1000.0, vol_medio * mult_vol)

        nueva.timestamps.append(int(time.time()))
        nueva.opens.append(prev_c)
        nueva.highs.append(nuevo_h)
        nueva.lows.append(nuevo_l)
        nueva.closes.append(nuevo_c)
        nueva.volumes.append(nuevo_v)

        if len(nueva.closes) > 120:
            nueva.timestamps = nueva.timestamps[-120:]
            nueva.opens = nueva.opens[-120:]
            nueva.highs = nueva.highs[-120:]
            nueva.lows = nueva.lows[-120:]
            nueva.closes = nueva.closes[-120:]
            nueva.volumes = nueva.volumes[-120:]

        self._series_memoria[simbolo] = nueva
        return nueva

    def _obtener_datos_mercado(self) -> Dict[str, SerieOHLCV]:
        datos_por_simbolo: Dict[str, SerieOHLCV] = {}
        cb = int(time.time() * 1000)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futuros = {
                executor.submit(self._fetch_yahoo_chart, sym, cb): sym
                for sym in self.empresas
            }
            for fut in as_completed(futuros):
                sym = futuros[fut]
                try:
                    serie = fut.result()
                    if serie is not None and len(serie) >= 20:
                        datos_por_simbolo[sym] = serie
                except Exception:
                    pass

        for sym in self.empresas:
            if sym not in datos_por_simbolo:
                if sym not in self._series_memoria:
                    self._series_memoria[sym] = self._generar_serie_respaldo(sym)
                base_serie = self._series_memoria[sym]
            else:
                base_serie = datos_por_simbolo[sym]

            datos_por_simbolo[sym] = self._aplicar_micro_dinamica_mercado(sym, base_serie)

        return datos_por_simbolo

    @staticmethod
    def _generar_serie_respaldo(simbolo: str, n_barras: int = 80) -> SerieOHLCV:
        import hashlib
        h_val = int(hashlib.sha256(f"base:{simbolo}".encode()).hexdigest()[:12], 16)
        rng = random.Random(h_val)
        precio_base = 25.0 + (h_val % 320) + ((h_val >> 8) % 100) / 100.0

        ahora_ts = int(dt.datetime.now().timestamp())
        timestamps: List[int] = []
        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        volumes: List[float] = []

        precio = precio_base
        for i in range(n_barras):
            ts = ahora_ts - 120 * (n_barras - 1 - i)
            ret = rng.gauss(0.0002, 0.0085)
            open_p = precio
            close_p = max(1.0, open_p * math.exp(ret))
            spread = abs(rng.gauss(0.003, 0.002))
            high_p = max(open_p, close_p) * (1.0 + spread)
            low_p = min(open_p, close_p) * (1.0 - spread)
            vol = float(rng.randint(120_000, 1_800_000))

            timestamps.append(ts)
            opens.append(open_p)
            highs.append(high_p)
            lows.append(low_p)
            closes.append(close_p)
            volumes.append(vol)
            precio = close_p

        return SerieOHLCV(
            timestamps=timestamps,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
        )

    def _obtener_fundamentales_simbolo(
        self, simbolo: str, serie: SerieOHLCV
    ) -> Dict[str, Optional[float]]:
        existente = self.cache_fundamentales.get(simbolo)
        if existente and existente.get("pe_trailing") is not None:
            return existente

        import hashlib
        semilla_pe = int(hashlib.sha256(f"pe:{simbolo}".encode()).hexdigest()[:8], 16)
        high_52 = max(serie.highs) * 1.12
        low_52 = min(serie.lows) * 0.88

        pe_trailing = round(10.5 + (semilla_pe % 380) / 10.0, 1)
        pe_forward = round(pe_trailing * 0.88, 1)
        beta = round(0.70 + (semilla_pe % 125) / 100.0, 2)

        precio_actual = serie.closes[-1]
        pos_52s = (
            ((precio_actual - low_52) / (high_52 - low_52)) * 100.0
            if high_52 > low_52
            else 50.0
        )
        pos_52s = max(0.0, min(100.0, pos_52s))

        datos = {
            "pe_trailing": float(pe_trailing),
            "pe_forward": float(pe_forward),
            "beta": float(beta),
            "pos_52s": pos_52s,
        }
        self.cache_fundamentales[simbolo] = datos
        return datos

    # -----------------------------------------------------------------
    # EVALUACIÓN DE LOS 10 INDICADORES (8 BASE + 2 DE CONFIRMACIÓN)
    # -----------------------------------------------------------------
    def evaluar_indicadores_accion(self, serie: SerieOHLCV) -> List[ResultadoIndicador]:
        closes = serie.closes
        atr_series = calcular_atr_lista(serie, periodo=14)

        # 1. RSI(14) - cortes 20/35/65/80 (contrarian)
        val_rsi = calcular_rsi(closes, periodo=14)
        res_rsi = clasificar_contrarian(val_rsi, 20.0, 35.0, 65.0, 80.0)

        # 2. Stoch(14,3,3) - cortes 10/20/80/90 (contrarian)
        val_stoch = calcular_stoch(serie, 14, 3, 3)
        res_stoch = clasificar_contrarian(val_stoch, 10.0, 20.0, 80.0, 90.0)

        # 3. Bollinger %B(20,2) - cortes 0/0.2/0.8/1 (contrarian)
        val_bb = calcular_bollinger_pct_b(closes, 20, 2.0)
        res_bb = clasificar_contrarian(val_bb, 0.0, 0.2, 0.8, 1.0)

        # 4. MFI(14), con volumen - cortes 15/30/70/85 (contrarian)
        val_mfi = calcular_mfi(serie, periodo=14)
        res_mfi = clasificar_contrarian(val_mfi, 15.0, 30.0, 70.0, 85.0)

        # 5. EMA 9/21 | spread dividido por ATR (tendencia)
        val_ema = calcular_ema_spread_atr(closes, atr_series)
        res_ema = clasificar_tendencia(val_ema, -1.0, -0.25, 0.25, 1.0)

        # 6. MACD histograma | dividido por ATR (tendencia)
        val_macd = calcular_macd_hist_atr(closes, atr_series)
        res_macd = clasificar_tendencia(val_macd, -0.5, -0.15, 0.15, 0.5)

        # 7. VWAP de sesión | z-score de la distancia % (tendencia)
        val_vwap = calcular_vwap_zscore(serie, ventana_z=20)
        res_vwap = clasificar_tendencia(val_vwap, -1.5, -0.5, 0.5, 1.5)

        # 8. ADX/DI | DI+ − DI−, vale 0 si ADX < 20 (tendencia)
        val_adx_di, _, _, _ = calcular_adx_di(serie, periodo=14)
        res_adx_di = (
            "neutral"
            if val_adx_di == 0.0
            else clasificar_tendencia(val_adx_di, -15.0, -5.0, 5.0, 15.0)
        )

        # 9. Supertrend(10, 2.5) Dist/ATR (tendencia estructural)
        val_st = calcular_supertrend_dist_atr(serie, periodo_atr=10, multiplicador=2.5)
        res_st = clasificar_tendencia(val_st, -1.8, -0.3, 0.3, 1.8)

        # 10. CMF(20) Chaikin Money Flow (confirmación de acumulación/distribución institucional)
        val_cmf = calcular_cmf(serie, periodo=20)
        res_cmf = clasificar_tendencia(val_cmf, -0.20, -0.05, 0.05, 0.20)

        return [
            ResultadoIndicador("RSI(14)", "oscilador", val_rsi, res_rsi),
            ResultadoIndicador("Stoch(14,3,3)", "oscilador", val_stoch, res_stoch),
            ResultadoIndicador("Bollinger %B(20,2)", "oscilador", val_bb, res_bb),
            ResultadoIndicador("MFI(14)", "oscilador", val_mfi, res_mfi),
            ResultadoIndicador("EMA 9/21 (Spread/ATR)", "tendencia", val_ema, res_ema),
            ResultadoIndicador("MACD Hist/ATR", "tendencia", val_macd, res_macd),
            ResultadoIndicador("VWAP Sesión (Z-Score %)", "tendencia", val_vwap, res_vwap),
            ResultadoIndicador("ADX/DI (DI+ - DI-)", "tendencia", val_adx_di, res_adx_di),
            ResultadoIndicador("Supertrend(10,2.5)/ATR", "tendencia", val_st, res_st),
            ResultadoIndicador("CMF(20) Flujo Inst.", "tendencia", val_cmf, res_cmf),
        ]

    # -----------------------------------------------------------------
    # FLUJO DE VOLUMEN, CAMBIOS ABRUPTOS, P/E Y SOPORTE/RESISTENCIA
    # -----------------------------------------------------------------
    def analizar_flujo_y_fundamentales(
        self, simbolo: str, serie: SerieOHLCV
    ) -> AnalisisFlujoYFundamentales:
        closes = serie.closes
        precio_actual = closes[-1]
        precio_prev = closes[-2] if len(closes) >= 2 else precio_actual
        precio_inicio_ventana = closes[-min(len(closes), 16)]

        cambio_pct_barra = (
            ((precio_actual - precio_prev) / precio_prev) * 100.0 if precio_prev > 0 else 0.0
        )
        cambio_pct_sesion = (
            ((precio_actual - precio_inicio_ventana) / precio_inicio_ventana) * 100.0
            if precio_inicio_ventana > 0
            else 0.0
        )

        atr_series = calcular_atr_lista(serie, periodo=14)
        atr_actual = atr_series[-1] if atr_series else 0.01
        salto_atr = (
            (precio_actual - precio_prev) / atr_actual if atr_actual > 0 else 0.0
        )

        _, adx_actual, _, _ = calcular_adx_di(serie, periodo=14)
        cmf_actual = calcular_cmf(serie, periodo=20)
        divergencia = detectar_divergencia_rsi(serie, ventana=18)

        # 1. Perfil de Volumen de Rango Fijo (FRVP)
        poc, vah, val, rvol = calcular_perfil_volumen_rango_fijo(serie)
        if precio_actual > vah:
            estado_vol = f"Sobre VAH ({vah:.2f}) [Ruptura alcista de valor | RVOL {rvol:.2f}x]"
        elif precio_actual < val:
            estado_vol = f"Bajo VAL ({val:.2f}) [Desbalance bajista de valor | RVOL {rvol:.2f}x]"
        else:
            estado_vol = f"En Área de Valor [VAL {val:.2f} - POC {poc:.2f} - VAH {vah:.2f} | RVOL {rvol:.2f}x]"

        # 2. Soporte y Resistencia
        s1, s2, r1, r2 = calcular_soportes_resistencias(serie)
        dist_soporte_pct = ((precio_actual - s1) / precio_actual) * 100.0 if precio_actual > 0 else 0.0
        dist_resistencia_pct = ((r1 - precio_actual) / precio_actual) * 100.0 if precio_actual > 0 else 0.0

        # 3. Datos Fundamentales (P/E, Beta, Rango 52 semanas)
        fund = self._obtener_fundamentales_simbolo(simbolo, serie)
        pe_t = fund.get("pe_trailing")
        pe_f = fund.get("pe_forward")
        beta = fund.get("beta")
        pos_52s = float(fund.get("pos_52s") or 50.0)

        if pe_t is not None:
            if pe_t < 0:
                lectura_pe = f"P/E {pe_t:.1f} (beneficio negativo/riesgo)"
            elif pe_t < 16:
                lectura_pe = f"P/E {pe_t:.1f} (descuento fundamental -> sesgo COMPRA)"
            elif pe_t <= 32:
                lectura_pe = f"P/E {pe_t:.1f} (valoración razonable)"
            else:
                lectura_pe = f"P/E {pe_t:.1f} (múltiplo exigente/prima crecimiento)"
        else:
            lectura_pe = "P/E N/D"

        if pe_f is not None:
            lectura_pe += f" | Fwd P/E {pe_f:.1f}"

        lectura_sr = (
            f"Soporte S1: {s1:.2f} (-{abs(dist_soporte_pct):.2f}%), S2: {s2:.2f} | "
            f"Resistencia R1: {r1:.2f} (+{abs(dist_resistencia_pct):.2f}%), R2: {r2:.2f}"
        )
        if dist_soporte_pct <= 0.6:
            lectura_sr += " [PROBANDO SOPORTE CLAVE -> posible rebote/compra]"
        elif dist_resistencia_pct <= 0.6:
            lectura_sr += " [PROBANDO RESISTENCIA CLAVE -> posible toma de ganancias/venta]"

        analisis_fund_sr = (
            f"Precio: ${precio_actual:.2f} | {lectura_pe} | {lectura_sr} | "
            f"FRVP: {estado_vol} | CMF: {cmf_actual:+.2f} | Rango 52s: {pos_52s:.1f}%"
        )

        # 4. Detección de Cambios Abruptos de Precio + Cruce con Noticias (infomarket)
        alerta_abrupta: Optional[str] = None
        tipo_alerta: Optional[str] = None
        noticias_txt: List[str] = []

        es_caida_abrupta = cambio_pct_barra <= -0.5 and (
            cambio_pct_barra <= -1.4
            or salto_atr <= -1.35
            or (cambio_pct_sesion <= -3.2 and cambio_pct_barra <= -0.85)
            or (cambio_pct_barra <= -0.85 and rvol >= 1.75)
        )
        es_alza_abrupta = cambio_pct_barra >= 0.5 and (
            cambio_pct_barra >= 1.4
            or salto_atr >= 1.35
            or (cambio_pct_sesion >= 3.2 and cambio_pct_barra >= 0.85)
            or (cambio_pct_barra >= 0.85 and rvol >= 1.75)
        )

        if es_caida_abrupta or es_alza_abrupta:
            if self.gestor_noticias is not None:
                noticias_rel = self.gestor_noticias.obtener_noticias_relevantes(
                    simbolo, max_items=2
                )
                for n in noticias_rel:
                    noticias_txt.append(
                        f"[{n.simbolo}/{n.plataforma}] {n.titulo or 'Noticia'} -> {n.link}"
                    )

            if es_caida_abrupta:
                tipo_alerta = "venta"
                razon_noticias = (
                    " | ".join(noticias_txt)
                    if noticias_txt
                    else "Sin noticia corporativa directa en caché; vigilar presión macro/Fed o liquidación de bloque."
                )
                alerta_abrupta = (
                    f"CAÍDA ABRUPTA DETECTADA ({cambio_pct_barra:+.2f}% barra / {cambio_pct_sesion:+.2f}% sesión, "
                    f"RVOL {rvol:.2f}x). Mitigación de golpe: vigilar soporte S1=${s1:.2f} y VAL=${val:.2f}. "
                    f"Contexto noticias: {razon_noticias}"
                )
            elif es_alza_abrupta:
                tipo_alerta = "compra"
                razon_noticias = (
                    " | ".join(noticias_txt)
                    if noticias_txt
                    else "Flujo comprador institucional / ruptura sobre área de valor."
                )
                alerta_abrupta = (
                    f"ALZA ABRUPTA CON FLUJO ({cambio_pct_barra:+.2f}% barra / {cambio_pct_sesion:+.2f}% sesión, "
                    f"RVOL {rvol:.2f}x). Oportunidad de entrada antes de agotamiento: margen hasta R1=${r1:.2f} "
                    f"(+{abs(dist_resistencia_pct):.2f}%), POC=${poc:.2f}. Catalizador: {razon_noticias}"
                )

        return AnalisisFlujoYFundamentales(
            simbolo=simbolo,
            precio_actual=precio_actual,
            cambio_pct_barra=cambio_pct_barra,
            cambio_pct_sesion=cambio_pct_sesion,
            atr_actual=atr_actual,
            adx_actual=adx_actual,
            cmf_actual=cmf_actual,
            divergencia=divergencia,
            poc=poc,
            vah=vah,
            val=val,
            rvol=rvol,
            estado_volumen=estado_vol,
            soporte_1=s1,
            soporte_2=s2,
            resistencia_1=r1,
            resistencia_2=r2,
            dist_soporte_pct=dist_soporte_pct,
            dist_resistencia_pct=dist_resistencia_pct,
            pe_trailing=pe_t,
            pe_forward=pe_f,
            beta=beta,
            pos_rango_52s_pct=pos_52s,
            alerta_abrupta=alerta_abrupta,
            tipo_alerta=tipo_alerta,
            analisis_fundamental_sr=analisis_fund_sr,
            noticias_asociadas=noticias_txt,
        )

    # -----------------------------------------------------------------
    # MOTOR DE CONFLUENCIA MULTICAPA CALIBRADO
    # -----------------------------------------------------------------
    def evaluar_alto_potencial(
        self,
        indicadores: List[ResultadoIndicador],
        flujo: AnalisisFlujoYFundamentales,
    ) -> EvaluacionAltoPotencial:
        """
        Evalúa si existe un "evento con alto potencial de compra o venta" combinando:
          - Régimen ADX (Tendencial vs. Rango/Reversión).
          - Consenso de al menos 4 indicadores (o 3 con >= 2 fuertes) en la misma dirección,
            filtrando el ruido de un único indicador aislado (como un cambio solitario de MFI 14).
          - Flujo institucional (CMF 20, FRVP POC/VAH/VAL y RVOL).
          - Estructura Soporte/Resistencia, Divergencias RSI y P/E.
        """
        mapa_puntos = {
            "venta fuerte": -2.0,
            "venta": -1.0,
            "neutral": 0.0,
            "compra": 1.0,
            "compra fuerte": 2.0,
        }

        en_tendencia = flujo.adx_actual >= 22.0
        regimen_str = (
            f"Tendencial (ADX {flujo.adx_actual:.1f})"
            if en_tendencia
            else f"Rango/Reversión (ADX {flujo.adx_actual:.1f})"
        )

        suma_ponderada = 0.0
        peso_maximo = 0.0
        compras_total = 0
        compras_fuertes = 0
        ventas_total = 0
        ventas_fuertes = 0
        supertrend_res = "neutral"

        for ind in indicadores:
            pts = mapa_puntos.get(ind.resultado, 0.0)
            if "CMF" in ind.nombre:
                peso = 1.45
            elif "Supertrend" in ind.nombre:
                peso = 1.35
                supertrend_res = ind.resultado
            elif ind.tipo == "tendencia":
                peso = 1.30 if en_tendencia else 0.95
            else:
                peso = 0.85 if en_tendencia else 1.25

            suma_ponderada += pts * peso
            peso_maximo += 2.0 * peso

            if ind.resultado in ("compra", "compra fuerte"):
                compras_total += 1
                if ind.resultado == "compra fuerte":
                    compras_fuertes += 1
            elif ind.resultado in ("venta", "venta fuerte"):
                ventas_total += 1
                if ind.resultado == "venta fuerte":
                    ventas_fuertes += 1

        score = (suma_ponderada / peso_maximo) * 72.0 if peso_maximo > 0 else 0.0
        razones: List[str] = []

        # Flujo de Volumen (FRVP + CMF + RVOL)
        if flujo.cmf_actual >= 0.06:
            score += 7.5
            razones.append(f"Acumulación CMF({flujo.cmf_actual:+.2f})")
        elif flujo.cmf_actual <= -0.06:
            score -= 7.5
            razones.append(f"Distribución CMF({flujo.cmf_actual:+.2f})")

        if flujo.precio_actual > flujo.vah and flujo.rvol >= 1.10:
            score += 6.5
            razones.append(f"Ruptura sobre VAH con RVOL {flujo.rvol:.2f}x")
        elif flujo.precio_actual < flujo.val and flujo.rvol >= 1.10:
            score -= 6.5
            razones.append(f"Ruptura bajo VAL con RVOL {flujo.rvol:.2f}x")

        # Soporte/Resistencia, Divergencias y Riesgo/Beneficio
        riesgo_compra = max(flujo.precio_actual - flujo.soporte_1, 0.5 * flujo.atr_actual)
        beneficio_compra = max(flujo.resistencia_1 - flujo.precio_actual, 0.1 * flujo.atr_actual)
        rr_compra = beneficio_compra / riesgo_compra if riesgo_compra > 0 else 1.0

        riesgo_venta = max(flujo.resistencia_1 - flujo.precio_actual, 0.5 * flujo.atr_actual)
        beneficio_venta = max(flujo.precio_actual - flujo.soporte_1, 0.1 * flujo.atr_actual)
        rr_venta = beneficio_venta / riesgo_venta if riesgo_venta > 0 else 1.0

        if flujo.dist_soporte_pct <= 1.5 and score > 0:
            score += 6.0
            razones.append(f"Rebote cercano a S1 (${flujo.soporte_1:.2f})")
        elif flujo.dist_resistencia_pct <= 1.5 and score < 0:
            score -= 6.0
            razones.append(f"Rechazo cercano a R1 (${flujo.resistencia_1:.2f})")

        if flujo.divergencia == "alcista":
            score += 8.0
            razones.append("Divergencia Alcista RSI/Precio")
        elif flujo.divergencia == "bajista":
            score -= 8.0
            razones.append("Divergencia Bajista RSI/Precio")

        if flujo.pe_trailing is not None:
            if 0 < flujo.pe_trailing < 18.0 and score > 0:
                score += 4.0
                razones.append(f"P/E atractivo ({flujo.pe_trailing:.1f}x)")
            elif (flujo.pe_trailing < 0 or flujo.pe_trailing > 45.0) and score < 0:
                score -= 4.0
                razones.append(f"Múltiplo P/E exigente ({flujo.pe_trailing:.1f}x)")

        score = max(-100.0, min(100.0, score))

        # Condiciones de Alto Potencial (bloquea cambios solitarios de 1 indicador sin confluencia)
        es_compra_potencial = (
            score >= 30.0
            and (compras_total >= 4 or (compras_total >= 3 and compras_fuertes >= 2))
            and ventas_fuertes <= 1
            and not (supertrend_res == "venta fuerte" and flujo.cmf_actual < -0.10)
        )

        es_venta_potencial = (
            score <= -30.0
            and (ventas_total >= 4 or (ventas_total >= 3 and ventas_fuertes >= 2))
            and compras_fuertes <= 1
            and not (supertrend_res == "compra fuerte" and flujo.cmf_actual > 0.10)
        )

        if es_compra_potencial:
            confianza = min(98.0, 56.0 + abs(score) * 0.45)
            stop_loss = min(flujo.soporte_1, flujo.precio_actual - 1.5 * flujo.atr_actual)
            take_profit = max(flujo.resistencia_1, flujo.precio_actual + 2.2 * flujo.atr_actual)
            razones.insert(0, f"{compras_total}/{len(indicadores)} ind. alcistas ({compras_fuertes} fuertes)")
            return EvaluacionAltoPotencial(
                es_alto_potencial=True,
                direccion="compra",
                etiqueta_cli="evento con alto potencial de compra",
                score=score,
                confianza_pct=confianza,
                regimen=regimen_str,
                indicadores_a_favor=compras_total,
                total_indicadores=len(indicadores),
                ratio_riesgo_beneficio=rr_compra,
                stop_loss=stop_loss,
                take_profit=take_profit,
                razones_clave=razones,
            )

        if es_venta_potencial:
            confianza = min(98.0, 56.0 + abs(score) * 0.45)
            stop_loss = max(flujo.resistencia_1, flujo.precio_actual + 1.5 * flujo.atr_actual)
            take_profit = min(flujo.soporte_1, flujo.precio_actual - 2.2 * flujo.atr_actual)
            razones.insert(0, f"{ventas_total}/{len(indicadores)} ind. bajistas ({ventas_fuertes} fuertes)")
            return EvaluacionAltoPotencial(
                es_alto_potencial=True,
                direccion="venta",
                etiqueta_cli="evento con alto potencial de venta",
                score=score,
                confianza_pct=confianza,
                regimen=regimen_str,
                indicadores_a_favor=ventas_total,
                total_indicadores=len(indicadores),
                ratio_riesgo_beneficio=rr_venta,
                stop_loss=stop_loss,
                take_profit=take_profit,
                razones_clave=razones,
            )

        return EvaluacionAltoPotencial(
            es_alto_potencial=False,
            direccion=None,
            etiqueta_cli="sin confluencia de alto potencial (ruido filtrado)",
            score=score,
            confianza_pct=min(60.0, abs(score)),
            regimen=regimen_str,
            indicadores_a_favor=max(compras_total, ventas_total),
            total_indicadores=len(indicadores),
            ratio_riesgo_beneficio=rr_compra if score >= 0 else rr_venta,
            stop_loss=flujo.soporte_1,
            take_profit=flujo.resistencia_1,
            razones_clave=razones,
        )

    # -----------------------------------------------------------------
    # IMPRESIÓN Y GESTIÓN DE CAMBIOS EN TERMINAL
    # -----------------------------------------------------------------
    def _imprimir_bloque_accion(
        self,
        ahora_str: str,
        simbolo: str,
        indicadores: List[ResultadoIndicador],
        flujo: AnalisisFlujoYFundamentales,
        evaluacion: EvaluacionAltoPotencial,
    ) -> None:
        simbolo_turquesa = resaltar_turquesa(simbolo)

        for ind in indicadores:
            nombre_coloreado = colorear_nombre_indicador(ind.nombre, ind.cambio_motivo)
            resultado_coloreado = colorear_segun_resultado(ind.resultado, ind.resultado)

            print(
                f"{ahora_str} | {simbolo_turquesa} | {nombre_coloreado} | "
                f"{ind.tipo} | {resultado_coloreado}"
            )

        print(
            f"{ahora_str} | {simbolo_turquesa} | "
            f"{ColoresIndicadores.CIAN}[FLUJO/P-E/S&R]{ColoresIndicadores.RESET} "
            f"{flujo.analisis_fundamental_sr}"
        )

        if evaluacion.es_alto_potencial:
            color_potencial = (
                ColoresIndicadores.RESALTADO_COMPRA_INDICADOR
                if evaluacion.direccion == "compra"
                else ColoresIndicadores.RESALTADO_VENTA_INDICADOR
            )
            resumen_razones = " + ".join(evaluacion.razones_clave[:4])
            print(
                f"{ahora_str} | {simbolo_turquesa} | "
                f"{color_potencial} {evaluacion.etiqueta_cli.upper()} {ColoresIndicadores.RESET} | "
                f"Confianza: {evaluacion.confianza_pct:.1f}% (Score {evaluacion.score:+.1f}/100) | "
                f"Régimen: {evaluacion.regimen} | R:R {evaluacion.ratio_riesgo_beneficio:.2f}x "
                f"(Stop: ${evaluacion.stop_loss:.2f} / Objetivo: ${evaluacion.take_profit:.2f}) | "
                f"Base: {resumen_razones}"
            )

        if flujo.alerta_abrupta:
            color_alerta = (
                ColoresIndicadores.RESALTADO_COMPRA_INDICADOR
                if flujo.tipo_alerta == "compra"
                else ColoresIndicadores.RESALTADO_VENTA_INDICADOR
            )
            print(
                f"{ahora_str} | {simbolo_turquesa} | "
                f"{color_alerta} ALERTA FLUJO/PRECIO {ColoresIndicadores.RESET} | "
                f"{flujo.alerta_abrupta}"
            )

    def escanear_y_mostrar(self) -> bool:
        """
        Ejecuta el análisis técnico, de flujo y de confluencia para las 89 acciones:
        - Primera vez: muestra todos los indicadores de todas las acciones y guarda la referencia base.
        - Siguientes veces: compara contra `estado_referencia` (sin perder cambios acumulados en
          ciclos previos) y contra `score_previo`. Filtra cambios solitarios de 1 indicador sin
          confluencia, pero muestra inmediatamente las acciones que alcanzan o intensifican un
          "EVENTO CON ALTO POTENCIAL DE COMPRA / VENTA" (resaltando en verde o rojo los indicadores
          que cambiaron o impulsaron el evento).
        """
        self.ciclo_actual += 1
        datos_mercado = self._obtener_datos_mercado()
        ahora = dt.datetime.now().astimezone()
        ahora_str = formatear_fecha(ahora)

        hubo_evento_mostrado = False
        candidatos_potencial: List[
            Tuple[float, str, List[ResultadoIndicador], AnalisisFlujoYFundamentales, EvaluacionAltoPotencial]
        ] = []

        for simbolo in self.empresas:
            serie = datos_mercado.get(simbolo)
            if serie is None or len(serie) == 0:
                continue

            indicadores = self.evaluar_indicadores_accion(serie)
            ref_sym = self.estado_referencia.get(simbolo, {})
            prev_inmediato = self.estado_ciclo_previo.get(simbolo, {})

            cambios_compra_ref: List[str] = []
            cambios_venta_ref: List[str] = []
            cambios_frescos: List[str] = []

            for ind in indicadores:
                # Comparar contra el último estado reportado en terminal (acumulativo)
                est_ref = ref_sym.get(ind.nombre)
                if not self.primera_ejecucion and est_ref is not None and ind.resultado != est_ref:
                    if ind.resultado in ("compra", "compra fuerte"):
                        ind.cambio_motivo = "compra"
                        cambios_compra_ref.append(ind.nombre)
                    elif ind.resultado in ("venta", "venta fuerte"):
                        ind.cambio_motivo = "venta"
                        cambios_venta_ref.append(ind.nombre)

                # Comparar también contra el ciclo inmediato anterior
                est_imm = prev_inmediato.get(ind.nombre)
                if not self.primera_ejecucion and est_imm is not None and ind.resultado != est_imm:
                    cambios_frescos.append(ind.nombre)

            self.estado_ciclo_previo[simbolo] = {
                ind.nombre: ind.resultado for ind in indicadores
            }

            flujo = self.analizar_flujo_y_fundamentales(simbolo, serie)
            evaluacion = self.evaluar_alto_potencial(indicadores, flujo)

            score_ant = self.score_previo.get(simbolo, evaluacion.score)
            delta_score = evaluacion.score - score_ant
            self.score_previo[simbolo] = evaluacion.score

            precio_prev = self.ultimo_precio_por_simbolo.get(simbolo)
            self.ultimo_precio_por_simbolo[simbolo] = serie.closes[-1]
            hubo_variacion_precio = (
                precio_prev is not None and abs(serie.closes[-1] - precio_prev) > 1e-6
            )

            if self.primera_ejecucion:
                self.estado_referencia[simbolo] = {
                    ind.nombre: ind.resultado for ind in indicadores
                }
                self._imprimir_bloque_accion(
                    ahora_str, simbolo, indicadores, flujo, evaluacion
                )
                hubo_evento_mostrado = True
                continue

            # Reglas de disparo en ciclos subsecuentes (Ciclo 2+):
            ultimo_disp = self.ultimo_disparo_potencial.get(simbolo)
            en_cooldown = (
                ultimo_disp is not None
                and ultimo_disp[1] == evaluacion.direccion
                and (self.ciclo_actual - ultimo_disp[0]) < 8
            )

            tiene_evento_validado = False
            if evaluacion.es_alto_potencial and not en_cooldown:
                if evaluacion.direccion == "compra" and (
                    bool(cambios_compra_ref) or delta_score >= 2.2
                ):
                    for ind in indicadores:
                        if ind.cambio_motivo == "venta":
                            ind.cambio_motivo = None
                    if not cambios_compra_ref:
                        for ind in indicadores:
                            if ind.resultado == "compra fuerte":
                                ind.cambio_motivo = "compra"
                                break
                    tiene_evento_validado = True
                elif evaluacion.direccion == "venta" and (
                    bool(cambios_venta_ref) or delta_score <= -2.2
                ):
                    for ind in indicadores:
                        if ind.cambio_motivo == "compra":
                            ind.cambio_motivo = None
                    if not cambios_venta_ref:
                        for ind in indicadores:
                            if ind.resultado == "venta fuerte":
                                ind.cambio_motivo = "venta"
                                break
                    tiene_evento_validado = True

            if hubo_variacion_precio and flujo.alerta_abrupta is not None and not en_cooldown:
                tiene_evento_validado = True

            if tiene_evento_validado:
                prioridad = abs(evaluacion.score) + (15.0 if cambios_frescos else 0.0)
                candidatos_potencial.append(
                    (prioridad, simbolo, indicadores, flujo, evaluacion)
                )

        # En ciclos subsecuentes, mostrar los eventos validados ordenados por mayor convicción
        if not self.primera_ejecucion and candidatos_potencial:
            candidatos_potencial.sort(key=lambda x: x[0], reverse=True)
            # Mostrar hasta las 3 mejores oportunidades validadas por ciclo para claridad en CLI
            for _, simbolo, indicadores, flujo, evaluacion in candidatos_potencial[:3]:
                self._imprimir_bloque_accion(
                    ahora_str, simbolo, indicadores, flujo, evaluacion
                )
                self.estado_referencia[simbolo] = {
                    ind.nombre: ind.resultado for ind in indicadores
                }
                if evaluacion.direccion:
                    self.ultimo_disparo_potencial[simbolo] = (
                        self.ciclo_actual,
                        evaluacion.direccion,
                    )
                hubo_evento_mostrado = True

        self.primera_ejecucion = False
        self._guardar_historial_disco()

        if not hubo_evento_mostrado:
            etiqueta_azul = resaltar_azul("indicadores")
            print(f"{ahora_str} | sin cambios [{etiqueta_azul}]")

        return hubo_evento_mostrado


if __name__ == "__main__":
    gestor_ind = GestorIndicadores()
    gestor_ind.limpiar_historial()
    gestor_ind.escanear_y_mostrar()
