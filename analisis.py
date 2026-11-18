# -*- coding: utf-8 -*-
"""
analisis.py
===========
Motor integral de inteligencia informativa (evolución de `infomarket.py`),
organizado explícitamente en **6 secciones independientes pero correlacionadas**:

1. **NOTICIAS**: Noticias corporativas y sectoriales en tiempo real (RSS + `yfinance`).
2. **FILINGS_8K**: Reportes corrientes SEC Form 8-K (`SEC EDGAR Atom` + `company_tickers.json` + `yfinance.sec_filings` + RSS).
3. **REPORTES_RESULTADOS**: Reportes trimestrales y análisis **conjunto con `scanner.py`**
   (fecha del próximo reporte, ingresos esperados vs ingresos reales reportados, EPS y sorpresa).
4. **DIVULGACIONES_CONGRESO**: Divulgaciones financieras del Congreso de EE. UU. clasificadas en
   las 6 categorías oficiales:
   - Ingresos ganados y no ganados
   - Activos y propiedades
   - Deudas y pasivos
   - Regalos y viajes pagados
   - Cargos externos y acuerdos
   - Finanzas familiares
5. **FED**: Tasas de interés, FOMC, discursos de la Reserva Federal, bonos del Tesoro e inflación (CPI/PCE).
6. **MACROECONOMIA**: Avances industriales/económicos en otros países, tensión geopolítica y movimientos bélicos.
"""

from __future__ import annotations

import csv
import email.utils
import hashlib
import io
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import feedparser
import requests
import yfinance as yf

from config import (
    INTERVALO_CICLO_NOTICIAS_SEG,
    SECCIONES_ANALISIS,
    SUBCATEGORIAS_CONGRESO,
    UMBRAL_LATENCIA_NOTICIAS_ADVERTENCIA,
    UMBRAL_LATENCIA_NOTICIAS_ANORMAL,
    formatear_fecha_hora_ampm,
    formatear_hora_ampm,
    obtener_sectores_ticker,
    simbolo_yfinance,
)
from historial import GestorHistorial

if TYPE_CHECKING:
    from scanner import ScannerMercado


# ==============================================================================
# 1. MODELO DE DATOS, CLASIFICACIÓN DE 6 SECCIONES Y LÉXICO FINANCIERO
# ==============================================================================

@dataclass
class EventoAnalisis:
    """Evento informativo clasificado en una de las 6 secciones de `analisis.py`."""

    id_noticia: str
    epoch_publicacion: float
    fecha_hora_ampm: str
    hora_ampm: str
    ticker: str               # Ticker de cartera (ej. NVDA, MU) o etiqueta institucional/macro
    titulo: str
    resumen: str
    plataforma: str           # Ej. SEC EDGAR, CapitolStockWatch, Federal Reserve, Yahoo Finance, Reuters
    link: str
    sentimiento: str          # "POSITIVA" | "NEGATIVA" | "NEUTRAL"
    impacto_score: float      # 0.0 - 100.0
    es_macro: bool
    categoria_mercado: str = "EMPRESA"
    sectores_relacionados: List[str] = field(default_factory=list)
    epoch_captura: float = 0.0
    seccion: str = "NOTICIAS"               # NOTICIAS | FILINGS_8K | REPORTES_RESULTADOS | DIVULGACIONES_CONGRESO | FED | MACROECONOMIA
    subcategoria: str = ""                  # Ej. categoría Congreso, ítem 8-K, evaluación conjunta scanner, tipo macro
    detalles_extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias de compatibilidad
NoticiaMercado = EventoAnalisis

PALABRAS_POSITIVAS = {
    "beat": 2.2, "beats": 2.2, "surge": 2.4, "surges": 2.4, "soar": 2.5, "soars": 2.5,
    "upgrade": 2.6, "upgraded": 2.6, "outperform": 2.0, "overweight": 1.8, "buy rating": 2.2,
    "record revenue": 2.8, "record profit": 2.8, "raises guidance": 3.0, "raised guidance": 3.0,
    "launch": 1.7, "launches": 1.7, "unveils": 1.9, "new processor": 2.3, "new chip": 2.3,
    "ai chip": 2.2, "breakthrough": 2.4, "partnership": 1.8, "contract": 1.8, "buyback": 2.0,
    "dividend hike": 1.9, "approval": 2.5, "fda approves": 3.0, "bullish": 1.9, "rally": 1.8,
    "rate cut": 2.3, "cooling inflation": 2.2, "soft landing": 1.8, "strong demand": 2.2,
    "memory shortage": 2.1, "hbm demand": 2.3, "ai spending": 2.2, "capex increase": 2.1,
    "ceasefire": 2.2, "trade deal": 2.1, "stimulus": 2.0, "purchase": 1.7, "bought": 1.6,
    "supera": 2.2, "sube": 1.6, "dispara": 2.4, "récord": 2.3, "alcista": 1.9, "compra": 1.7,
}

PALABRAS_NEGATIVAS = {
    "miss": 2.4, "misses": 2.4, "plunge": 2.7, "plunges": 2.7, "tumble": 2.4, "slump": 2.3,
    "downgrade": 2.7, "downgraded": 2.7, "underperform": 2.2, "underweight": 2.0, "sell rating": 2.5,
    "cuts guidance": 3.0, "lowered guidance": 3.0, "warning": 2.3, "profit warning": 3.0,
    "investigation": 2.6, "sec probe": 2.9, "lawsuit": 2.2, "antitrust": 2.3, "subpoena": 2.7,
    "recall": 2.5, "delay": 2.0, "delayed": 2.0, "layoffs": 1.9, "bankruptcy": 3.5, "default": 3.0,
    "tariff": 2.1, "tariffs": 2.1, "export ban": 2.8, "export curb": 2.7, "sanctions": 2.4,
    "war": 2.4, "conflict": 2.1, "escalation": 2.2, "missile": 2.3, "blockade": 2.4, "strike": 1.9,
    "hawkish": 2.1, "hot inflation": 2.4, "rate hike": 2.5, "recession": 2.6, "bearish": 2.0,
    "shortfall": 2.4, "oversupply": 2.3, "chip glut": 2.5, "terminates": 2.1, "resigns": 1.8,
    "cae": 1.7, "desploma": 2.6, "recorta": 2.1, "demanda débil": 2.1, "inflación sube": 2.3, "bajista": 2.0,
}

CATALIZADORES_ALTO_IMPACTO = {
    "earnings", "eps", "quarterly results", "guidance", "revenue", "8-k", "form 8-k", "material agreement",
    "fed", "fomc", "powell", "cpi", "pce", "interest rate", "treasury yield", "inflation",
    "congress", "stock act", "lawmaker", "senator", "representative", "financial disclosure",
    "acquisition", "merger", "buyout", "fda", "processor", "gpu", "semiconductor", "dram", "nand", "hbm",
    "memory", "upgrade", "downgrade", "price target", "contract", "billion", "ai", "data center",
    "tariff", "sanctions", "war", "geopolitical", "military", "middle east", "china", "opec", "oil",
}

REGLAS_CATEGORIA_SECTORES: List[Tuple[str, List[str], Tuple[str, ...]]] = [
    (
        "SEMICONDUCTORES_IA",
        ["SEMICONDUCTORES_IA", "MEMORIA_ALMACENAMIENTO", "OPTICA_RED_DATACENTER"],
        (
            "semiconductor", "chip", "chips", "gpu", "npu", "tpu", "wafer", "foundry", "tsmc",
            "asml", "euv", "dram", "nand", "hbm", "memory", "ram ", "ssd", "sk hynix", "micron",
            "nvidia", "amd", "broadcom", "marvell", "optical transceiver", "data center", "ai infrastructure",
        ),
    ),
    (
        "TASAS_FED",
        ["FINANCIERO", "TECNOLOGIA", "CONSUMO"],
        ("federal reserve", "fomc", "interest rate", "rate cut", "rate hike", "powell", "treasury yield", "monetary policy", "fed "),
    ),
    (
        "INFLACION",
        ["CONSUMO", "FINANCIERO", "TECNOLOGIA"],
        ("cpi", "pce", "inflation", "consumer price", "producer price", "ppi", "jobs report", "payroll"),
    ),
    (
        "GEOPOLITICA",
        ["DEFENSA_AERO_INDUSTRIAL", "ENERGIA_MATERIAS", "SEMICONDUCTORES_IA"],
        ("war", "conflict", "missile", "military", "troops", "naval", "airstrike", "sanctions", "tariff", "tariffs", "trade war", "export ban", "export curb", "geopolitic", "strait", "taiwan", "middle east", "ukraine", "russia", "china trade"),
    ),
    (
        "AVANCES_EXTRANJEROS",
        ["SEMICONDUCTORES_IA", "TECNOLOGIA", "ENERGIA_MATERIAS"],
        ("china ", "europe ", "eu ", "japan ", "korea ", "germany ", "india ", "uk ", "stimulus", "gdp", "pboc", "ecb", "boj", "industrial output", "foreign"),
    ),
    (
        "TECNOLOGIA",
        ["TECNOLOGIA", "OPTICA_RED_DATACENTER"],
        ("cloud", "saas", "cybersecurity", "software", "enterprise ai", "hyperscaler", "big tech", "nasdaq"),
    ),
    (
        "FINANCIERO",
        ["FINANCIERO"],
        ("bank", "banking", "investment bank", "brokerage", "credit", "wall street", "goldman", "jpmorgan", "asset management", "private equity", "fintech"),
    ),
    (
        "SALUD",
        ["SALUD"],
        ("fda", "pharma", "biotech", "clinical trial", "phase 3", "drug", "glp-1", "obesity", "oncology", "medicare", "healthcare"),
    ),
    (
        "CONSUMO",
        ["CONSUMO"],
        ("retail", "consumer spending", "same-store", "walmart", "hospitality", "hotel", "auto sales", "beverage", "apparel"),
    ),
    (
        "ENERGIA_DEFENSA",
        ["ENERGIA_MATERIAS", "DEFENSA_AERO_INDUSTRIAL"],
        ("crude", "oil", "opec", "uranium", "nuclear", "defense", "pentagon", "aerospace", "rare earth", "copper", "steel"),
    ),
]

REGLAS_CONGRESO_6_CATEGORIAS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "Deudas y pasivos",
        ("liabilit", "debt", "mortgage", "loan", "credit line", "promissory", "revolving", "deuda", "pasivo"),
    ),
    (
        "Regalos y viajes pagados",
        ("gift", "travel", "reimbursement", "hospitality", "sponsored trip", "lodging", "airfare", "ethics travel", "regalo", "viaje"),
    ),
    (
        "Cargos externos y acuerdos",
        ("outside position", "board member", "director", "trustee", "agreement", "severance", "future employment", "leave of absence", "cargo externo", "acuerdo"),
    ),
    (
        "Finanzas familiares",
        ("spouse", "dependent", "child", "family", "blind trust", "joint", "cónyuge", "familiar"),
    ),
    (
        "Ingresos ganados y no ganados",
        ("earned income", "unearned income", "dividend", "capital gain", "royalt", "salary", "honoraria", "consulting fee", "interest income", "rental income", "schedule c"),
    ),
    (
        "Activos y propiedades",
        ("purchase", "sale", "exchange", "stock", "equity", "shares", "asset", "property", "real estate", "holding", "bond", "etf", "fund", "ptr", "periodic transaction", "trade", "buy", "sold"),
    ),
]


def clasificar_subcategoria_congreso(texto: str) -> str:
    txt = texto.lower()
    for subcat, kws in REGLAS_CONGRESO_6_CATEGORIAS:
        if any(k in txt for k in kws):
            return subcat
    return SUBCATEGORIAS_CONGRESO[1]  # Default: "Activos y propiedades"


def evaluar_sentimiento_e_impacto(titulo: str, resumen: str = "", es_macro: bool = False) -> Tuple[str, float]:
    """Evalúa el sentimiento (`POSITIVA`, `NEGATIVA`, `NEUTRAL`) y la fuerza catalizadora (0-100)."""
    texto = f"{titulo} {resumen}".lower()
    pts_pos = 0.0
    pts_neg = 0.0

    for palabra, peso in PALABRAS_POSITIVAS.items():
        if palabra in texto:
            pts_pos += peso

    for palabra, peso in PALABRAS_NEGATIVAS.items():
        if palabra in texto:
            pts_neg += peso

    catalizadores = sum(1 for cat in CATALIZADORES_ALTO_IMPACTO if cat in texto)

    neto = pts_pos - pts_neg
    if neto >= 0.9:
        sentimiento = "POSITIVA"
    elif neto <= -0.9:
        sentimiento = "NEGATIVA"
    else:
        if pts_pos > pts_neg and pts_pos > 0:
            sentimiento = "POSITIVA"
        elif pts_neg > pts_pos and pts_neg > 0:
            sentimiento = "NEGATIVA"
        else:
            sentimiento = "NEUTRAL"

    magnitud_sentimiento = abs(pts_pos) + abs(pts_neg)
    base_score = 28.0 + min(42.0, magnitud_sentimiento * 9.5) + min(30.0, catalizadores * 8.5)
    if es_macro:
        base_score = min(100.0, base_score + 6.0)

    return sentimiento, round(min(100.0, base_score), 1)


def clasificar_categoria_y_sectores(
    titulo: str,
    resumen: str,
    etiqueta_defecto: str,
    ticker_asignado: str,
    es_ticker_lista: bool,
) -> Tuple[str, List[str]]:
    """Determina la categoría y sectores de la cartera afectados para propagación sectorial."""
    texto = f" {titulo} {resumen} ".lower()
    sectores_hit: List[str] = []
    categoria = "EMPRESA" if es_ticker_lista else etiqueta_defecto

    for cat_nombre, sectores_asociados, palabras in REGLAS_CATEGORIA_SECTORES:
        if any(p in texto for p in palabras):
            if categoria == etiqueta_defecto or (not es_ticker_lista and categoria in ("MACRO", "FED", "MACROECONOMIA")):
                categoria = cat_nombre
            elif es_ticker_lista and cat_nombre in ("SEMICONDUCTORES_IA", "GEOPOLITICA", "TASAS_FED"):
                categoria = cat_nombre
            for s in sectores_asociados:
                if s not in sectores_hit:
                    sectores_hit.append(s)

    if es_ticker_lista:
        for s_tk in obtener_sectores_ticker(ticker_asignado):
            if s_tk not in sectores_hit:
                sectores_hit.append(s_tk)

    if not sectores_hit:
        mapa_etq = {
            "FED": ["FINANCIERO", "TECNOLOGIA", "CONSUMO"],
            "TASAS_FED": ["FINANCIERO", "TECNOLOGIA", "CONSUMO"],
            "INFLACION": ["CONSUMO", "FINANCIERO", "TECNOLOGIA"],
            "GEOPOLITICA": ["DEFENSA_AERO_INDUSTRIAL", "ENERGIA_MATERIAS", "SEMICONDUCTORES_IA"],
            "MACROECONOMIA": ["DEFENSA_AERO_INDUSTRIAL", "ENERGIA_MATERIAS", "SEMICONDUCTORES_IA", "TECNOLOGIA"],
            "CONGRESO": ["DEFENSA_AERO_INDUSTRIAL", "TECNOLOGIA", "FINANCIERO"],
            "8-K": ["TECNOLOGIA", "FINANCIERO"],
            "RESULTADOS": ["TECNOLOGIA", "SEMICONDUCTORES_IA", "FINANCIERO"],
            "SEMICONDUCTORES_IA": ["SEMICONDUCTORES_IA", "MEMORIA_ALMACENAMIENTO"],
        }
        sectores_hit = mapa_etq.get(etiqueta_defecto, ["MACRO_GLOBAL"])

    return categoria, sectores_hit


# ==============================================================================
# 2. GESTOR PRINCIPAL DE ANÁLISIS EN 6 SECCIONES (`GestorAnalisisMercado`)
# ==============================================================================

class GestorAnalisisMercado:
    """
    Motor de inteligencia en 6 secciones:
    1. NOTICIAS | 2. FILINGS_8K | 3. REPORTES_RESULTADOS (+ Scanner) |
    4. DIVULGACIONES_CONGRESO (6 subcategorías) | 5. FED | 6. MACROECONOMIA.
    """

    # Fuentes organizadas explícitamente por sección
    # Tupla: (seccion, etiqueta_ticker, nombre_fuente, url, es_macro, subcategoria_fija)
    FEEDS_POR_SECCION: List[Tuple[str, str, str, str, bool, str]] = [
        # --- SECCIÓN 1: NOTICIAS (sectoriales/mercado) ---
        (
            "NOTICIAS",
            "SEMICONDUCTORES_IA",
            "Google News Chips/IA/RAM",
            "https://news.google.com/rss/search?q=semiconductor+OR+AI+chips+OR+DRAM+memory+HBM+OR+NVIDIA+OR+TSMC+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Semiconductores/IA",
        ),
        (
            "NOTICIAS",
            "TECNOLOGIA",
            "Google News Tech/Cloud",
            "https://news.google.com/rss/search?q=tech+stocks+cloud+cybersecurity+software+AI+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Tecnología/Cloud",
        ),
        (
            "NOTICIAS",
            "FINANCIERO",
            "Google News Financiero",
            "https://news.google.com/rss/search?q=US+banking+financial+stocks+Wall+Street+credit+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Financiero",
        ),
        (
            "NOTICIAS",
            "SALUD",
            "Google News Salud/Pharma",
            "https://news.google.com/rss/search?q=healthcare+biotech+pharma+FDA+approval+stocks+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Salud/Biotech",
        ),
        (
            "NOTICIAS",
            "CONSUMO",
            "Google News Consumo/Retail",
            "https://news.google.com/rss/search?q=consumer+spending+retail+stocks+Walmart+sales+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Consumo/Retail",
        ),
        # --- SECCIÓN 2: FILINGS 8-K ---
        (
            "FILINGS_8K",
            "8-K",
            "Google News SEC Form 8-K",
            'https://news.google.com/rss/search?q="Form+8-K"+OR+"8-K+filing"+SEC+material+agreement+when:2d&hl=en-US&gl=US&ceid=US:en',
            True,
            "SEC Form 8-K",
        ),
        # --- SECCIÓN 3: REPORTES DE RESULTADOS ---
        (
            "REPORTES_RESULTADOS",
            "RESULTADOS",
            "Google News Resultados",
            "https://news.google.com/rss/search?q=earnings+report+revenue+beats+OR+misses+guidance+Wall+Street+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Earnings/Resultados",
        ),
        # --- SECCIÓN 4: DIVULGACIONES DEL CONGRESO ---
        (
            "DIVULGACIONES_CONGRESO",
            "CONGRESO",
            "Google News Congreso Trades",
            'https://news.google.com/rss/search?q="Congress"+stock+trade+OR+"STOCK+Act"+OR+lawmaker+financial+disclosure+shares+when:3d&hl=en-US&gl=US&ceid=US:en',
            True,
            "Activos y propiedades",
        ),
        (
            "DIVULGACIONES_CONGRESO",
            "CONGRESO",
            "Google News Congreso Ética",
            'https://news.google.com/rss/search?q="House+Ethics"+OR+"Senate+Ethics"+gift+travel+OR+"outside+position"+OR+spouse+income+disclosure+when:4d&hl=en-US&gl=US&ceid=US:en',
            True,
            "",
        ),
        # --- SECCIÓN 5: FED ---
        (
            "FED",
            "TASAS_FED",
            "Federal Reserve Official",
            "https://www.federalreserve.gov/feeds/press_all.xml",
            True,
            "Comunicado Oficial Fed",
        ),
        (
            "FED",
            "TASAS_FED",
            "Fed Monetary Policy",
            "https://www.federalreserve.gov/feeds/press_monetary.xml",
            True,
            "Política Monetaria FOMC",
        ),
        (
            "FED",
            "TASAS_FED",
            "Google News Fed/Tasas",
            "https://news.google.com/rss/search?q=Federal+Reserve+interest+rates+OR+FOMC+OR+Powell+OR+Treasury+yields+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Tasas e Interés",
        ),
        (
            "FED",
            "INFLACION",
            "Google News Inflación CPI/PCE",
            "https://news.google.com/rss/search?q=US+CPI+inflation+OR+PCE+inflation+OR+payroll+Fed+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Inflación / Empleo",
        ),
        # --- SECCIÓN 6: MACROECONOMÍA ---
        (
            "MACROECONOMIA",
            "AVANCES_EXTRANJEROS",
            "Google News Avances Países",
            "https://news.google.com/rss/search?q=(China+OR+Europe+OR+Japan+OR+Taiwan+OR+India)+(stimulus+OR+GDP+OR+semiconductor+breakthrough+OR+industrial+output)+market+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Avances Económicos Otros Países",
        ),
        (
            "MACROECONOMIA",
            "GEOPOLITICA",
            "Google News Tensión Geopolítica",
            "https://news.google.com/rss/search?q=geopolitical+tension+OR+tariffs+OR+sanctions+OR+export+controls+trade+war+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Tensión Geopolítica / Aranceles",
        ),
        (
            "MACROECONOMIA",
            "MOVIMIENTOS_BELICOS",
            "Google News Movimientos Bélicos",
            "https://news.google.com/rss/search?q=military+conflict+OR+war+escalation+OR+missile+strike+OR+Middle+East+oil+shipping+when:2d&hl=en-US&gl=US&ceid=US:en",
            True,
            "Movimientos Bélicos",
        ),
        (
            "MACROECONOMIA",
            "MACRO",
            "CNBC Markets",
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
            True,
            "Mercado Global",
        ),
        (
            "MACROECONOMIA",
            "MACRO",
            "MarketWatch Top",
            "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
            True,
            "Mercado Global",
        ),
        (
            "MACROECONOMIA",
            "MACRO",
            "Yahoo Finance Market",
            "https://finance.yahoo.com/news/rssindex",
            True,
            "Mercado Global",
        ),
    ]

    URL_SEC_ATOM_8K = (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        "?action=getcurrent&CIK=&type=8-K&company=&dateb=&owner=include&start=0&count=80&output=atom"
    )
    URL_SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
    URL_CAPITOL_STOCK_WATCH_CSV = "https://www.capitolstockwatch.com/data/open-data/congress_trades_latest.csv"

    def __init__(
        self,
        tickers: List[str],
        historial: GestorHistorial,
        callback_noticia_relevante: Optional[Callable[[EventoAnalisis], None]] = None,
        scanner: Optional["ScannerMercado"] = None,
    ) -> None:
        self.tickers = [t.strip().upper() for t in tickers if t.strip()]
        self.tickers_set: Set[str] = set(self.tickers)
        self.historial = historial
        self.callback_noticia_relevante = callback_noticia_relevante
        self._scanner: Optional["ScannerMercado"] = scanner

        self._lock = threading.RLock()
        self._ids_vistos: Set[str] = set()
        self._noticias_por_ticker: Dict[str, List[EventoAnalisis]] = {t: [] for t in self.tickers}
        self._noticias_macro: List[EventoAnalisis] = []
        self._noticias_por_sector: Dict[str, List[EventoAnalisis]] = {}
        self._eventos_por_seccion: Dict[str, List[EventoAnalisis]] = {s: [] for s in SECCIONES_ANALISIS}
        self._todas_recientes: List[EventoAnalisis] = []

        # Mapas auxiliares para SEC CIK <-> ticker
        self._cik_a_ticker: Dict[str, str] = {}
        self._ts_cik_mapa: float = 0.0
        self._ts_ultimo_congreso: float = 0.0
        self._ts_ultimo_8k_edgar: float = 0.0

        self._cursor_ticker = 0
        self._cursor_macro = 0
        self._detener = threading.Event()
        self._hilo_monitoreo: Optional[threading.Thread] = None

        self._inicio_monitoreo_epoch: float = time.time()
        self._ultimo_ciclo_epoch: float = 0.0
        self._ultima_noticia_hallada_epoch: float = 0.0
        self._ultima_noticia_obj: Optional[EventoAnalisis] = None
        self._ciclos_completados: int = 0
        self._fuentes_consultadas_ciclo: int = 0
        self._fuentes_exitosas_ciclo: int = 0
        self._errores_red_consecutivos: int = 0
        self._intervalos_hallazgo: List[float] = []

        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*;q=0.8",
        }
        # SEC EDGAR exige User-Agent institucional con contacto explícito
        self._headers_sec = {
            "User-Agent": "VAGY-M1 Research QuantitativeMonitor/2.0 (contact@vagy-m1.local)",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/atom+xml, application/json, text/xml, */*;q=0.8",
        }

        patrones_seguros = [re.escape(t) for t in sorted(self.tickers, key=len, reverse=True) if len(t) >= 2]
        self._regex_tickers = (
            re.compile(r"\b(" + "|".join(patrones_seguros) + r")\b")
            if patrones_seguros
            else None
        )

    def vincular_scanner(self, scanner: "ScannerMercado") -> None:
        """Vincula la instancia de `ScannerMercado` para análisis conjunto de Reportes de Resultados."""
        with self._lock:
            self._scanner = scanner

    # ------------------------------------------------------------------
    # 1. UTILIDADES DE FECHA, DEDUPLICACIÓN Y CLASIFICACIÓN DE SECCIONES
    # ------------------------------------------------------------------

    @staticmethod
    def _generar_id(titulo: str, link: str, extra: str = "") -> str:
        titulo_norm = re.sub(r"\s+", " ", titulo.strip().lower())
        semilla = f"{titulo_norm}|{link.strip().split('?')[0]}|{extra}"
        return hashlib.sha1(semilla.encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def _parsear_fecha_local(valor_fecha: Optional[str]) -> Optional[datetime]:
        if not valor_fecha:
            return None
        texto = str(valor_fecha).strip()
        try:
            dt = email.utils.parsedate_to_datetime(texto)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone()
        except Exception:
            pass
        try:
            iso_limpio = texto.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_limpio)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone()
        except Exception:
            pass
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(texto, fmt).replace(tzinfo=timezone.utc)
                return dt.astimezone()
            except Exception:
                continue
        return None

    @staticmethod
    def _extraer_plataforma(entry: dict, fuente_default: str) -> Tuple[str, str]:
        titulo = str(entry.get("title", "")).strip()
        plataforma = fuente_default
        source_obj = entry.get("source")
        if isinstance(source_obj, dict) and source_obj.get("title"):
            plataforma = str(source_obj["title"]).strip()
        elif " - " in titulo and "Google News" in fuente_default:
            partes = titulo.rsplit(" - ", 1)
            if len(partes) == 2 and 2 <= len(partes[1]) <= 35:
                titulo = partes[0].strip()
                plataforma = partes[1].strip()
        return titulo, plataforma

    def _detectar_ticker_en_texto(self, texto: str, ticker_defecto: str) -> str:
        if ticker_defecto in self.tickers_set:
            return ticker_defecto
        if self._regex_tickers:
            coincidencia = self._regex_tickers.search(texto)
            if coincidencia:
                hallado = coincidencia.group(1)
                if hallado in self.tickers_set:
                    return hallado
        return ticker_defecto

    @staticmethod
    def _inferir_seccion_y_subcategoria(
        seccion_base: str,
        titulo: str,
        resumen: str,
        subcategoria_fija: str = "",
    ) -> Tuple[str, str]:
        """Refina dinámicamente la sección entre las 6 oficiales si un titular de empresa trata sobre 8-K/Resultados/Fed."""
        txt = f" {titulo} {resumen} ".lower()
        sub = subcategoria_fija
        sec = seccion_base if seccion_base in SECCIONES_ANALISIS else "NOTICIAS"

        if sec == "NOTICIAS":
            if any(k in txt for k in ("form 8-k", "8-k filing", "sec item ", "8-k ")):
                return "FILINGS_8K", "SEC Form 8-K"
            if any(k in txt for k in ("earnings", "q1 ", "q2 ", "q3 ", "q4 ", "quarterly results", "eps ", "revenue beat", "revenue miss", "guidance")):
                return "REPORTES_RESULTADOS", "Earnings/Resultados"
            if any(k in txt for k in ("stock act", "congress", "lawmaker", "financial disclosure", "senator stock")):
                return "DIVULGACIONES_CONGRESO", clasificar_subcategoria_congreso(txt)
            if any(k in txt for k in ("fomc", "federal reserve", "powell", "interest rate", "cpi ", "pce ")):
                return "FED", "Reserva Federal / Tasas"

        if sec == "DIVULGACIONES_CONGRESO" and not sub:
            sub = clasificar_subcategoria_congreso(txt)
        elif sec == "MACROECONOMIA" and not sub:
            if any(k in txt for k in ("war", "missile", "military", "airstrike", "naval", "troops", "escalat")):
                sub = "Movimientos Bélicos"
            elif any(k in txt for k in ("tariff", "sanction", "export ban", "geopolitic", "taiwan", "trade war")):
                sub = "Tensión Geopolítica"
            elif any(k in txt for k in ("china", "europe", "japan", "india", "stimulus", "gdp")):
                sub = "Avances Económicos Otros Países"
            else:
                sub = "Macro Global"
        return sec, sub

    # ------------------------------------------------------------------
    # 2. SECCIÓN 1, 5, 6 (y RSS apoyo): DESCARGADOR RSS Y YFINANCE NEWS
    # ------------------------------------------------------------------

    def _enriquecer_evento_con_scanner(self, evento: EventoAnalisis) -> None:
        """Si es Reporte/8-K de una empresa en cartera, adjunta métricas esperadas vs reales de `scanner.py`."""
        scanner = self._scanner
        if scanner is None or evento.ticker not in self.tickers_set:
            return
        if evento.seccion not in ("REPORTES_RESULTADOS", "FILINGS_8K"):
            return
        try:
            rep = scanner.obtener_reporte_resultados(evento.ticker)
            if rep is None:
                return
            evento.detalles_extra.update({
                "fecha_proximo_reporte": rep.fecha_proximo_reporte,
                "ingresos_esperados": rep.ingresos_esperados,
                "ingresos_reportados": rep.ingresos_reportados,
                "eps_esperado": rep.eps_esperado,
                "eps_reportado": rep.eps_reportado,
                "evaluacion_reporte": rep.evaluacion_reporte,
            })
            if rep.resumen_reporte and "Scanner:" not in evento.resumen:
                evento.resumen = f"{evento.resumen} | Scanner: {rep.resumen_reporte}".strip(" |")[:380]
        except Exception:
            pass

    def _descargar_rss(
        self,
        seccion_base: str,
        etiqueta_ticker: str,
        nombre_fuente: str,
        url: str,
        corte_minimo_epoch: float,
        es_macro: bool,
        subcategoria_fija: str = "",
    ) -> Tuple[bool, List[EventoAnalisis]]:
        halladas: List[EventoAnalisis] = []
        try:
            resp = requests.get(url, headers=self._headers, timeout=6.5)
            if resp.status_code != 200 or not resp.content:
                return False, []

            feed = feedparser.parse(resp.content)
            ahora_epoch = time.time()

            for entry in feed.entries[:25]:
                raw_fecha = (
                    entry.get("published")
                    or entry.get("updated")
                    or entry.get("pubDate")
                )
                dt_pub = self._parsear_fecha_local(raw_fecha) or datetime.now().astimezone()
                epoch_pub = dt_pub.timestamp()

                if epoch_pub < corte_minimo_epoch or epoch_pub > ahora_epoch + 600:
                    continue

                link = str(entry.get("link", "")).strip()
                titulo_limpio, plataforma = self._extraer_plataforma(entry, nombre_fuente)
                if not titulo_limpio or not link:
                    continue

                resumen_raw = str(entry.get("summary") or entry.get("description") or "")
                resumen_limpio = re.sub(r"<[^>]+>", " ", resumen_raw)
                resumen_limpio = re.sub(r"\s+", " ", resumen_limpio).strip()[:360]

                ticker_asignado = self._detectar_ticker_en_texto(
                    f"{titulo_limpio} {resumen_limpio}",
                    etiqueta_ticker,
                )
                es_tk_lista = ticker_asignado in self.tickers_set
                es_macro_final = es_macro and not es_tk_lista
                seccion_final, subcat_final = self._inferir_seccion_y_subcategoria(
                    seccion_base, titulo_limpio, resumen_limpio, subcategoria_fija
                )
                sentimiento, impacto = evaluar_sentimiento_e_impacto(
                    titulo_limpio, resumen_limpio, es_macro=es_macro_final
                )
                if seccion_final in ("FED", "MACROECONOMIA", "FILINGS_8K", "REPORTES_RESULTADOS") and impacto < 85.0:
                    impacto = round(min(100.0, impacto + 5.0), 1)
                cat_mkt, sectores_rel = clasificar_categoria_y_sectores(
                    titulo_limpio, resumen_limpio, etiqueta_ticker, ticker_asignado, es_tk_lista
                )

                ev = EventoAnalisis(
                    id_noticia=self._generar_id(titulo_limpio, link),
                    epoch_publicacion=epoch_pub,
                    fecha_hora_ampm=formatear_fecha_hora_ampm(dt_pub),
                    hora_ampm=formatear_hora_ampm(dt_pub),
                    ticker=ticker_asignado,
                    titulo=titulo_limpio,
                    resumen=resumen_limpio,
                    plataforma=plataforma,
                    link=link,
                    sentimiento=sentimiento,
                    impacto_score=impacto,
                    es_macro=es_macro_final,
                    categoria_mercado=cat_mkt,
                    sectores_relacionados=sectores_rel,
                    seccion=seccion_final,
                    subcategoria=subcat_final,
                )
                self._enriquecer_evento_con_scanner(ev)
                halladas.append(ev)

            return True, halladas
        except Exception:
            return False, []

    def _descargar_yfinance_ticker_news(
        self,
        ticker_usuario: str,
        corte_minimo_epoch: float,
    ) -> Tuple[bool, List[EventoAnalisis]]:
        halladas: List[EventoAnalisis] = []
        try:
            simbolo = simbolo_yfinance(ticker_usuario)
            tk = yf.Ticker(simbolo)
            items = tk.news or []
            for raw in items[:12]:
                c = raw.get("content") if isinstance(raw, dict) and isinstance(raw.get("content"), dict) else raw
                if not isinstance(c, dict):
                    continue

                titulo = str(c.get("title") or "").strip()
                resumen = str(c.get("summary") or c.get("description") or "").strip()[:360]

                link = ""
                for clave_url in ("canonicalUrl", "clickThroughUrl"):
                    obj_url = c.get(clave_url)
                    if isinstance(obj_url, dict) and obj_url.get("url"):
                        link = str(obj_url["url"]).strip()
                        break
                if not link and c.get("link"):
                    link = str(c.get("link")).strip()

                if not titulo or not link:
                    continue

                plataforma = "Yahoo Finance"
                prov = c.get("provider")
                if isinstance(prov, dict) and prov.get("displayName"):
                    plataforma = str(prov["displayName"]).strip()
                elif c.get("publisher"):
                    plataforma = str(c["publisher"]).strip()

                dt_pub: Optional[datetime] = None
                if c.get("pubDate"):
                    dt_pub = self._parsear_fecha_local(str(c["pubDate"]))
                elif c.get("providerPublishTime"):
                    try:
                        dt_pub = datetime.fromtimestamp(float(c["providerPublishTime"]), tz=timezone.utc).astimezone()
                    except Exception:
                        dt_pub = None

                dt_pub = dt_pub or datetime.now().astimezone()
                epoch_pub = dt_pub.timestamp()
                if epoch_pub < corte_minimo_epoch:
                    continue

                sentimiento, impacto = evaluar_sentimiento_e_impacto(titulo, resumen, es_macro=False)
                seccion_final, subcat_final = self._inferir_seccion_y_subcategoria("NOTICIAS", titulo, resumen)
                cat_mkt, sectores_rel = clasificar_categoria_y_sectores(
                    titulo, resumen, "EMPRESA", ticker_usuario, True
                )
                ev = EventoAnalisis(
                    id_noticia=self._generar_id(titulo, link),
                    epoch_publicacion=epoch_pub,
                    fecha_hora_ampm=formatear_fecha_hora_ampm(dt_pub),
                    hora_ampm=formatear_hora_ampm(dt_pub),
                    ticker=ticker_usuario,
                    titulo=titulo,
                    resumen=resumen,
                    plataforma=plataforma,
                    link=link,
                    sentimiento=sentimiento,
                    impacto_score=impacto,
                    es_macro=False,
                    categoria_mercado=cat_mkt,
                    sectores_relacionados=sectores_rel,
                    seccion=seccion_final,
                    subcategoria=subcat_final,
                )
                self._enriquecer_evento_con_scanner(ev)
                halladas.append(ev)
            return True, halladas
        except Exception:
            return False, []

    # ------------------------------------------------------------------
    # 3. SECCIÓN 2: SEC FILINGS 8-K (EDGAR ATOM + YFINANCE SEC_FILINGS)
    # ------------------------------------------------------------------

    def _asegurar_mapa_cik_sec(self) -> None:
        ahora = time.time()
        with self._lock:
            if self._cik_a_ticker and (ahora - self._ts_cik_mapa) < 86400:
                return
        try:
            resp = requests.get(self.URL_SEC_COMPANY_TICKERS, headers=self._headers_sec, timeout=8.0)
            if resp.status_code != 200:
                return
            data = resp.json()
            mapa: Dict[str, str] = {}
            for entry in data.values():
                tk_raw = str(entry.get("ticker", "")).strip().upper()
                cik_int = entry.get("cik_str")
                if tk_raw and cik_int is not None:
                    cik_pad = str(int(cik_int)).zfill(10)
                    # Normalizar BRK-B / BF-B frente a BRK.B
                    tk_norm = tk_raw.replace("-", ".") if f"{tk_raw[:3]}.{tk_raw[-1]}" in self.tickers_set else tk_raw
                    mapa[cik_pad] = tk_norm
            with self._lock:
                self._cik_a_ticker = mapa
                self._ts_cik_mapa = time.time()
        except Exception:
            pass

    @staticmethod
    def _clasificar_items_8k(texto: str) -> Tuple[str, str, float]:
        """Clasifica ítems SEC 8-K (1.01, 2.02, 5.02, 7.01, 8.01) en impacto y sentimiento."""
        t = texto.lower()
        sent, imp = evaluar_sentimiento_e_impacto(texto, "", es_macro=False)
        if "2.02" in t or "results of operations" in t:
            return sent, "Item 2.02 Resultados Operativos", max(imp, 72.0)
        if "1.01" in t or "material definitive agreement" in t:
            return "POSITIVA" if sent == "NEUTRAL" else sent, "Item 1.01 Contrato Material", max(imp, 70.0)
        if "1.02" in t or "termination of a material" in t:
            return "NEGATIVA", "Item 1.02 Terminación Acuerdo", max(imp, 72.0)
        if "5.02" in t or "departure of directors" in t:
            return sent, "Item 5.02 Cambio Directivo", max(imp, 64.0)
        if "7.01" in t or "regulation fd" in t:
            return sent, "Item 7.01 Divulgación Reg FD", max(imp, 62.0)
        return sent, "SEC Form 8-K", max(imp, 58.0)

    def _descargar_filings_8k_edgar(self, corte_minimo_epoch: float) -> Tuple[bool, List[EventoAnalisis]]:
        """Descarga el feed en vivo SEC EDGAR de Form 8-K y filtra los relevantes para la lista de seguimiento."""
        self._asegurar_mapa_cik_sec()
        halladas: List[EventoAnalisis] = []
        try:
            resp = requests.get(self.URL_SEC_ATOM_8K, headers=self._headers_sec, timeout=8.0)
            if resp.status_code != 200 or not resp.content:
                return False, []
            feed = feedparser.parse(resp.content)
            with self._lock:
                cik_map = dict(self._cik_a_ticker)

            for entry in feed.entries[:60]:
                titulo_raw = str(entry.get("title", "")).strip()
                link = str(entry.get("link", "")).strip()
                resumen_raw = str(entry.get("summary", "") or "")
                resumen_limpio = re.sub(r"<[^>]+>", " ", resumen_raw)
                resumen_limpio = re.sub(r"\s+", " ", resumen_limpio).strip()[:340]

                dt_pub = self._parsear_fecha_local(entry.get("updated") or entry.get("published")) or datetime.now().astimezone()
                epoch_pub = dt_pub.timestamp()
                if epoch_pub < corte_minimo_epoch:
                    continue

                # Intentar mapear CIK del título `(0001045810)` al ticker en cartera
                tk_hallado = "8-K"
                m_cik = re.search(r"\((\d{10})\)", titulo_raw)
                if m_cik and m_cik.group(1) in cik_map:
                    cand = cik_map[m_cik.group(1)]
                    if cand in self.tickers_set:
                        tk_hallado = cand
                if tk_hallado == "8-K":
                    tk_hallado = self._detectar_ticker_en_texto(f"{titulo_raw} {resumen_limpio}", "8-K")

                es_tk_lista = tk_hallado in self.tickers_set
                sent, sub_item, impacto = self._clasificar_items_8k(f"{titulo_raw} {resumen_limpio}")
                # Filtrar ruido: si no es de la lista, exigir ítem material (2.02, 1.01, 1.02, 5.02)
                if not es_tk_lista and sub_item == "SEC Form 8-K":
                    continue

                _, sectores_rel = clasificar_categoria_y_sectores(
                    titulo_raw, resumen_limpio, "8-K", tk_hallado, es_tk_lista
                )
                ev = EventoAnalisis(
                    id_noticia=self._generar_id(titulo_raw, link, "SEC8K"),
                    epoch_publicacion=epoch_pub,
                    fecha_hora_ampm=formatear_fecha_hora_ampm(dt_pub),
                    hora_ampm=formatear_hora_ampm(dt_pub),
                    ticker=tk_hallado,
                    titulo=titulo_raw[:220],
                    resumen=f"[{sub_item}] {resumen_limpio}"[:360],
                    plataforma="SEC EDGAR 8-K",
                    link=link,
                    sentimiento=sent,
                    impacto_score=impacto,
                    es_macro=not es_tk_lista,
                    categoria_mercado="FILING_8K",
                    sectores_relacionados=sectores_rel,
                    seccion="FILINGS_8K",
                    subcategoria=sub_item,
                )
                self._enriquecer_evento_con_scanner(ev)
                halladas.append(ev)

            return True, halladas
        except Exception:
            return False, []

    def _descargar_yfinance_sec_8k(self, ticker_usuario: str, dias_ventana: int = 7) -> Tuple[bool, List[EventoAnalisis]]:
        """Consulta `tk.sec_filings` de Yahoo Finance para capturar 8-K recientes de empresas vigiladas."""
        halladas: List[EventoAnalisis] = []
        corte_date = (datetime.now() - timedelta(days=dias_ventana)).date()
        try:
            sym = simbolo_yfinance(ticker_usuario)
            filings = yf.Ticker(sym).sec_filings or []
            for f in filings[:10]:
                tipo = str(f.get("type", "")).upper()
                if "8-K" not in tipo:
                    continue
                fecha_raw = f.get("date")
                try:
                    d_f = fecha_raw if hasattr(fecha_raw, "year") else datetime.fromisoformat(str(fecha_raw)).date()
                except Exception:
                    continue
                if d_f < corte_date:
                    continue

                titulo = str(f.get("title") or f"{ticker_usuario} Form 8-K").strip()
                url_edgar = str(f.get("edgarUrl") or f"https://www.sec.gov/edgar/searchedgar/companysearch?q={ticker_usuario}").strip()
                exhibits = f.get("exhibits") if isinstance(f.get("exhibits"), dict) else {}
                res_ex = ", ".join(list(exhibits.keys())[:4])
                dt_pub = datetime.combine(d_f, datetime.min.time(), tzinfo=timezone.utc).astimezone()
                sent, sub_item, impacto = self._clasificar_items_8k(f"{titulo} {res_ex}")
                _, sectores_rel = clasificar_categoria_y_sectores(titulo, res_ex, "8-K", ticker_usuario, True)

                ev = EventoAnalisis(
                    id_noticia=self._generar_id(f"{ticker_usuario}_{d_f}_{titulo}", url_edgar, "YF8K"),
                    epoch_publicacion=dt_pub.timestamp(),
                    fecha_hora_ampm=formatear_fecha_hora_ampm(dt_pub),
                    hora_ampm=formatear_hora_ampm(dt_pub),
                    ticker=ticker_usuario,
                    titulo=f"Filing {tipo}: {titulo}"[:220],
                    resumen=f"[{sub_item}] Anexos: {res_ex or 'N/D'}",
                    plataforma="SEC 8-K (Yahoo)",
                    link=url_edgar,
                    sentimiento=sent,
                    impacto_score=impacto,
                    es_macro=False,
                    categoria_mercado="FILING_8K",
                    sectores_relacionados=sectores_rel,
                    seccion="FILINGS_8K",
                    subcategoria=sub_item,
                )
                self._enriquecer_evento_con_scanner(ev)
                halladas.append(ev)
            return True, halladas
        except Exception:
            return False, []

    # ------------------------------------------------------------------
    # 4. SECCIÓN 3: REPORTES DE RESULTADOS + ANÁLISIS CONJUNTO CON SCANNER
    # ------------------------------------------------------------------

    def evaluar_reportes_con_scanner(self, scanner: Optional["ScannerMercado"] = None) -> List[EventoAnalisis]:
        """
        Evaluación conjunta entre `analisis.py` y `scanner.py`:
        Combina fecha del próximo reporte, ingresos esperados, ingresos reales/reportados,
        EPS esperado/reportado y la estructura técnica del escáner (`cambio_pct`, `RVOL`, `patrones`).
        Genera eventos deduplicados en la sección `REPORTES_RESULTADOS`.
        """
        sc = scanner or self._scanner
        if sc is None:
            return []

        reportes = sc.obtener_reportes_resultados_todos()
        snapshots = sc.obtener_snapshots_todos()
        if not reportes:
            return []

        from scanner import formatear_monto_compacto

        nuevos_o_relevantes: List[EventoAnalisis] = []
        ahora = time.time()
        dt_now = datetime.now().astimezone()

        for tk, rep in reportes.items():
            snap = snapshots.get(tk)
            # Emitir análisis conjunto si:
            # 1) Próximo reporte dentro de 14 días (catalizador pre-earnings), o
            # 2) El último reporte fue hace <= 10 días, o
            # 3) Muestra crecimiento/sorpresa material con respaldo técnico
            dias_ult: Optional[int] = None
            if rep.fecha_ultimo_reporte:
                try:
                    d_ult = datetime.fromisoformat(rep.fecha_ultimo_reporte).date()
                    dias_ult = (dt_now.date() - d_ult).days
                except Exception:
                    dias_ult = None

            es_prox = rep.dias_hasta_reporte is not None and 0 <= rep.dias_hasta_reporte <= 14
            es_reciente = dias_ult is not None and 0 <= dias_ult <= 12
            sorp_material = rep.eps_sorpresa_pct is not None and abs(rep.eps_sorpresa_pct) >= 5.0

            if not (es_prox or es_reciente or (sorp_material and snap is not None and snap.es_evento_alto_potencial)):
                continue

            ing_esp_txt = formatear_monto_compacto(rep.ingresos_esperados)
            ing_rep_txt = formatear_monto_compacto(rep.ingresos_reportados)
            yoy_txt = f"{rep.ingresos_crecimiento_yoy_pct:+.1f}% YoY" if rep.ingresos_crecimiento_yoy_pct is not None else "YoY N/D"

            if rep.evaluacion_reporte == "POSITIVO":
                sentimiento = "POSITIVA"
                impacto = 78.0
            elif rep.evaluacion_reporte == "NEGATIVO":
                sentimiento = "NEGATIVA"
                impacto = 78.0
            elif rep.evaluacion_reporte == "PROXIMO_INMINENTE":
                sentimiento = "POSITIVA" if (snap and snap.direccion_potencial == "ALCISTA") else "NEUTRAL"
                impacto = 72.0
            else:
                sentimiento = "NEUTRAL"
                impacto = 62.0

            # Incorporar verificación técnica de scanner.py en análisis conjunto
            tech_txt = "Scanner pendiente"
            if snap is not None:
                dfin = snap.datos_financieros
                pat_txt = f" | Patrones: {', '.join(snap.patrones_detectados[:2])}" if snap.patrones_detectados else ""
                tech_txt = (
                    f"Precio ${dfin.precio_actual:.2f} ({dfin.cambio_diario_pct:+.2f}%) | "
                    f"Eval Técnica: {snap.evaluacion_general} ({snap.certeza_pct:.0f}%){pat_txt}"
                )
                if (sentimiento == "POSITIVA" and snap.direccion_potencial == "ALCISTA") or (
                    sentimiento == "NEGATIVA" and snap.direccion_potencial == "BAJISTA"
                ):
                    impacto = min(98.0, impacto + 12.0)

            subcat = "Próx. Reporte Inminente" if es_prox else f"Resultados ({rep.evaluacion_reporte})"
            if es_prox and rep.fecha_proximo_reporte:
                titulo = (
                    f"{tk} Reporte en {rep.dias_hasta_reporte}d ({rep.fecha_proximo_reporte}) | "
                    f"Ing Esp: {ing_esp_txt} vs Ult Rep: {ing_rep_txt} ({yoy_txt})"
                )
            else:
                eps_txt = (
                    f" | EPS {rep.eps_reportado:.2f} vs {rep.eps_esperado:.2f} ({rep.eps_sorpresa_pct:+.1f}%)"
                    if (rep.eps_reportado is not None and rep.eps_esperado is not None and rep.eps_sorpresa_pct is not None)
                    else ""
                )
                titulo = f"{tk} Resultados [{rep.evaluacion_reporte}]: Ing Rep {ing_rep_txt} ({yoy_txt}) / Esp {ing_esp_txt}{eps_txt}"

            clave_unico = f"{tk}|{rep.fecha_proximo_reporte}|{rep.fecha_ultimo_reporte}|{rep.evaluacion_reporte}"
            ev = EventoAnalisis(
                id_noticia=self._generar_id(clave_unico, "scanner://reportes", "JOINT"),
                epoch_publicacion=ahora,
                fecha_hora_ampm=formatear_fecha_hora_ampm(dt_now),
                hora_ampm=formatear_hora_ampm(dt_now),
                ticker=tk,
                titulo=titulo[:220],
                resumen=f"Evaluación Conjunta Scanner+Análisis: {rep.resumen_reporte} | {tech_txt}"[:380],
                plataforma="Scanner + Análisis",
                link=f"https://finance.yahoo.com/quote/{tk}/financials",
                sentimiento=sentimiento,
                impacto_score=round(impacto, 1),
                es_macro=False,
                categoria_mercado="RESULTADOS_CONJUNTO",
                sectores_relacionados=obtener_sectores_ticker(tk),
                seccion="REPORTES_RESULTADOS",
                subcategoria=subcat,
                detalles_extra=rep.to_dict(),
            )
            nuevos_o_relevantes.append(ev)

        if nuevos_o_relevantes:
            self._registrar_lote_noticias(nuevos_o_relevantes, notificar_callbacks=bool(self._ciclos_completados > 0))
        return nuevos_o_relevantes

    # ------------------------------------------------------------------
    # 5. SECCIÓN 4: DIVULGACIONES DEL CONGRESO (6 CATEGORÍAS OFICIALES)
    # ------------------------------------------------------------------

    def _descargar_divulgaciones_congreso(self, dias_ventana: int = 10) -> Tuple[bool, List[EventoAnalisis]]:
        """
        Descarga divulgaciones reales del Congreso de EE. UU. desde:
        1) CapitolStockWatch open-data CSV (transacciones PTR vinculadas a tickers de cartera).
        2) House Clerk Disclosure XML ({año}FD.xml) cubriendo todas las 6 categorías oficiales.
        """
        halladas: List[EventoAnalisis] = []
        hubo_ok = False
        corte_date = (datetime.now() - timedelta(days=dias_ventana)).date()

        # Fuente A: CapitolStockWatch Open Data CSV
        try:
            resp = requests.get(self.URL_CAPITOL_STOCK_WATCH_CSV, headers=self._headers, timeout=9.0)
            if resp.status_code == 200 and resp.text:
                hubo_ok = True
                reader = csv.DictReader(io.StringIO(resp.text))
                filas_validas: List[Tuple[datetime.date, dict]] = []
                for row in reader:
                    f_raw = (row.get("filing_date") or row.get("transaction_date") or "").strip()
                    try:
                        d_fil = datetime.strptime(f_raw[:10], "%Y-%m-%d").date()
                    except Exception:
                        continue
                    if d_fil < corte_date:
                        continue
                    filas_validas.append((d_fil, row))

                # Priorizar los que coincidan con nuestra lista de tickers, y limitar macro a top recientes
                filas_validas.sort(key=lambda par: par[0], reverse=True)
                macro_count = 0
                for d_fil, row in filas_validas:
                    tk_raw = str(row.get("ticker", "")).strip().upper()
                    if not tk_raw or tk_raw in ("--", "N/A"):
                        tk_asignado = "CONGRESO"
                        es_tk_lista = False
                    else:
                        es_tk_lista = tk_raw in self.tickers_set
                        tk_asignado = tk_raw if es_tk_lista else "CONGRESO"

                    if not es_tk_lista:
                        if macro_count >= 12:
                            continue
                        macro_count += 1

                    pol = str(row.get("politician") or row.get("representative") or "Legislador").strip()
                    partido = str(row.get("party") or "").strip()
                    camara = str(row.get("chamber") or "Congreso").strip()
                    tx_type = str(row.get("tx_type") or row.get("type") or "").strip()
                    amount = str(row.get("amount") or "").strip()
                    owner = str(row.get("owner") or "").strip()
                    doc_id = str(row.get("doc_id") or "").strip()

                    tx_low = tx_type.lower()
                    if "purch" in tx_low or "buy" in tx_low:
                        sentimiento = "POSITIVA"
                    elif "sale" in tx_low or "sell" in tx_low:
                        sentimiento = "NEGATIVA"
                    else:
                        sentimiento = "NEUTRAL"

                    impacto = 74.0 if ("100,001" in amount or "250,001" in amount or "500,001" in amount or "1,000,001" in amount) else 62.0
                    subcat = SUBCATEGORIAS_CONGRESO[5] if any(w in owner.lower() for w in ("spouse", "child", "dependent", "joint")) else SUBCATEGORIAS_CONGRESO[1]
                    titulo = f"Congreso ({camara} {partido}) {pol}: {tx_type} {tk_raw or 'activo'} [{amount}]"
                    resumen = f"Divulgación STOCK Act ({subcat}) | Fecha registro: {d_fil.isoformat()} | Propietario: {owner or 'Directo'}"
                    url_src = f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{d_fil.year}/{doc_id}.pdf" if doc_id else self.URL_CAPITOL_STOCK_WATCH_CSV
                    dt_pub = datetime.combine(d_fil, datetime.min.time(), tzinfo=timezone.utc).astimezone()
                    _, sectores_rel = clasificar_categoria_y_sectores(titulo, resumen, "CONGRESO", tk_asignado, es_tk_lista)

                    halladas.append(EventoAnalisis(
                        id_noticia=self._generar_id(f"{pol}|{tk_raw}|{tx_type}|{d_fil}|{amount}|{doc_id}", url_src, "CAPITOL"),
                        epoch_publicacion=dt_pub.timestamp(),
                        fecha_hora_ampm=formatear_fecha_hora_ampm(dt_pub),
                        hora_ampm=formatear_hora_ampm(dt_pub),
                        ticker=tk_asignado,
                        titulo=titulo[:220],
                        resumen=resumen[:360],
                        plataforma="CapitolStockWatch (Congreso)",
                        link=url_src,
                        sentimiento=sentimiento,
                        impacto_score=impacto,
                        es_macro=not es_tk_lista,
                        categoria_mercado="DIVULGACION_CONGRESO",
                        sectores_relacionados=sectores_rel,
                        seccion="DIVULGACIONES_CONGRESO",
                        subcategoria=subcat,
                        detalles_extra={"politico": pol, "monto": amount, "tx_type": tx_type, "ticker_reportado": tk_raw},
                    ))
        except Exception:
            pass

        # Fuente B: US House Clerk Official FD XML (cubre FD anual: ingresos, pasivos, regalos, cargos externos, familia)
        try:
            anio = datetime.now().year
            url_zip = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{anio}FD.zip"
            resp_z = requests.get(url_zip, headers=self._headers, timeout=10.0)
            if resp_z.status_code == 200 and resp_z.content:
                hubo_ok = True
                zf = zipfile.ZipFile(io.BytesIO(resp_z.content))
                xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
                if xml_name:
                    root = ET.fromstring(zf.read(xml_name))
                    mapa_ftype = {
                        "O": ("Ingresos ganados y no ganados", "FD Anual Oficial (Ingresos/Activos/Deudas/Cargos)"),
                        "A": ("Activos y propiedades", "Enmienda Declaración Financiera"),
                        "N": ("Activos y propiedades", "FD Nuevo Candidato/Miembro"),
                        "T": ("Finanzas familiares", "FD Terminación/Cierre"),
                        "H": ("Cargos externos y acuerdos", "Reporte Acuerdos/Cargos Externos"),
                        "D": ("Deudas y pasivos", "Declaración Duplicado/Pasivos"),
                        "W": ("Deudas y pasivos", "Extensión/Solicitud"),
                        "X": ("Finanzas familiares", "Exención/Familiar"),
                    }
                    cont_fd = 0
                    for mem in reversed(list(root.findall(".//Member"))):
                        fdate = (mem.findtext("FilingDate") or "").strip()
                        ftype = (mem.findtext("FilingType") or "").strip().upper()
                        # Ignorar PTR 'P' aquí si ya vienen detallados por ticker en CapitolStockWatch
                        if ftype == "P":
                            continue
                        try:
                            d_f = datetime.strptime(fdate, "%m/%d/%Y").date()
                        except Exception:
                            continue
                        if d_f < corte_date:
                            continue

                        nombre_pol = f"{mem.findtext('Prefix', '')} {mem.findtext('First', '')} {mem.findtext('Last', '')}".strip()
                        nombre_pol = re.sub(r"\s+", " ", nombre_pol)
                        dist = (mem.findtext("StateDst") or "").strip()
                        doc_id = (mem.findtext("DocID") or "").strip()
                        subcat, desc_tipo = mapa_ftype.get(ftype, ("Ingresos ganados y no ganados", f"FD Tipo {ftype}"))
                        dt_pub = datetime.combine(d_f, datetime.min.time(), tzinfo=timezone.utc).astimezone()
                        pdf_url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{anio}/{doc_id}.pdf" if doc_id else url_zip

                        halladas.append(EventoAnalisis(
                            id_noticia=self._generar_id(f"HOUSEFD|{doc_id}|{nombre_pol}", pdf_url, "HOUSEFD"),
                            epoch_publicacion=dt_pub.timestamp(),
                            fecha_hora_ampm=formatear_fecha_hora_ampm(dt_pub),
                            hora_ampm=formatear_hora_ampm(dt_pub),
                            ticker="CONGRESO",
                            titulo=f"House Disclosure [{ftype}] {nombre_pol} ({dist}): {desc_tipo}"[:220],
                            resumen=f"Cámara de Representantes EE.UU. | Categoría: {subcat} | Fecha: {d_f.isoformat()} | Doc: {doc_id}",
                            plataforma="US House Clerk",
                            link=pdf_url,
                            sentimiento="NEUTRAL",
                            impacto_score=56.0,
                            es_macro=True,
                            categoria_mercado="DIVULGACION_CONGRESO",
                            sectores_relacionados=["FINANCIERO", "DEFENSA_AERO_INDUSTRIAL"],
                            seccion="DIVULGACIONES_CONGRESO",
                            subcategoria=subcat,
                        ))
                        cont_fd += 1
                        if cont_fd >= 10:
                            break
        except Exception:
            pass

        return hubo_ok, halladas

    # ------------------------------------------------------------------
    # 6. CONSTRUCCIÓN DE TAREAS POR TICKER + REGISTRO Y CONTEXTO
    # ------------------------------------------------------------------

    def _construir_tareas_rss_lote_tickers(self, lote_tickers: List[str], ventana_dias: int = 2) -> List[Tuple[str, str, str, str, bool, str]]:
        tareas: List[Tuple[str, str, str, str, bool, str]] = []
        paso = 6
        for i in range(0, len(lote_tickers), paso):
            grupo = lote_tickers[i : i + paso]
            simbolos_yf = ",".join(simbolo_yfinance(t) for t in grupo)
            etiqueta = grupo[0] if len(grupo) == 1 else "MERCADO"
            url_yf = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(simbolos_yf)}&region=US&lang=en-US"
            tareas.append(("NOTICIAS", etiqueta, "Yahoo Finance", url_yf, False, ""))

        for t in lote_tickers[:10]:
            consulta = urllib.parse.quote(f"{t} stock OR earnings OR 8-K OR guidance when:{ventana_dias}d")
            url_gn = f"https://news.google.com/rss/search?q={consulta}&hl=en-US&gl=US&ceid=US:en"
            tareas.append(("NOTICIAS", t, "Google News Finance", url_gn, False, ""))

        return tareas

    def _registrar_lote_noticias(
        self,
        candidatas: List[EventoAnalisis],
        notificar_callbacks: bool,
    ) -> int:
        if not candidatas:
            return 0

        ordenadas = sorted(candidatas, key=lambda n: n.epoch_publicacion)
        nuevas_confirmadas: List[EventoAnalisis] = []

        with self._lock:
            ahora = time.time()
            for item in ordenadas:
                if item.id_noticia in self._ids_vistos:
                    continue
                if item.epoch_captura <= 0.0:
                    item.epoch_captura = ahora
                if item.seccion not in SECCIONES_ANALISIS:
                    item.seccion = "NOTICIAS"
                self._ids_vistos.add(item.id_noticia)
                nuevas_confirmadas.append(item)

                if item.ticker in self._noticias_por_ticker:
                    lista_tk = self._noticias_por_ticker[item.ticker]
                    lista_tk.append(item)
                    if len(lista_tk) > 50:
                        del lista_tk[:-50]
                else:
                    self._noticias_macro.append(item)
                    if len(self._noticias_macro) > 160:
                        del self._noticias_macro[:-160]

                lista_sec_6 = self._eventos_por_seccion.setdefault(item.seccion, [])
                lista_sec_6.append(item)
                if len(lista_sec_6) > 80:
                    del lista_sec_6[:-80]

                for sec in item.sectores_relacionados:
                    lista_sec = self._noticias_por_sector.setdefault(sec, [])
                    lista_sec.append(item)
                    if len(lista_sec) > 50:
                        del lista_sec[:-50]

                self._todas_recientes.append(item)
                if len(self._todas_recientes) > 450:
                    del self._todas_recientes[:-450]

            if nuevas_confirmadas:
                if self._ultima_noticia_hallada_epoch > 0:
                    delta = max(1.0, ahora - self._ultima_noticia_hallada_epoch)
                    self._intervalos_hallazgo.append(delta)
                    if len(self._intervalos_hallazgo) > 30:
                        del self._intervalos_hallazgo[:-30]
                self._ultima_noticia_hallada_epoch = ahora
                self._ultima_noticia_obj = nuevas_confirmadas[-1]

        for noticia in nuevas_confirmadas:
            self.historial.registrar_noticia_stream(noticia.to_dict())
            if notificar_callbacks and self.callback_noticia_relevante is not None:
                es_sector_clave = bool(set(noticia.sectores_relacionados) - {"MACRO_GLOBAL", "GENERAL"})
                if (
                    noticia.ticker in self.tickers_set
                    or noticia.seccion in ("FILINGS_8K", "REPORTES_RESULTADOS", "FED", "DIVULGACIONES_CONGRESO")
                    or noticia.impacto_score >= 62.0
                    or (es_sector_clave and noticia.impacto_score >= 56.0 and noticia.sentimiento != "NEUTRAL")
                ):
                    try:
                        self.callback_noticia_relevante(noticia)
                    except Exception:
                        pass

        return len(nuevas_confirmadas)

    # ------------------------------------------------------------------
    # 7. CARGA INICIAL (2 DÍAS PREVIOS) Y MONITOREO CONTINUO DE 6 SECCIONES
    # ------------------------------------------------------------------

    def descargar_contexto_inicial(self, dias_previos: int = 2) -> int:
        corte_epoch = (datetime.now().astimezone() - timedelta(days=dias_previos)).timestamp()
        tareas_rss = list(self.FEEDS_POR_SECCION)
        tareas_rss.extend(self._construir_tareas_rss_lote_tickers(self.tickers, ventana_dias=dias_previos))

        recolectadas: List[EventoAnalisis] = []
        exitos = 0
        total_fuentes = len(tareas_rss) + min(8, len(self.tickers)) + 2 + min(6, len(self.tickers))

        pool = ThreadPoolExecutor(max_workers=16)
        try:
            futuros = [
                pool.submit(self._descargar_rss, sec, etq, fuente, url, corte_epoch, es_macro, subcat)
                for (sec, etq, fuente, url, es_macro, subcat) in tareas_rss
            ]
            for tk_sym in self.tickers[:8]:
                futuros.append(pool.submit(self._descargar_yfinance_ticker_news, tk_sym, corte_epoch))
            for tk_sym in self.tickers[:6]:
                futuros.append(pool.submit(self._descargar_yfinance_sec_8k, tk_sym, 10))
            futuros.append(pool.submit(self._descargar_filings_8k_edgar, corte_epoch))
            futuros.append(pool.submit(self._descargar_divulgaciones_congreso, 10))

            for fut in as_completed(futuros):
                if self._detener.is_set():
                    break
                try:
                    ok, items = fut.result()
                    if ok:
                        exitos += 1
                    recolectadas.extend(items)
                except Exception:
                    pass
        finally:
            pool.shutdown(wait=not self._detener.is_set(), cancel_futures=True)

        self._ts_ultimo_8k_edgar = time.time()
        self._ts_ultimo_congreso = time.time()

        with self._lock:
            self._ultimo_ciclo_epoch = time.time()
            self._ciclos_completados = 1
            self._fuentes_consultadas_ciclo = total_fuentes
            self._fuentes_exitosas_ciclo = exitos
            self._errores_red_consecutivos = 0 if exitos > 0 else 1

        registradas = self._registrar_lote_noticias(recolectadas, notificar_callbacks=False)
        if self._scanner is not None:
            self.evaluar_reportes_con_scanner(self._scanner)
        return registradas

    def ejecutar_ciclo_tiempo_real(self) -> int:
        corte_epoch = (datetime.now().astimezone() - timedelta(days=2)).timestamp()
        ahora = time.time()

        tam_ventana = min(14, len(self.tickers))
        inicio = self._cursor_ticker
        lote_rotativo = [
            self.tickers[(inicio + idx) % len(self.tickers)]
            for idx in range(tam_ventana)
        ]
        self._cursor_ticker = (inicio + tam_ventana) % max(1, len(self.tickers))

        # Mantener siempre FED, 8-K, Resultados, Geopolítica/Bélico + rotación de resto
        criticos = [
            f for f in self.FEEDS_POR_SECCION
            if f[0] in ("FED", "FILINGS_8K", "REPORTES_RESULTADOS") or f[1] in ("GEOPOLITICA", "MOVIMIENTOS_BELICOS")
        ]
        rotativos = [f for f in self.FEEDS_POR_SECCION if f not in criticos]
        seleccion_feeds = list(criticos)
        if rotativos:
            for j in range(3):
                seleccion_feeds.append(rotativos[(self._cursor_macro + j) % len(rotativos)])
            self._cursor_macro += 3

        tareas_rss = list(seleccion_feeds)
        tareas_rss.extend(self._construir_tareas_rss_lote_tickers(lote_rotativo, ventana_dias=1))

        incluir_edgar_8k = (ahora - self._ts_ultimo_8k_edgar) >= 90.0
        incluir_congreso = (ahora - self._ts_ultimo_congreso) >= 300.0
        sub_yf = lote_rotativo[:4]
        sub_8k_yf = lote_rotativo[:3]
        total_fuentes = (
            len(tareas_rss)
            + len(sub_yf)
            + len(sub_8k_yf)
            + (1 if incluir_edgar_8k else 0)
            + (1 if incluir_congreso else 0)
        )

        recolectadas: List[EventoAnalisis] = []
        exitos = 0

        pool = ThreadPoolExecutor(max_workers=14)
        try:
            futuros = [
                pool.submit(self._descargar_rss, sec, etq, fuente, url, corte_epoch, es_macro, subcat)
                for (sec, etq, fuente, url, es_macro, subcat) in tareas_rss
            ]
            for tk_sym in sub_yf:
                futuros.append(pool.submit(self._descargar_yfinance_ticker_news, tk_sym, corte_epoch))
            for tk_sym in sub_8k_yf:
                futuros.append(pool.submit(self._descargar_yfinance_sec_8k, tk_sym, 5))
            if incluir_edgar_8k:
                futuros.append(pool.submit(self._descargar_filings_8k_edgar, corte_epoch))
                self._ts_ultimo_8k_edgar = ahora
            if incluir_congreso:
                futuros.append(pool.submit(self._descargar_divulgaciones_congreso, 7))
                self._ts_ultimo_congreso = ahora

            for fut in as_completed(futuros):
                if self._detener.is_set():
                    break
                try:
                    ok, items = fut.result()
                    if ok:
                        exitos += 1
                    recolectadas.extend(items)
                except Exception:
                    pass
        finally:
            pool.shutdown(wait=not self._detener.is_set(), cancel_futures=True)

        with self._lock:
            self._ultimo_ciclo_epoch = time.time()
            self._ciclos_completados += 1
            self._fuentes_consultadas_ciclo = total_fuentes
            self._fuentes_exitosas_ciclo = exitos
            self._errores_red_consecutivos = 0 if exitos > 0 else (self._errores_red_consecutivos + 1)

        registrados = self._registrar_lote_noticias(recolectadas, notificar_callbacks=True)
        if self._scanner is not None:
            self.evaluar_reportes_con_scanner(self._scanner)
        return registrados

    def iniciar_monitoreo_continuo(self) -> None:
        if self._hilo_monitoreo and self._hilo_monitoreo.is_alive():
            return
        self._detener.clear()
        self._hilo_monitoreo = threading.Thread(
            target=self._bucle_monitoreo,
            name="AnalisisMercado6SeccionesThread",
            daemon=True,
        )
        self._hilo_monitoreo.start()

    def detener(self) -> None:
        self._detener.set()

    def _bucle_monitoreo(self) -> None:
        while not self._detener.is_set():
            try:
                self.ejecutar_ciclo_tiempo_real()
            except Exception:
                with self._lock:
                    self._errores_red_consecutivos += 1
            self._detener.wait(timeout=INTERVALO_CICLO_NOTICIAS_SEG)

    # ------------------------------------------------------------------
    # 8. CONSULTA DE CONTEXTO POR EMPRESA, LOTE BALANCEADO Y SUPERVISIÓN
    # ------------------------------------------------------------------

    def obtener_contexto_ticker(
        self,
        ticker: str,
        max_ticker: int = 8,
        max_macro: int = 6,
    ) -> Tuple[List[EventoAnalisis], List[EventoAnalisis]]:
        """Devuelve `(eventos_empresa_y_pares, eventos_macro_6_secciones)` para el ticker dado."""
        tk = ticker.upper()
        with self._lock:
            directas = list(self._noticias_por_ticker.get(tk, []))
            vistos_ids = {n.id_noticia for n in directas}

            rel_sector: List[EventoAnalisis] = []
            for sec in obtener_sectores_ticker(tk):
                for n_s in self._noticias_por_sector.get(sec, []):
                    if n_s.id_noticia not in vistos_ids:
                        vistos_ids.add(n_s.id_noticia)
                        rel_sector.append(n_s)

            directas.sort(key=lambda n: (1 if n.seccion in ("FILINGS_8K", "REPORTES_RESULTADOS", "DIVULGACIONES_CONGRESO") else 0, n.epoch_publicacion), reverse=True)
            rel_sector.sort(key=lambda n: (n.impacto_score, n.epoch_publicacion), reverse=True)
            especificas = (directas[:6] + rel_sector[:4])[:max_ticker]

            macro = list(self._noticias_macro)
            macro.sort(key=lambda n: n.epoch_publicacion, reverse=True)
            return especificas, macro[:max_macro]

    def obtener_contexto_lote(
        self,
        tickers_lote: List[str],
        max_por_ticker: int = 5,
        max_mercado_general: int = 12,
    ) -> Tuple[Dict[str, List[EventoAnalisis]], List[EventoAnalisis]]:
        """
        Construye en un solo paso las dos ventanas de contexto para la IA:
        1. `mapa_especificas`: noticias, filings 8-K, reportes de resultados y divulgaciones de congreso de cada empresa.
        2. `contexto_general_mercado`: balanceado entre FED, MACROECONOMIA (avances exteriores, geopolítica, bélico),
           DIVULGACIONES_CONGRESO y sectores clave.
        """
        with self._lock:
            mapa_especificas: Dict[str, List[EventoAnalisis]] = {}
            for tk in tickers_lote:
                tk_u = tk.upper()
                directas = sorted(
                    self._noticias_por_ticker.get(tk_u, []),
                    key=lambda n: (
                        2 if n.seccion in ("FILINGS_8K", "REPORTES_RESULTADOS") else (1 if n.seccion == "DIVULGACIONES_CONGRESO" else 0),
                        n.epoch_publicacion,
                    ),
                    reverse=True,
                )
                vistos_tk = {n.id_noticia for n in directas}
                rel_pares: List[EventoAnalisis] = []
                for sec in obtener_sectores_ticker(tk_u):
                    for n_s in self._noticias_por_sector.get(sec, []):
                        if n_s.id_noticia not in vistos_tk and n_s.impacto_score >= 56.0:
                            vistos_tk.add(n_s.id_noticia)
                            rel_pares.append(n_s)
                rel_pares.sort(key=lambda n: n.epoch_publicacion, reverse=True)
                sel = (directas[:4] + rel_pares[:2])[:max_por_ticker]
                mapa_especificas[tk_u] = sel

            # Ventana general balanceada explícitamente entre FED, MACROECONOMIA, CONGRESO, 8-K y NOTICIAS
            seleccion_general: List[EventoAnalisis] = []
            ids_gen: Set[str] = set()
            for seccion_prio in ("FED", "MACROECONOMIA", "DIVULGACIONES_CONGRESO", "FILINGS_8K", "REPORTES_RESULTADOS", "NOTICIAS"):
                pool_sec = sorted(
                    [it for it in self._eventos_por_seccion.get(seccion_prio, []) if it.es_macro or it.impacto_score >= 68.0],
                    key=lambda n: (n.impacto_score, n.epoch_publicacion),
                    reverse=True,
                )
                cupo = 3 if seccion_prio in ("FED", "MACROECONOMIA") else 2
                agregados = 0
                for item in pool_sec:
                    if item.id_noticia in ids_gen:
                        continue
                    seleccion_general.append(item)
                    ids_gen.add(item.id_noticia)
                    agregados += 1
                    if agregados >= cupo or len(seleccion_general) >= max_mercado_general:
                        break
                if len(seleccion_general) >= max_mercado_general:
                    break

            if len(seleccion_general) < max_mercado_general:
                candidatas_macro = sorted(
                    self._noticias_macro,
                    key=lambda n: (n.impacto_score, n.epoch_publicacion),
                    reverse=True,
                )
                for item in candidatas_macro:
                    if item.id_noticia not in ids_gen:
                        seleccion_general.append(item)
                        ids_gen.add(item.id_noticia)
                        if len(seleccion_general) >= max_mercado_general:
                            break

            return mapa_especificas, seleccion_general

    def obtener_noticias_en_rango(
        self,
        inicio_epoch: float,
        fin_epoch: Optional[float] = None,
        tickers_prioritarios: Optional[List[str]] = None,
        max_items: int = 24,
    ) -> List[EventoAnalisis]:
        """Devuelve los eventos capturados dentro de `[inicio_epoch, fin_epoch]` tras salir de modo offline."""
        prio_set = {t.strip().upper() for t in (tickers_prioritarios or []) if t.strip()}
        fin = float(fin_epoch) if fin_epoch is not None else (time.time() + 1.0)
        ini = float(inicio_epoch) - 0.5

        with self._lock:
            pool: Dict[str, EventoAnalisis] = {}
            for item in self._todas_recientes:
                pool[item.id_noticia] = item
            for tk_p in prio_set:
                for item in self._noticias_por_ticker.get(tk_p, []):
                    pool[item.id_noticia] = item
            for item in self._noticias_macro:
                pool[item.id_noticia] = item

            candidatas: List[Tuple[Tuple[int, int, float, float], EventoAnalisis]] = []
            for item in pool.values():
                ts_ref = item.epoch_captura if item.epoch_captura > 0 else item.epoch_publicacion
                if ini <= ts_ref <= fin:
                    es_tk_prio = 1 if item.ticker in prio_set else 0
                    es_dir = 1 if item.sentimiento in ("POSITIVA", "NEGATIVA") else 0
                    candidatas.append(
                        ((es_tk_prio, es_dir, float(item.impacto_score), float(ts_ref)), item)
                    )

            candidatas.sort(key=lambda par: par[0], reverse=True)
            elegidas = [par[1] for par in candidatas[:max_items]]
            elegidas.sort(key=lambda n: n.epoch_captura if n.epoch_captura > 0 else n.epoch_publicacion)
            return elegidas

    def obtener_resumen_secciones(self) -> Dict[str, int]:
        """Conteo de eventos acumulados en cada una de las 6 secciones de `analisis.py`."""
        with self._lock:
            return {sec: len(lst) for sec, lst in self._eventos_por_seccion.items()}

    def total_noticias_capturadas(self) -> int:
        with self._lock:
            return len(self._ids_vistos)

    def obtener_estado_supervision(self) -> Dict[str, object]:
        with self._lock:
            ahora = time.time()
            ref_ultima = self._ultima_noticia_hallada_epoch or self._inicio_monitoreo_epoch
            seg_sin_nueva = max(0, int(ahora - ref_ultima))
            seg_desde_sondeo = max(0, int(ahora - self._ultimo_ciclo_epoch)) if self._ultimo_ciclo_epoch else 0

            umbral_adv = UMBRAL_LATENCIA_NOTICIAS_ADVERTENCIA
            umbral_anormal = UMBRAL_LATENCIA_NOTICIAS_ANORMAL
            if len(self._intervalos_hallazgo) >= 4:
                media_int = sum(self._intervalos_hallazgo) / len(self._intervalos_hallazgo)
                umbral_adv = max(UMBRAL_LATENCIA_NOTICIAS_ADVERTENCIA, int(media_int * 3.0))
                umbral_anormal = max(UMBRAL_LATENCIA_NOTICIAS_ANORMAL, int(media_int * 6.0))

            if self._errores_red_consecutivos >= 2:
                estado = "ANORMAL (FALLO RED)"
                nivel = "CRITICO"
            elif self._ultimo_ciclo_epoch > 0 and seg_desde_sondeo > (INTERVALO_CICLO_NOTICIAS_SEG * 4):
                estado = "ANORMAL (SONDEO DETENIDO)"
                nivel = "CRITICO"
            elif seg_sin_nueva >= umbral_anormal:
                estado = "ANORMAL (SIN EVENTOS NUEVOS)"
                nivel = "CRITICO"
            elif seg_sin_nueva >= umbral_adv:
                estado = "VIGILANCIA (INTERVALO LARGO)"
                nivel = "ADVERTENCIA"
            else:
                estado = "NORMAL"
                nivel = "OK"

            if self._ultima_noticia_obj is not None:
                hora_ult = self._ultima_noticia_obj.hora_ampm
                ticker_ult = self._ultima_noticia_obj.ticker
                plat_ult = self._ultima_noticia_obj.plataforma
                sec_ult = self._ultima_noticia_obj.seccion
            else:
                hora_ult = "--:--:-- --"
                ticker_ult = "ESPERANDO"
                plat_ult = "Inicializando"
                sec_ult = "NOTICIAS"

            return {
                "estado": estado,
                "nivel": nivel,
                "hora_ultima_noticia": hora_ult,
                "segundos_desde_ultima": seg_sin_nueva,
                "ultimo_ticker": ticker_ult,
                "ultima_plataforma": plat_ult,
                "ultima_seccion": sec_ult,
                "fuentes_ok": self._fuentes_exitosas_ciclo,
                "fuentes_total": self._fuentes_consultadas_ciclo,
                "total_noticias": len(self._ids_vistos),
                "conteo_secciones": {s: len(l) for s, l in self._eventos_por_seccion.items()},
                "ciclos": self._ciclos_completados,
            }


# Alias de compatibilidad hacia atrás
GestorInfoMarket = GestorAnalisisMercado
