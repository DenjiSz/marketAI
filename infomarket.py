# -*- coding: utf-8 -*-
"""
infomarket.py
=============
Motor de inteligencia de noticias bursátiles, contexto macroeconómico global y detección sectorial
para consultas múltiples interrelacionadas en tiempo real.

Responsabilidades principales:
1. Precarga histórica de hasta 2 días previos (48h) + vigilancia continua en paralelo mediante
   Google News RSS, Yahoo Finance RSS, CNBC, MarketWatch y `yfinance.Ticker.news`.
2. Cobertura explícita de contexto general de mercado y sectores clave:
   - Tasas de interés / Fed / Treasury Yields / Inflación CPI-PCE
   - Conflictos geopolíticos entre países / Aranceles / Sanciones / Energía
   - Noticias sectoriales e interrelacionadas: Semiconductores + IA + Chips + Memoria RAM/HBM,
     Óptica/Data Centers, Tecnológico, Financiero, Salud/Biotech, Consumo/Retail y Defensa/Aero.
3. Clasificación automática de sentimiento (`POSITIVA`, `NEGATIVA`, `NEUTRAL`), puntuación
   de impacto catalizador (0-100) y asignación de sectores relacionados (`sectores_relacionados`)
   para propagación inteligente hacia acciones hermanas en el método de consulta múltiple.
4. Supervisión continua de latencia y salud del flujo para la penúltima línea (`Supervisión Noticias`).
"""

from __future__ import annotations

import email.utils
import hashlib
import re
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

import feedparser
import requests
import yfinance as yf

from config import (
    INTERVALO_CICLO_NOTICIAS_SEG,
    UMBRAL_LATENCIA_NOTICIAS_ADVERTENCIA,
    UMBRAL_LATENCIA_NOTICIAS_ANORMAL,
    formatear_fecha_hora_ampm,
    formatear_hora_ampm,
    obtener_sectores_ticker,
    simbolo_yfinance,
)
from historial import GestorHistorial


# ==============================================================================
# 1. MODELO DE DATOS, LÉXICO Y CLASIFICACIÓN SECTORIAL/MACROECONÓMICA
# ==============================================================================

@dataclass
class NoticiaMercado:
    """Representa una noticia verificada de una acción, sector o evento macroeconómico global."""

    id_noticia: str
    epoch_publicacion: float
    fecha_hora_ampm: str
    hora_ampm: str
    ticker: str               # Ticker de la lista (ej. NVDA, MU) o etiqueta MACRO/FED/GEOPOLITICA/SECTOR
    titulo: str
    resumen: str
    plataforma: str           # Ej. Yahoo Finance, Reuters, Bloomberg, CNBC, MarketWatch
    link: str
    sentimiento: str          # "POSITIVA" | "NEGATIVA" | "NEUTRAL"
    impacto_score: float      # 0.0 - 100.0 (nivel de relevancia como catalizador)
    es_macro: bool
    categoria_mercado: str = "EMPRESA"
    sectores_relacionados: List[str] = field(default_factory=list)
    epoch_captura: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


PALABRAS_POSITIVAS = {
    "beat": 2.2, "beats": 2.2, "surge": 2.4, "surges": 2.4, "soar": 2.5, "soars": 2.5,
    "upgrade": 2.6, "upgraded": 2.6, "outperform": 2.0, "overweight": 1.8, "buy rating": 2.2,
    "record revenue": 2.8, "record profit": 2.8, "raises guidance": 3.0, "raised guidance": 3.0,
    "launch": 1.7, "launches": 1.7, "unveils": 1.9, "new processor": 2.3, "new chip": 2.3,
    "ai chip": 2.2, "breakthrough": 2.4, "partnership": 1.8, "contract": 1.8, "buyback": 2.0,
    "dividend hike": 1.9, "approval": 2.5, "fda approves": 3.0, "bullish": 1.9, "rally": 1.8,
    "rate cut": 2.3, "cooling inflation": 2.2, "soft landing": 1.8, "strong demand": 2.2,
    "memory shortage": 2.1, "hbm demand": 2.3, "ai spending": 2.2, "capex increase": 2.1,
    "ceasefire": 2.2, "trade deal": 2.1, "stimulus": 2.0,
    "supera": 2.2, "sube": 1.6, "dispara": 2.4, "récord": 2.3, "alcista": 1.9, "nuevo procesador": 2.3,
}

PALABRAS_NEGATIVAS = {
    "miss": 2.4, "misses": 2.4, "plunge": 2.7, "plunges": 2.7, "tumble": 2.4, "slump": 2.3,
    "downgrade": 2.7, "downgraded": 2.7, "underperform": 2.2, "underweight": 2.0, "sell rating": 2.5,
    "cuts guidance": 3.0, "lowered guidance": 3.0, "warning": 2.3, "profit warning": 3.0,
    "investigation": 2.6, "sec probe": 2.9, "lawsuit": 2.2, "antitrust": 2.3, "subpoena": 2.7,
    "recall": 2.5, "delay": 2.0, "delayed": 2.0, "layoffs": 1.9, "bankruptcy": 3.5, "default": 3.0,
    "tariff": 2.1, "tariffs": 2.1, "export ban": 2.8, "export curb": 2.7, "sanctions": 2.4,
    "war": 2.4, "conflict": 2.1, "escalation": 2.2, "missile": 2.3, "blockade": 2.4,
    "hawkish": 2.1, "hot inflation": 2.4, "rate hike": 2.5, "recession": 2.6, "bearish": 2.0,
    "shortfall": 2.4, "oversupply": 2.3, "chip glut": 2.5,
    "cae": 1.7, "desploma": 2.6, "recorta": 2.1, "demanda débil": 2.1, "inflación sube": 2.3, "bajista": 2.0,
}

CATALIZADORES_ALTO_IMPACTO = {
    "earnings", "eps", "quarterly results", "guidance", "fed", "fomc", "powell", "cpi", "pce",
    "interest rate", "treasury yield", "inflation", "acquisition", "merger", "buyout", "fda",
    "processor", "gpu", "semiconductor", "dram", "nand", "hbm", "memory", "upgrade", "downgrade",
    "price target", "contract", "billion", "ai", "data center", "optical", "tariff", "sanctions",
    "war", "geopolitical", "middle east", "china", "opec", "oil",
}

# Reglas de clasificación sectorial para relacionar noticias entre empresas pares y macroeconomía
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
        ("federal reserve", "fomc", "interest rate", "rate cut", "rate hike", "powell", "treasury yield", "fed "),
    ),
    (
        "INFLACION",
        ["CONSUMO", "FINANCIERO", "TECNOLOGIA"],
        ("cpi", "pce", "inflation", "consumer price", "producer price", "ppi", "jobs report", "payroll"),
    ),
    (
        "GEOPOLITICA",
        ["DEFENSA_AERO_INDUSTRIAL", "ENERGIA_MATERIAS", "SEMICONDUCTORES_IA"],
        ("war", "conflict", "missile", "sanctions", "tariff", "tariffs", "trade war", "export ban", "export curb", "geopolitic", "strait", "taiwan", "middle east", "ukraine", "russia", "china trade"),
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


def evaluar_sentimiento_e_impacto(titulo: str, resumen: str = "", es_macro: bool = False) -> Tuple[str, float]:
    """
    Evalúa el sentimiento (`POSITIVA`, `NEGATIVA`, `NEUTRAL`) y la fuerza catalizadora (0-100)
    analizando el titular y el extracto de la noticia.
    """
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
    """
    Determina la categoría macro/sectorial de la noticia y qué sectores de la cartera afecta,
    permitiendo capturar ripples entre empresas interrelacionadas (ej. chips + memoria RAM + IA).
    """
    texto = f" {titulo} {resumen} ".lower()
    sectores_hit: List[str] = []
    categoria = "EMPRESA" if es_ticker_lista else etiqueta_defecto

    for cat_nombre, sectores_asociados, palabras in REGLAS_CATEGORIA_SECTORES:
        if any(p in texto for p in palabras):
            if categoria == etiqueta_defecto or (not es_ticker_lista and categoria == "MACRO"):
                categoria = cat_nombre
            elif es_ticker_lista and cat_nombre in ("SEMICONDUCTORES_IA", "GEOPOLITICA", "TASAS_FED"):
                # Permite identificar si la noticia de una empresa también sacude su cadena de suministro
                categoria = cat_nombre
            for s in sectores_asociados:
                if s not in sectores_hit:
                    sectores_hit.append(s)

    if es_ticker_lista:
        for s_tk in obtener_sectores_ticker(ticker_asignado):
            if s_tk not in sectores_hit:
                sectores_hit.append(s_tk)

    # Si no cayó en ninguna regla específica y vino de etiqueta macro conocida
    if not sectores_hit:
        mapa_etq = {
            "FED": ["FINANCIERO", "TECNOLOGIA", "CONSUMO"],
            "TASAS_FED": ["FINANCIERO", "TECNOLOGIA", "CONSUMO"],
            "INFLACION": ["CONSUMO", "FINANCIERO", "TECNOLOGIA"],
            "GEOPOLITICA": ["DEFENSA_AERO_INDUSTRIAL", "ENERGIA_MATERIAS", "SEMICONDUCTORES_IA"],
            "SEMICONDUCTORES_IA": ["SEMICONDUCTORES_IA", "MEMORIA_ALMACENAMIENTO"],
            "TECNOLOGIA": ["TECNOLOGIA"],
            "FINANCIERO": ["FINANCIERO"],
            "SALUD": ["SALUD"],
            "CONSUMO": ["CONSUMO"],
        }
        sectores_hit = mapa_etq.get(etiqueta_defecto, ["MACRO_GLOBAL"])

    return categoria, sectores_hit


# ==============================================================================
# 2. GESTOR PRINCIPAL INFOMARKET
# ==============================================================================

class GestorInfoMarket:
    """
    Descarga, clasifica sectorialmente, deduplica y supervisa noticias específicas por acción,
    sectores interrelacionados y eventos macroeconómicos/geopolíticos globales.
    """

    # Fuentes RSS macroeconómicas, geopolíticas y sectoriales de respuesta rápida
    FEEDS_MACRO: List[Tuple[str, str, str]] = [
        (
            "TASAS_FED",
            "Google News Fed/Tasas",
            "https://news.google.com/rss/search?q=Federal+Reserve+interest+rates+OR+FOMC+OR+Powell+OR+Treasury+yields+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "INFLACION",
            "Google News Inflación",
            "https://news.google.com/rss/search?q=US+CPI+inflation+OR+PCE+inflation+OR+payroll+market+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "GEOPOLITICA",
            "Google News Geopolítica",
            "https://news.google.com/rss/search?q=geopolitical+conflict+OR+tariffs+OR+sanctions+OR+Middle+East+oil+market+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "SEMICONDUCTORES_IA",
            "Google News Chips/IA/RAM",
            "https://news.google.com/rss/search?q=semiconductor+OR+AI+chips+OR+DRAM+memory+HBM+OR+NVIDIA+OR+TSMC+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "TECNOLOGIA",
            "Google News Tech/Cloud",
            "https://news.google.com/rss/search?q=tech+stocks+cloud+cybersecurity+software+AI+earnings+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "FINANCIERO",
            "Google News Financiero",
            "https://news.google.com/rss/search?q=US+banking+financial+stocks+Wall+Street+credit+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "SALUD",
            "Google News Salud/Pharma",
            "https://news.google.com/rss/search?q=healthcare+biotech+pharma+FDA+approval+stocks+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "CONSUMO",
            "Google News Consumo/Retail",
            "https://news.google.com/rss/search?q=consumer+spending+retail+stocks+Walmart+sales+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "EARNINGS",
            "Google News Resultados",
            "https://news.google.com/rss/search?q=earnings+report+beats+OR+misses+guidance+Wall+Street+when:2d&hl=en-US&gl=US&ceid=US:en",
        ),
        (
            "MACRO",
            "CNBC Markets",
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        ),
        (
            "MACRO",
            "MarketWatch Top",
            "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
        ),
        (
            "MACRO",
            "Yahoo Finance Market",
            "https://finance.yahoo.com/news/rssindex",
        ),
    ]

    def __init__(
        self,
        tickers: List[str],
        historial: GestorHistorial,
        callback_noticia_relevante: Optional[Callable[[NoticiaMercado], None]] = None,
    ) -> None:
        self.tickers = [t.strip().upper() for t in tickers if t.strip()]
        self.tickers_set: Set[str] = set(self.tickers)
        self.historial = historial
        self.callback_noticia_relevante = callback_noticia_relevante

        self._lock = threading.RLock()
        self._ids_vistos: Set[str] = set()
        self._noticias_por_ticker: Dict[str, List[NoticiaMercado]] = {t: [] for t in self.tickers}
        self._noticias_macro: List[NoticiaMercado] = []
        self._noticias_por_sector: Dict[str, List[NoticiaMercado]] = {}
        self._todas_recientes: List[NoticiaMercado] = []

        self._cursor_ticker = 0
        self._cursor_macro = 0
        self._detener = threading.Event()
        self._hilo_monitoreo: Optional[threading.Thread] = None

        self._inicio_monitoreo_epoch: float = time.time()
        self._ultimo_ciclo_epoch: float = 0.0
        self._ultima_noticia_hallada_epoch: float = 0.0
        self._ultima_noticia_obj: Optional[NoticiaMercado] = None
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

        patrones_seguros = [re.escape(t) for t in sorted(self.tickers, key=len, reverse=True) if len(t) >= 2]
        self._regex_tickers = (
            re.compile(r"\b(" + "|".join(patrones_seguros) + r")\b")
            if patrones_seguros
            else None
        )

    # ------------------------------------------------------------------
    # 1. UTILIDADES DE FECHA, DEDUPLICACIÓN Y EXTRACCIÓN DE PLATAFORMA
    # ------------------------------------------------------------------

    @staticmethod
    def _generar_id(titulo: str, link: str) -> str:
        titulo_norm = re.sub(r"\s+", " ", titulo.strip().lower())
        semilla = f"{titulo_norm}|{link.strip().split('?')[0]}"
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

    # ------------------------------------------------------------------
    # 2. DESCARGADORES RSS Y YFINANCE CON TIMEOUT ESTRICTO
    # ------------------------------------------------------------------

    def _descargar_rss(
        self,
        etiqueta_ticker: str,
        nombre_fuente: str,
        url: str,
        corte_minimo_epoch: float,
        es_macro: bool,
    ) -> Tuple[bool, List[NoticiaMercado]]:
        halladas: List[NoticiaMercado] = []
        try:
            resp = requests.get(url, headers=self._headers, timeout=6.0)
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
                sentimiento, impacto = evaluar_sentimiento_e_impacto(
                    titulo_limpio, resumen_limpio, es_macro=es_macro_final
                )
                cat_mkt, sectores_rel = clasificar_categoria_y_sectores(
                    titulo_limpio, resumen_limpio, etiqueta_ticker, ticker_asignado, es_tk_lista
                )

                noticia = NoticiaMercado(
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
                )
                halladas.append(noticia)

            return True, halladas
        except Exception:
            return False, []

    def _descargar_yfinance_ticker_news(
        self,
        ticker_usuario: str,
        corte_minimo_epoch: float,
    ) -> Tuple[bool, List[NoticiaMercado]]:
        halladas: List[NoticiaMercado] = []
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
                cat_mkt, sectores_rel = clasificar_categoria_y_sectores(
                    titulo, resumen, "EMPRESA", ticker_usuario, True
                )
                halladas.append(
                    NoticiaMercado(
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
                    )
                )
            return True, halladas
        except Exception:
            return False, []

    def _construir_tareas_rss_lote_tickers(self, lote_tickers: List[str], ventana_dias: int = 2) -> List[Tuple[str, str, str, bool]]:
        tareas: List[Tuple[str, str, str, bool]] = []
        paso = 6
        for i in range(0, len(lote_tickers), paso):
            grupo = lote_tickers[i : i + paso]
            simbolos_yf = ",".join(simbolo_yfinance(t) for t in grupo)
            etiqueta = grupo[0] if len(grupo) == 1 else "MERCADO"
            url_yf = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(simbolos_yf)}&region=US&lang=en-US"
            tareas.append((etiqueta, "Yahoo Finance", url_yf, False))

        for t in lote_tickers[:10]:
            consulta = urllib.parse.quote(f"{t} stock OR earnings OR chip OR guidance when:{ventana_dias}d")
            url_gn = f"https://news.google.com/rss/search?q={consulta}&hl=en-US&gl=US&ceid=US:en"
            tareas.append((t, "Google News Finance", url_gn, False))

        return tareas

    # ------------------------------------------------------------------
    # 3. REGISTRO DEDUPLICADO Y PUBLICACIÓN A /mkinfo + CALLBACK IA
    # ------------------------------------------------------------------

    def _registrar_lote_noticias(
        self,
        candidatas: List[NoticiaMercado],
        notificar_callbacks: bool,
    ) -> int:
        if not candidatas:
            return 0

        ordenadas = sorted(candidatas, key=lambda n: n.epoch_publicacion)
        nuevas_confirmadas: List[NoticiaMercado] = []

        with self._lock:
            ahora = time.time()
            for item in ordenadas:
                if item.id_noticia in self._ids_vistos:
                    continue
                if item.epoch_captura <= 0.0:
                    item.epoch_captura = ahora
                self._ids_vistos.add(item.id_noticia)
                nuevas_confirmadas.append(item)

                if item.ticker in self._noticias_por_ticker:
                    lista_tk = self._noticias_por_ticker[item.ticker]
                    lista_tk.append(item)
                    if len(lista_tk) > 50:
                        del lista_tk[:-50]
                else:
                    self._noticias_macro.append(item)
                    if len(self._noticias_macro) > 120:
                        del self._noticias_macro[:-120]

                for sec in item.sectores_relacionados:
                    lista_sec = self._noticias_por_sector.setdefault(sec, [])
                    lista_sec.append(item)
                    if len(lista_sec) > 50:
                        del lista_sec[:-50]

                self._todas_recientes.append(item)
                if len(self._todas_recientes) > 360:
                    del self._todas_recientes[:-360]

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
                # Permitir paso al Smart Gatekeeper para noticias de acciones O macro/sectoriales relevantes
                es_sector_clave = bool(set(noticia.sectores_relacionados) - {"MACRO_GLOBAL", "GENERAL"})
                if (
                    noticia.ticker in self.tickers_set
                    or noticia.impacto_score >= 60.0
                    or (es_sector_clave and noticia.impacto_score >= 54.0 and noticia.sentimiento != "NEUTRAL")
                ):
                    try:
                        self.callback_noticia_relevante(noticia)
                    except Exception:
                        pass

        return len(nuevas_confirmadas)

    # ------------------------------------------------------------------
    # 4. CARGA INICIAL DE 2 DÍAS PREVIOS Y MONITOREO CONTINUO
    # ------------------------------------------------------------------

    def descargar_contexto_inicial(self, dias_previos: int = 2) -> int:
        corte_epoch = (datetime.now().astimezone() - timedelta(days=dias_previos)).timestamp()
        tareas_rss: List[Tuple[str, str, str, bool]] = []

        for etq, nombre_fuente, url in self.FEEDS_MACRO:
            tareas_rss.append((etq, nombre_fuente, url, True))

        tareas_rss.extend(self._construir_tareas_rss_lote_tickers(self.tickers, ventana_dias=dias_previos))

        recolectadas: List[NoticiaMercado] = []
        exitos = 0
        total_fuentes = len(tareas_rss) + min(8, len(self.tickers))

        pool = ThreadPoolExecutor(max_workers=14)
        try:
            futuros = [
                pool.submit(self._descargar_rss, etq, fuente, url, corte_epoch, es_macro)
                for (etq, fuente, url, es_macro) in tareas_rss
            ]
            for tk_sym in self.tickers[:8]:
                futuros.append(pool.submit(self._descargar_yfinance_ticker_news, tk_sym, corte_epoch))

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
            self._ciclos_completados = 1
            self._fuentes_consultadas_ciclo = total_fuentes
            self._fuentes_exitosas_ciclo = exitos
            self._errores_red_consecutivos = 0 if exitos > 0 else 1

        return self._registrar_lote_noticias(recolectadas, notificar_callbacks=False)

    def ejecutar_ciclo_tiempo_real(self) -> int:
        corte_epoch = (datetime.now().astimezone() - timedelta(days=2)).timestamp()

        tam_ventana = min(14, len(self.tickers))
        inicio = self._cursor_ticker
        lote_rotativo = [
            self.tickers[(inicio + idx) % len(self.tickers)]
            for idx in range(tam_ventana)
        ]
        self._cursor_ticker = (inicio + tam_ventana) % max(1, len(self.tickers))

        # Rotar feeds macro/sectoriales manteniendo siempre los 4 críticos + bloque rotativo
        feeds_criticos = self.FEEDS_MACRO[:4] + self.FEEDS_MACRO[-3:]
        resto_feeds = self.FEEDS_MACRO[4:-3]
        if resto_feeds:
            idx_m = self._cursor_macro % len(resto_feeds)
            feeds_criticos.extend([resto_feeds[idx_m], resto_feeds[(idx_m + 1) % len(resto_feeds)]])
            self._cursor_macro += 2

        tareas_rss: List[Tuple[str, str, str, bool]] = []
        for etq, nombre_fuente, url in feeds_criticos:
            tareas_rss.append((etq, nombre_fuente, url, True))
        tareas_rss.extend(self._construir_tareas_rss_lote_tickers(lote_rotativo, ventana_dias=1))

        recolectadas: List[NoticiaMercado] = []
        exitos = 0
        sub_yf = lote_rotativo[:4]
        total_fuentes = len(tareas_rss) + len(sub_yf)

        pool = ThreadPoolExecutor(max_workers=12)
        try:
            futuros = [
                pool.submit(self._descargar_rss, etq, fuente, url, corte_epoch, es_macro)
                for (etq, fuente, url, es_macro) in tareas_rss
            ]
            for tk_sym in sub_yf:
                futuros.append(pool.submit(self._descargar_yfinance_ticker_news, tk_sym, corte_epoch))

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
            if exitos == 0:
                self._errores_red_consecutivos += 1
            else:
                self._errores_red_consecutivos = 0

        return self._registrar_lote_noticias(recolectadas, notificar_callbacks=True)

    def iniciar_monitoreo_continuo(self) -> None:
        if self._hilo_monitoreo and self._hilo_monitoreo.is_alive():
            return
        self._detener.clear()
        self._hilo_monitoreo = threading.Thread(
            target=self._bucle_monitoreo,
            name="InfoMarketThread",
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
    # 5. CONSULTA DE CONTEXTO POR EMPRESA, POR LOTE Y SUPERVISIÓN
    # ------------------------------------------------------------------

    def obtener_contexto_ticker(
        self,
        ticker: str,
        max_ticker: int = 8,
        max_macro: int = 6,
    ) -> Tuple[List[NoticiaMercado], List[NoticiaMercado]]:
        """
        Devuelve `(noticias_especificas_o_interrelacionadas, noticias_macro_y_sectoriales)`
        para el ticker dado, incorporando noticias de pares del mismo sector si faltan titulares directos.
        """
        tk = ticker.upper()
        with self._lock:
            directas = list(self._noticias_por_ticker.get(tk, []))
            vistos_ids = {n.id_noticia for n in directas}

            # Añadir noticias interrelacionadas de los mismos sectores (ej. RAM/IA/Chips para MU o AMD)
            rel_sector: List[NoticiaMercado] = []
            for sec in obtener_sectores_ticker(tk):
                for n_s in self._noticias_por_sector.get(sec, []):
                    if n_s.id_noticia not in vistos_ids:
                        vistos_ids.add(n_s.id_noticia)
                        rel_sector.append(n_s)

            directas.sort(key=lambda n: n.epoch_publicacion, reverse=True)
            rel_sector.sort(key=lambda n: (n.impacto_score, n.epoch_publicacion), reverse=True)
            especificas = (directas[:6] + rel_sector[:4])[:max_ticker]

            macro = list(self._noticias_macro)
            macro.sort(key=lambda n: n.epoch_publicacion, reverse=True)
            return especificas, macro[:max_macro]

    def obtener_contexto_lote(
        self,
        tickers_lote: List[str],
        max_por_ticker: int = 5,
        max_mercado_general: int = 10,
    ) -> Tuple[Dict[str, List[NoticiaMercado]], List[NoticiaMercado]]:
        """
        Construye en un solo paso las dos ventanas de contexto para el método de consulta múltiple:
        1. `mapa_especificas`: noticias específicas (+ interrelacionadas clave) por cada empresa del lote.
        2. `contexto_general_mercado`: ventana unificada de tasas Fed, inflación, conflictos entre países
           y noticias de los principales sectores (Tecnología/Chips/RAM, Financiero, Salud, Consumo, Defensa).
        """
        with self._lock:
            mapa_especificas: Dict[str, List[NoticiaMercado]] = {}
            sectores_lote: Set[str] = set()
            ids_especificos_usados: Set[str] = set()

            for tk in tickers_lote:
                tk_u = tk.upper()
                directas = sorted(
                    self._noticias_por_ticker.get(tk_u, []),
                    key=lambda n: n.epoch_publicacion,
                    reverse=True,
                )
                vistos_tk = {n.id_noticia for n in directas}
                rel_pares: List[NoticiaMercado] = []
                for sec in obtener_sectores_ticker(tk_u):
                    sectores_lote.add(sec)
                    for n_s in self._noticias_por_sector.get(sec, []):
                        if n_s.id_noticia not in vistos_tk and n_s.impacto_score >= 55.0:
                            vistos_tk.add(n_s.id_noticia)
                            rel_pares.append(n_s)
                rel_pares.sort(key=lambda n: n.epoch_publicacion, reverse=True)
                sel = (directas[:4] + rel_pares[:2])[:max_por_ticker]
                mapa_especificas[tk_u] = sel
                for it in sel:
                    ids_especificos_usados.add(it.id_noticia)

            # Construir ventana general balanceada: Macro/Tasas/Geopolítica + Sectores clave
            candidatas_macro = sorted(
                self._noticias_macro,
                key=lambda n: (
                    1 if n.categoria_mercado in ("TASAS_FED", "GEOPOLITICA", "INFLACION") else 0,
                    n.impacto_score,
                    n.epoch_publicacion,
                ),
                reverse=True,
            )

            seleccion_general: List[NoticiaMercado] = []
            ids_gen: Set[str] = set()
            categorias_cubiertas: Dict[str, int] = {}

            # Priorizar diversidad de categorías (Fed/Tasas, Geopolítica, Inflación, Sectores del lote, Otros sectores)
            for item in candidatas_macro:
                if item.id_noticia in ids_gen:
                    continue
                cat = item.categoria_mercado or "MACRO"
                if categorias_cubiertas.get(cat, 0) >= 2 and len(seleccion_general) < max_mercado_general - 2:
                    continue
                seleccion_general.append(item)
                ids_gen.add(item.id_noticia)
                categorias_cubiertas[cat] = categorias_cubiertas.get(cat, 0) + 1
                if len(seleccion_general) >= max_mercado_general:
                    break

            # Completar si aún hay espacio con las noticias macro más recientes
            if len(seleccion_general) < max_mercado_general:
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
    ) -> List[NoticiaMercado]:
        """
        Devuelve las noticias capturadas o publicadas dentro del intervalo `[inicio_epoch, fin_epoch]`
        para alimentar el marco de contexto cuando el sistema vuelve de modo offline.
        """
        prio_set = {t.strip().upper() for t in (tickers_prioritarios or []) if t.strip()}
        fin = float(fin_epoch) if fin_epoch is not None else (time.time() + 1.0)
        ini = float(inicio_epoch) - 0.5

        with self._lock:
            pool: Dict[str, NoticiaMercado] = {}
            for item in self._todas_recientes:
                pool[item.id_noticia] = item
            for tk_p in prio_set:
                for item in self._noticias_por_ticker.get(tk_p, []):
                    pool[item.id_noticia] = item
            for item in self._noticias_macro:
                pool[item.id_noticia] = item

            candidatas: List[Tuple[Tuple[int, int, float, float], NoticiaMercado]] = []
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
                estado = "ANORMAL (SIN NOTICIAS NUEVAS)"
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
            else:
                hora_ult = "--:--:-- --"
                ticker_ult = "ESPERANDO"
                plat_ult = "Inicializando"

            return {
                "estado": estado,
                "nivel": nivel,
                "hora_ultima_noticia": hora_ult,
                "segundos_desde_ultima": seg_sin_nueva,
                "ultimo_ticker": ticker_ult,
                "ultima_plataforma": plat_ult,
                "fuentes_ok": self._fuentes_exitosas_ciclo,
                "fuentes_total": self._fuentes_consultadas_ciclo,
                "total_noticias": len(self._ids_vistos),
                "ciclos": self._ciclos_completados,
            }
