"""
Módulo infomarket.py
Escaneo periódico en tiempo real de noticias bursátiles (89 empresas), Reserva Federal (Fed),
tasas de interés, macroeconomía y eventos geopolíticos/económicos en Europa, Medio Oriente y Asia.

Correcciones operativas clave:
- Consultas con cabeceras y parámetros anti-caché (Cache-Control: no-cache + timestamp) para evitar
  respuestas congeladas por CDN en ejecuciones continuas.
- Integración del endpoint JSON en tiempo real de Yahoo Finance Search News (/v1/finance/search),
  TradingView Headlines API, Google News RSS, Yahoo RSS, CNBC, MarketWatch y Federal Reserve RSS.
- Ventana temporal de sesión con deduplicación estricta por UID/título: en el primer ciclo muestra
  hasta 2 días previos; en los ciclos siguientes detecta cualquier noticia nueva que se indexe durante
  la sesión (incluso si la fuente RSS tuvo latencia de minutos en publicarla), sin repetir ninguna.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import colorama
import requests

colorama.init(autoreset=True)


# =====================================================================
# PALETA DE COLORES ANSI (Resaltado de fondo en terminal)
# =====================================================================
class Colores:
    RESET = "\033[0m"
    # Morado (resaltado de fondo para símbolo en evento "nueva noticia")
    MORADO_RESALTADO = "\033[1;97;45m"
    # Azul (resaltado de fondo azul para [noticias] o [indicadores] cuando no hay entradas/cambios)
    AZUL_RESALTADO = "\033[1;97;44m"
    CIAN = "\033[1;36m"


def resaltar_morado(texto: str) -> str:
    """Resalta el símbolo de la acción en morado cuando hay una nueva noticia."""
    return f"{Colores.MORADO_RESALTADO} {texto} {Colores.RESET}"


def resaltar_azul(texto: str) -> str:
    """Resalta con fondo azul la palabra dentro de los corchetes (ej. [noticias] / [indicadores])."""
    return f"{Colores.AZUL_RESALTADO}{texto}{Colores.RESET}"


def formatear_fecha(fecha: dt.datetime) -> str:
    """
    Formatea un datetime al estilo: 11/11/2026 12:00:00 pm
    """
    if fecha.tzinfo is not None:
        fecha = fecha.astimezone()
    am_pm = "am" if fecha.hour < 12 else "pm"
    hora_12 = fecha.strftime("%I:%M:%S")
    fecha_str = fecha.strftime("%d/%m/%Y")
    return f"{fecha_str} {hora_12} {am_pm}"


# =====================================================================
# LISTA DE 89 EMPRESAS, NOMBRES CORPORATIVOS Y TÓPICOS MACRO
# =====================================================================
EMPRESAS: List[str] = [
    "PANW", "LITE", "CLS", "STRL", "CIEN", "AGX", "MU", "WDC", "VIAV", "FIX",
    "SNDK", "NVDA", "STX", "PWR", "LINC", "POWL", "HWM", "ENLT", "EZPW", "RBC",
    "GLW", "CAT", "TTMI", "VRT", "CECO", "ALM", "CRS", "FTI", "TSEM", "MOD",
    "FTAI", "IESC", "COHR", "JBL", "GTX", "KLAC", "CW", "FEIM", "IBKR", "CPRX",
    "ESLT", "LRCX", "ADEA", "ASX", "COKE", "SNEX", "CASY", "BWAY", "BTSG", "ENVA",
    "BLX", "MUFG", "WELL", "APG", "TSM", "FCFS", "SMFG", "ESE", "WMT", "AAMI",
    "HLT", "AIT", "CDNL", "PLPC", "PL", "NVT", "STLD", "AXSM", "WWD", "SEI",
    "CAH", "ODC", "CCJ", "ATI", "MFG", "MCK", "ANAB", "TKO", "MIRM", "KNSA",
    "UBS", "AEIS", "R", "NGS", "GS", "NOK", "ADI", "ITRN", "AER",
]

MAPA_NOMBRES_EMPRESAS: Dict[str, str] = {
    "PANW": "Palo Alto Networks", "LITE": "Lumentum", "CLS": "Celestica",
    "STRL": "Sterling Infrastructure", "CIEN": "Ciena", "AGX": "Argan",
    "MU": "Micron", "WDC": "Western Digital", "VIAV": "Viavi",
    "FIX": "Comfort Systems", "SNDK": "SanDisk", "NVDA": "Nvidia",
    "STX": "Seagate", "PWR": "Quanta Services", "LINC": "Lincoln Educational",
    "POWL": "Powell Industries", "HWM": "Howmet Aerospace", "ENLT": "Enlight Renewable",
    "EZPW": "EZCORP", "RBC": "RBC Bearings", "GLW": "Corning",
    "CAT": "Caterpillar", "TTMI": "TTM Technologies", "VRT": "Vertiv",
    "CECO": "CECO Environmental", "ALM": "Almonty", "CRS": "Carpenter Technology",
    "FTI": "TechnipFMC", "TSEM": "Tower Semiconductor", "MOD": "Modine",
    "FTAI": "FTAI Aviation", "IESC": "IES Holdings", "COHR": "Coherent",
    "JBL": "Jabil", "GTX": "Garrett Motion", "KLAC": "KLA Corp",
    "CW": "Curtiss-Wright", "FEIM": "Frequency Electronics", "IBKR": "Interactive Brokers",
    "CPRX": "Catalyst Pharma", "ESLT": "Elbit Systems", "LRCX": "Lam Research",
    "ADEA": "Adeia", "ASX": "ASE Technology", "COKE": "Coca-Cola Consolidated",
    "SNEX": "StoneX", "CASY": "Casey's General", "BWAY": "BrainsWay",
    "BTSG": "BrightSpring", "ENVA": "Enova", "BLX": "Bladex",
    "MUFG": "Mitsubishi UFJ", "WELL": "Welltower", "APG": "APi Group",
    "TSM": "Taiwan Semiconductor", "FCFS": "FirstCash", "SMFG": "Sumitomo Mitsui",
    "ESE": "ESCO Technologies", "WMT": "Walmart", "AAMI": "Acadian Asset",
    "HLT": "Hilton", "AIT": "Applied Industrial", "CDNL": "Cardinal",
    "PLPC": "Preformed Line", "PL": "Planet Labs", "NVT": "nVent Electric",
    "STLD": "Steel Dynamics", "AXSM": "Axsome", "WWD": "Woodward",
    "SEI": "Solaris Energy", "CAH": "Cardinal Health", "ODC": "Oil-Dri",
    "CCJ": "Cameco", "ATI": "ATI Inc", "MFG": "Mizuho",
    "MCK": "McKesson", "ANAB": "AnaptysBio", "TKO": "TKO Group",
    "MIRM": "Mirum Pharma", "KNSA": "Kiniksa", "UBS": "UBS Group",
    "AEIS": "Advanced Energy", "R": "Ryder System", "NGS": "Natural Gas Services",
    "GS": "Goldman Sachs", "NOK": "Nokia", "ADI": "Analog Devices",
    "ITRN": "Ituran", "AER": "AerCap",
}

TOPICOS_MACRO: Dict[str, str] = {
    "FED": '("Federal Reserve" OR "FOMC" OR "interest rates" OR "Jerome Powell" OR "rate cut" OR "rate hike" OR "Fed minutes")',
    "MACRO": '("US inflation" OR "CPI" OR "GDP" OR "nonfarm payrolls" OR "treasury yields" OR "semiconductors" OR "AI chips" OR "AMD processors")',
    "EUROPA": '("ECB" OR "European Central Bank" OR "Eurozone economy" OR "Europe inflation" OR "Bank of England")',
    "MEDIO_ORIENTE": '("Middle East" OR "oil prices" OR "Red Sea" OR "OPEC" OR "geopolitical tension")',
    "ASIA": '("China economy" OR "Bank of Japan" OR "BOJ" OR "Taiwan semiconductor" OR "Asia markets" OR "PBOC")',
}


# =====================================================================
# ESTRUCTURA DE NOTICIA
# =====================================================================
class Noticia:
    def __init__(
        self,
        simbolo: str,
        plataforma: str,
        fecha_publicacion: dt.datetime,
        link: str,
        titulo: str = "",
        fecha_encontrada: Optional[dt.datetime] = None,
    ):
        self.simbolo = simbolo.upper()
        self.plataforma = re.sub(r"\s+", "_", plataforma.strip().lower()) or "web"
        self.fecha_publicacion = (
            fecha_publicacion.astimezone()
            if fecha_publicacion.tzinfo is not None
            else fecha_publicacion.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
        )
        self.link = link.strip()
        self.titulo = titulo.strip()
        self.fecha_encontrada = (
            fecha_encontrada
            if fecha_encontrada is not None
            else dt.datetime.now().astimezone()
        )

    @property
    def uid(self) -> str:
        """Identificador único para evitar duplicados por link o título normalizado."""
        base_link = self.link.split("?")[0].strip().lower()
        titulo_norm = re.sub(r"\W+", "", self.titulo.lower())[:80]
        raw = f"{self.simbolo}|{base_link or titulo_norm}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    @property
    def uid_titulo(self) -> str:
        """Hash secundario basado en el título para filtrar la misma noticia republicada."""
        titulo_norm = re.sub(r"\W+", "", self.titulo.lower())[:85]
        if not titulo_norm:
            return self.uid
        return hashlib.sha1(
            f"{titulo_norm}".encode("utf-8", errors="ignore")
        ).hexdigest()

    def formatear_linea(self) -> str:
        """
        Formato requerido:
        11/11/2026 12:00:00 pm | símbolo de la accion | tradingview | 11/11/2026 12:00:07 pm | link exacto de la noticia
        Con el símbolo resaltado en morado.
        """
        str_encontrada = formatear_fecha(self.fecha_encontrada)
        str_publicada = formatear_fecha(self.fecha_publicacion)
        simbolo_coloreado = resaltar_morado(self.simbolo)
        return (
            f"{str_encontrada} | {simbolo_coloreado} | {self.plataforma} | "
            f"{str_publicada} | {self.link}"
        )


# =====================================================================
# GESTOR DE INFOMARKET
# =====================================================================
class GestorInfoMarket:
    HISTORIAL_PATH = Path(__file__).resolve().parent / "historial_noticias.json"

    def __init__(
        self,
        empresas: Optional[List[str]] = None,
        max_workers: int = 20,
        timeout_http: int = 5,
    ):
        self.empresas = empresas if empresas is not None else list(EMPRESAS)
        self.max_workers = max_workers
        self.timeout_http = timeout_http

        self.primera_ejecucion: bool = True
        self.inicio_sesion: Optional[dt.datetime] = None
        self.ultimo_escaneo: Optional[dt.datetime] = None
        self.ciclo_actual: int = 0

        self.uids_vistos: Set[str] = set()
        self.titulos_vistos: Set[str] = set()
        self.noticias_por_simbolo: Dict[str, List[Noticia]] = {}
        self.noticias_macro: List[Noticia] = []

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, application/rss+xml, application/xml, text/xml, */*",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )

    def limpiar_historial(self) -> None:
        """
        Auto-gestión de historial: elimina datos previos en disco y reinicia estado en memoria.
        """
        self.uids_vistos.clear()
        self.titulos_vistos.clear()
        self.noticias_por_simbolo.clear()
        self.noticias_macro.clear()
        self.primera_ejecucion = True
        self.inicio_sesion = None
        self.ultimo_escaneo = None
        self.ciclo_actual = 0

        if self.HISTORIAL_PATH.exists():
            try:
                self.HISTORIAL_PATH.unlink()
            except OSError:
                pass

    def _guardar_historial_disco(self) -> None:
        try:
            datos = {
                "inicio_sesion": (
                    self.inicio_sesion.isoformat() if self.inicio_sesion else None
                ),
                "ultimo_escaneo": (
                    self.ultimo_escaneo.isoformat() if self.ultimo_escaneo else None
                ),
                "total_vistos": len(self.uids_vistos),
                "uids_vistos": list(self.uids_vistos)[-8000:],
            }
            self.HISTORIAL_PATH.write_text(
                json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    # -----------------------------------------------------------------
    # PARSEO DE FECHAS Y FUENTES EN TIEMPO REAL
    # -----------------------------------------------------------------
    @staticmethod
    def _parsear_fecha(raw_date: object) -> Optional[dt.datetime]:
        if raw_date is None:
            return None
        if isinstance(raw_date, (int, float)):
            try:
                return dt.datetime.fromtimestamp(float(raw_date), tz=dt.timezone.utc).astimezone()
            except (ValueError, OSError):
                return None
        if isinstance(raw_date, str):
            texto = raw_date.strip()
            if not texto:
                return None
            try:
                parsed = parsedate_to_datetime(texto)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=dt.timezone.utc)
                return parsed.astimezone()
            except Exception:
                pass
            try:
                iso_txt = texto.replace("Z", "+00:00")
                parsed = dt.datetime.fromisoformat(iso_txt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=dt.timezone.utc)
                return parsed.astimezone()
            except Exception:
                pass
        return None

    def _fetch_yahoo_search_news_symbol(self, simbolo: str, cb: int) -> List[Noticia]:
        """
        Consulta el endpoint JSON en tiempo real de Yahoo Finance (/v1/finance/search).
        A diferencia del RSS tradicional, este endpoint devuelve noticias al instante con
        su timestamp exacto (providerPublishTime) sin demoras de caché RSS.
        """
        noticias: List[Noticia] = []
        url = (
            "https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={urllib.parse.quote(simbolo)}&quotesCount=0&newsCount=8&_={cb}"
        )
        try:
            resp = self._session.get(url, timeout=self.timeout_http)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("news", []) if isinstance(data, dict) else []
                for item in items:
                    pub_ts = item.get("providerPublishTime")
                    fecha_pub = self._parsear_fecha(pub_ts)
                    link = (item.get("link") or "").strip()
                    titulo = (item.get("title") or "").strip()
                    publisher = (item.get("publisher") or "yahoo_finance").strip().lower()
                    if link and fecha_pub:
                        noticias.append(
                            Noticia(
                                simbolo=simbolo,
                                plataforma=publisher,
                                fecha_publicacion=fecha_pub,
                                link=link,
                                titulo=titulo,
                            )
                        )
        except Exception:
            pass
        return noticias

    def _fetch_tradingview_symbol(self, simbolo: str, cb: int) -> List[Noticia]:
        """Obtiene titulares en tiempo real desde TradingView Headlines API."""
        noticias: List[Noticia] = []
        url = (
            "https://news-headlines.tradingview.com/v2/headlines"
            f"?client=web&lang=en&symbol={urllib.parse.quote(simbolo)}&_={cb}"
        )
        try:
            resp = self._session.get(url, timeout=self.timeout_http)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", []) if isinstance(data, dict) else []
                for item in items[:8]:
                    pub_ts = item.get("published")
                    fecha_pub = self._parsear_fecha(pub_ts)
                    if not fecha_pub:
                        continue
                    story_path = item.get("storyPath") or ""
                    link = item.get("link") or (
                        f"https://www.tradingview.com{story_path}" if story_path else ""
                    )
                    titulo = item.get("title", "")
                    if link:
                        noticias.append(
                            Noticia(
                                simbolo=simbolo,
                                plataforma="tradingview",
                                fecha_publicacion=fecha_pub,
                                link=link,
                                titulo=titulo,
                            )
                        )
        except Exception:
            pass
        return noticias

    def _fetch_yahoo_rss_symbol(self, simbolo: str, cb: int) -> List[Noticia]:
        """Obtiene noticias del RSS de Yahoo Finance con parámetro anti-caché."""
        noticias: List[Noticia] = []
        url = (
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={urllib.parse.quote(simbolo)}&region=US&lang=en-US&_={cb}"
        )
        try:
            resp = self._session.get(url, timeout=self.timeout_http)
            if resp.status_code != 200 or not resp.content:
                return noticias
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:8]:
                titulo = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date_str = (item.findtext("pubDate") or "").strip()
                fecha_pub = self._parsear_fecha(pub_date_str)
                if link and fecha_pub:
                    plataforma = "tradingview" if "tradingview.com" in link else "yahoo_finance"
                    noticias.append(
                        Noticia(
                            simbolo=simbolo,
                            plataforma=plataforma,
                            fecha_publicacion=fecha_pub,
                            link=link,
                            titulo=titulo,
                        )
                    )
        except Exception:
            pass
        return noticias

    def _fetch_google_news_batch(
        self,
        simbolos_lote: List[str],
        cb: int,
        etiqueta_fija: Optional[str] = None,
        query_custom: Optional[str] = None,
    ) -> List[Noticia]:
        """
        Consulta Google News RSS por lote de símbolos (usando tanto el ticker como el
        nombre de la empresa) o por consulta macroeconómica, con ordenamiento por fecha.
        """
        noticias: List[Noticia] = []
        ventana_q = "when:2d" if self.primera_ejecucion else "when:1d"
        if query_custom:
            q = f"{query_custom} {ventana_q}"
        else:
            partes = []
            for s in simbolos_lote:
                nombre = MAPA_NOMBRES_EMPRESAS.get(s.upper(), "")
                if nombre:
                    partes.append(f'"{s}" OR "{nombre}"')
                else:
                    partes.append(f'"{s} stock"')
            terminos = " OR ".join(partes)
            q = f"({terminos}) {ventana_q}"

        url = (
            "https://news.google.com/rss/search"
            f"?q={urllib.parse.quote(q)}&hl=en-US&gl=US&ceid=US:en&_={cb}"
        )
        try:
            resp = self._session.get(url, timeout=self.timeout_http)
            if resp.status_code != 200 or not resp.content:
                return noticias
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:18]:
                titulo = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date_str = (item.findtext("pubDate") or "").strip()
                source_el = item.find("source")
                fuente = (
                    source_el.text.strip().lower().replace(" ", "_")
                    if source_el is not None and source_el.text
                    else "google_news"
                )
                fecha_pub = self._parsear_fecha(pub_date_str)
                if not link or not fecha_pub:
                    continue

                if etiqueta_fija:
                    simbolo_detectado = etiqueta_fija
                else:
                    simbolo_detectado = self._detectar_simbolo_en_texto(
                        titulo, simbolos_lote
                    )
                if not simbolo_detectado:
                    continue

                noticias.append(
                    Noticia(
                        simbolo=simbolo_detectado,
                        plataforma=fuente,
                        fecha_publicacion=fecha_pub,
                        link=link,
                        titulo=titulo,
                    )
                )
        except Exception:
            pass
        return noticias

    def _fetch_fed_and_world_rss(self, cb: int) -> List[Noticia]:
        """
        Consulta feeds RSS de la Reserva Federal (Fed), CNBC y MarketWatch en tiempo real.
        """
        noticias: List[Noticia] = []
        feeds = [
            ("FED", "federal_reserve", f"https://www.federalreserve.gov/feeds/press_all.xml?_={cb}"),
            ("FED", "cnbc", f"https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258&_={cb}"),
            ("MACRO", "cnbc", f"https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362&_={cb}"),
            ("MACRO", "cnbc", f"https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664&_={cb}"),
            ("MACRO", "marketwatch", f"https://feeds.content.dowjones.io/public/rss/mw_marketpulse?_={cb}"),
        ]
        for etiqueta, plataforma, url in feeds:
            try:
                resp = self._session.get(url, timeout=self.timeout_http)
                if resp.status_code != 200 or not resp.content:
                    continue
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:10]:
                    titulo = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    pub_date_str = (
                        item.findtext("pubDate")
                        or item.findtext("{http://purl.org/dc/elements/1.1/}date")
                        or ""
                    ).strip()
                    fecha_pub = self._parsear_fecha(pub_date_str)
                    if not link or not fecha_pub:
                        continue
                    # Verificar si menciona directamente alguna de nuestras 89 empresas
                    sym_empresa = self._detectar_simbolo_en_texto(titulo, self.empresas)
                    simbolo_final = sym_empresa if sym_empresa else self._clasificar_region_macro(titulo, etiqueta)
                    noticias.append(
                        Noticia(
                            simbolo=simbolo_final,
                            plataforma=plataforma,
                            fecha_publicacion=fecha_pub,
                            link=link,
                            titulo=titulo,
                        )
                    )
            except Exception:
                pass
        return noticias

    @staticmethod
    def _detectar_simbolo_en_texto(texto: str, candidatos: List[str]) -> Optional[str]:
        texto_up = texto.upper()
        texto_low = texto.lower()
        # 1. Buscar símbolo explícito
        for sym in candidatos:
            s_up = sym.upper()
            if len(s_up) >= 2:
                patron = rf"\b{re.escape(s_up)}\b"
                if re.search(patron, texto_up):
                    return s_up
        # 2. Buscar por nombre de la empresa
        for sym in candidatos:
            s_up = sym.upper()
            nombre = MAPA_NOMBRES_EMPRESAS.get(s_up, "").lower()
            if nombre:
                palabras = [p for p in nombre.split() if len(p) > 2]
                clave = palabras[0] if palabras else nombre
                if clave and clave in texto_low:
                    return s_up
        return candidatos[0].upper() if len(candidatos) == 1 else None

    @staticmethod
    def _clasificar_region_macro(titulo: str, default_tag: str = "MACRO") -> str:
        t = titulo.lower()
        if any(k in t for k in ("fed", "fomc", "powell", "interest rate", "rate cut", "rate hike", "treasury")):
            return "FED"
        if any(k in t for k in ("middle east", "israel", "iran", "gaza", "red sea", "opec", "oil", "crude")):
            return "MEDIO_ORIENTE"
        if any(k in t for k in ("europe", "ecb", "eurozone", "germany", "france", "uk ", "boe", "britain")):
            return "EUROPA"
        if any(k in t for k in ("china", "japan", "boj", "asia", "taiwan", "korea", "beijing", "tokyo")):
            return "ASIA"
        return default_tag

    # -----------------------------------------------------------------
    # RECOLECCIÓN PARALELA Y FILTRADO TEMPORAL CORREGIDO
    # -----------------------------------------------------------------
    def _recolectar_todas_las_fuentes(self) -> List[Noticia]:
        """
        Ejecuta en paralelo las consultas para las 89 empresas y temas macro.
        En el primer ciclo consulta exhaustivamente todas las fuentes; en los ciclos siguientes
        combina el barrido completo de RSS + lotes rotativos de APIs JSON en tiempo real
        (Yahoo Search News v1 y TradingView) con cache-busting para máxima frescura.
        """
        recolectadas: List[Noticia] = []
        tareas = []
        cb = int(time.time() * 1000)

        # Seleccionar subconjunto rotativo para APIs JSON instantáneas en ciclos subsecuentes
        n_emp = len(self.empresas)
        if self.primera_ejecucion or n_emp <= 30:
            empresas_json = self.empresas
        else:
            tam_rot = 30
            inicio = ((self.ciclo_actual - 1) * tam_rot) % n_emp
            empresas_json = [
                self.empresas[(inicio + idx) % n_emp] for idx in range(tam_rot)
            ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 1) Yahoo Finance Search News JSON API (tiempo real sin caché RSS)
            for sym in empresas_json:
                tareas.append(executor.submit(self._fetch_yahoo_search_news_symbol, sym, cb))

            # 2) TradingView Headlines API (tiempo real)
            for sym in empresas_json:
                tareas.append(executor.submit(self._fetch_tradingview_symbol, sym, cb))

            # 3) Yahoo Finance RSS para todas las empresas
            for sym in self.empresas:
                tareas.append(executor.submit(self._fetch_yahoo_rss_symbol, sym, cb))

            # 4) Google News en lotes de 5 empresas (con nombre corporativo + ticker)
            tam_lote = 5
            for i in range(0, len(self.empresas), tam_lote):
                lote = self.empresas[i : i + tam_lote]
                tareas.append(executor.submit(self._fetch_google_news_batch, lote, cb, None, None))

            # 5) Tópicos Macro (Fed, Macro, Europa, Medio Oriente, Asia) en Google News
            for tag, query in TOPICOS_MACRO.items():
                tareas.append(executor.submit(self._fetch_google_news_batch, [], cb, tag, query))

            # 6) Feeds RSS directos de la Fed, CNBC y MarketWatch
            tareas.append(executor.submit(self._fetch_fed_and_world_rss, cb))

            for future in as_completed(tareas):
                try:
                    res = future.result()
                    if res:
                        recolectadas.extend(res)
                except Exception:
                    pass

        return recolectadas

    def escanear_y_mostrar(self) -> List[Noticia]:
        """
        Realiza un ciclo de escaneo de noticias:
        - Primera vez: lee y muestra noticias de hasta 2 días previos (48 horas) para contexto.
        - Siguientes veces: verifica todas las noticias recién aparecidas en las fuentes durante
          la sesión activa (con margen amplio para cubrir la latencia de indexación RSS de
          Google News/Yahoo/TradingView) y filtra estrictamente cualquier noticia ya vista.
        - Si no se encontraron noticias nuevas:
          `11/11/2026 12:00:00 pm | sin entradas [noticias]` (con `noticias` resaltado en azul).
        """
        self.ciclo_actual += 1
        ahora = dt.datetime.now().astimezone()

        if self.primera_ejecucion:
            self.inicio_sesion = ahora
            ventana_inicio = ahora - dt.timedelta(days=2)
        else:
            # Permitimos noticias cuya hora de publicación sea de las últimas 6 horas (o desde
            # 1 hora antes de iniciar la sesión), ya que los feeds RSS suelen tardar entre
            # 2 y 40 minutos en indexar una noticia tras su hora oficial de publicación.
            # Como `uids_vistos` y `titulos_vistos` ya registraron todo lo existente al arrancar,
            # solo pasarán noticias genuinamente nuevas que acaban de aparecer en la plataforma.
            base_sesion = self.inicio_sesion or ahora
            ventana_inicio = min(
                ahora - dt.timedelta(hours=6),
                base_sesion - dt.timedelta(hours=1),
            )

        brutas = self._recolectar_todas_las_fuentes()
        nuevas_noticias: List[Noticia] = []

        # Ordenar cronológicamente por fecha de publicación
        brutas.sort(key=lambda n: n.fecha_publicacion)

        for noticia in brutas:
            if noticia.fecha_publicacion < ventana_inicio:
                continue
            if noticia.fecha_publicacion > ahora + dt.timedelta(minutes=10):
                continue

            if noticia.uid in self.uids_vistos or noticia.uid_titulo in self.titulos_vistos:
                continue

            self.uids_vistos.add(noticia.uid)
            self.titulos_vistos.add(noticia.uid_titulo)
            noticia.fecha_encontrada = dt.datetime.now().astimezone()
            nuevas_noticias.append(noticia)

            # Guardar en caché en memoria para cruce con cambios abruptos de precio en indicadores.py
            if noticia.simbolo in TOPICOS_MACRO or noticia.simbolo in (
                "FED", "MACRO", "EUROPA", "MEDIO_ORIENTE", "ASIA"
            ):
                self.noticias_macro.append(noticia)
                self.noticias_macro = self.noticias_macro[-50:]
            else:
                lista_sym = self.noticias_por_simbolo.setdefault(noticia.simbolo, [])
                lista_sym.append(noticia)
                if len(lista_sym) > 25:
                    self.noticias_por_simbolo[noticia.simbolo] = lista_sym[-25:]

        self.ultimo_escaneo = ahora
        self.primera_ejecucion = False
        self._guardar_historial_disco()

        if nuevas_noticias:
            for n in nuevas_noticias:
                print(n.formatear_linea())
        else:
            str_ahora = formatear_fecha(ahora)
            etiqueta_azul = resaltar_azul("noticias")
            print(f"{str_ahora} | sin entradas [{etiqueta_azul}]")

        return nuevas_noticias

    def obtener_noticias_relevantes(self, simbolo: str, max_items: int = 3) -> List[Noticia]:
        """
        Devuelve las noticias más recientes de la acción (y en su defecto macro/Fed)
        para explicar caídas o alzas abruptas detectadas en indicadores.py.
        """
        simbolo_up = simbolo.upper()
        propias = list(self.noticias_por_simbolo.get(simbolo_up, []))
        propias.sort(key=lambda x: x.fecha_publicacion, reverse=True)
        resultado = propias[:max_items]

        if len(resultado) < max_items and self.noticias_macro:
            macros_ordenadas = sorted(
                self.noticias_macro, key=lambda x: x.fecha_publicacion, reverse=True
            )
            for m in macros_ordenadas:
                if len(resultado) >= max_items:
                    break
                resultado.append(m)
        return resultado


if __name__ == "__main__":
    gestor = GestorInfoMarket()
    gestor.limpiar_historial()
    gestor.escanear_y_mostrar()
