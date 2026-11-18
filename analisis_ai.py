# -*- coding: utf-8 -*-
"""
analisis_ai.py
==============
Módulo central de inteligencia artificial con:
1. Método de Consulta Múltiple (Multi-Ticker Batching):
   Evalúa hasta 6 empresas interrelacionadas en una sola llamada al modelo combinando:
   - Ventana unificada de Contexto General de Mercado (FED, MACROECONOMIA, Congresos y Sectores).
   - Ventana de contexto específica por cada empresa del lote con las 6 secciones de `analisis.py`
     (NOTICIAS, FILINGS_8K, REPORTES_RESULTADOS conjunto con `scanner.py`, DIVULGACIONES_CONGRESO,
     FED y MACROECONOMIA) y los 17 indicadores/patrones cuantitativos de `scanner.py`.
2. Smart Gatekeeper Anti-Day-Trading y Anti-Neutrales (reducción agresiva de consultas):
   - Matriz de evidencia multi-pilar (Técnico/Patrones + Noticias + 8-K/Resultados + Congreso + Sector/Macro):
     nunca consulta por una noticia aislada.
   - Ventanas flexibles (`09:15-09:30`, `09:30-09:50` y `15:45-16:00 ET`) vs ventanas estrictas en todo
     otro horario exigiendo combinaciones multi-pilar de altísima convicción.
   - Tope por minuto + tope por hora, cooldown extendido y huella estructural.
3. Contabilidad exacta de Tokens (input/output/thought/cached/total) y Costo estimado USD por consulta
   y acumulado de sesión para el supervisor.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Set, Tuple

import requests
from google import genai
from pydantic import BaseModel, Field

from analisis import EventoAnalisis, GestorAnalisisMercado
from cartera import CarteraVirtual
from config import (
    COOLDOWN_CONSULTA_TICKER_SEG,
    COOLDOWN_NEUTRAL_EXTENDIDO_SEG,
    MAX_ASIGNACION_POR_POSICION_PCT,
    MAX_CONSULTAS_AI_POR_HORA_ESTRICTA,
    MAX_CONSULTAS_AI_POR_MINUTO,
    MAX_TICKERS_LOTE_AI,
    MIN_TICKERS_LOTE_AI,
    MODELOS_IA_DISPONIBLES,
    UMBRAL_CERTEZA_AI_OPERACION,
    calcular_costo_consulta_usd,
    formatear_fecha_hora_ampm,
    normalizar_esfuerzo_modelo,
    obtener_regimen_horario_mercado,
    obtener_sectores_ticker,
    obtener_tickers_relacionados_por_sectores,
    resolver_clave_modelo,
)
from historial import GestorHistorial
from scanner import AnalisisTecnicoTicker, ScannerMercado, formatear_monto_compacto

# Alias internos de compatibilidad
NoticiaMercado = EventoAnalisis
GestorInfoMarket = GestorAnalisisMercado
GestorIndicadores = ScannerMercado


# ==============================================================================
# 1. ESQUEMAS ESTRUCTURADOS PYDANTIC (INDIVIDUAL Y CONSULTA MÚLTIPLE EN LOTE)
# ==============================================================================

class LiquidacionRotacionIA(BaseModel):
    """Orden sugerida por la IA para liquidar parcial o totalmente una posición menos atractiva."""

    ticker: str = Field(description="Símbolo de la posición actual en cartera a reducir o cerrar.")
    porcentaje_liquidar: float = Field(
        description="Porcentaje de esa posición a liquidar (entre 10 y 100)."
    )
    justificacion: str = Field(
        description="Razón cuantitativa/fundamental por la que tiene menor potencial que la nueva oportunidad."
    )


class DecisionModeloIA(BaseModel):
    """Respuesta estructurada individual para compatibilidad puntual o unitaria."""

    alto_potencial: bool = Field(
        description="True SOLO si la acción presenta un alto potencial claro alcista o bajista."
    )
    prediccion_movimiento: Literal["ALCISTA", "BAJISTA", "NEUTRAL"] = Field(
        description="Dirección esperada del movimiento de mercado."
    )
    certeza_pct: float = Field(
        description="Nivel de certeza/confianza de 0 a 100 sobre la tesis planteada."
    )
    accion_cartera: Literal[
        "COMPRAR_LONG",
        "VENDER_SHORT",
        "CERRAR_POSICION",
        "ROTAR_CARTERA",
        "MANTENER",
    ] = Field(description="Acción operativa porcentual a ejecutar en la cartera virtual.")
    porcentaje_capital_asignar: float = Field(
        description="Porcentaje del patrimonio total de la cartera a asignar (0 a 25)."
    )
    liquidaciones_rotacion: List[LiquidacionRotacionIA] = Field(
        default_factory=list,
        description="Lista opcional de posiciones abiertas a liquidar parcial/totalmente si falta liquidez o tienen menor potencial.",
    )
    panorama_mercado: Literal["FAVORABLE", "NEUTRAL", "MALO"] = Field(
        description="Evaluación del panorama macro/técnico general."
    )
    explicacion_ejecutiva: str = Field(
        description="Síntesis concisa en español explicando catalizadores, indicadores clave y gestión de riesgo."
    )


class DecisionTickerLote(BaseModel):
    """Decisión estructurada individual para una empresa dentro de una consulta múltiple en lote."""

    ticker: str = Field(description="Ticker exacto de la empresa evaluada dentro del lote.")
    alto_potencial: bool = Field(
        description="True SOLO si esta empresa específica presenta alto potencial claro alcista o bajista."
    )
    prediccion_movimiento: Literal["ALCISTA", "BAJISTA", "NEUTRAL"] = Field(
        description="Dirección esperada del movimiento de esta acción."
    )
    certeza_pct: float = Field(
        description="Nivel de certeza/confianza de 0 a 100 para esta acción."
    )
    accion_cartera: Literal[
        "COMPRAR_LONG",
        "VENDER_SHORT",
        "CERRAR_POSICION",
        "ROTAR_CARTERA",
        "MANTENER",
    ] = Field(description="Acción operativa porcentual sugerida para esta acción.")
    porcentaje_capital_asignar: float = Field(
        description="Porcentaje del patrimonio total a asignar (0 a 25)."
    )
    liquidaciones_rotacion: List[LiquidacionRotacionIA] = Field(
        default_factory=list,
        description="Posiciones abiertas a reducir/cerrar si se requiere financiar esta entrada.",
    )
    explicacion_ejecutiva: str = Field(
        description="Tesis breve en español (máx 3 líneas) indicando catalizador sectorial/noticioso e indicadores."
    )


class DecisionLoteModeloIA(BaseModel):
    """Respuesta estructurada conjunta para el método de consulta múltiple."""

    panorama_mercado_global: Literal["FAVORABLE", "NEUTRAL", "MALO"] = Field(
        description="Evaluación global del mercado según tasas Fed, geopolítica y flujos sectoriales."
    )
    resumen_macro_sectorial: str = Field(
        description="Síntesis breve del contexto macro/sectorial que afecta al lote."
    )
    decisiones: List[DecisionTickerLote] = Field(
        description="Lista exacta con una decisión por cada ticker incluido en la consulta múltiple."
    )


@dataclass
class EventoImpresionPrincipal:
    """Contiene todos los campos requeridos para imprimir un hallazgo de alto potencial en `main.py`."""

    fecha_hora_ampm: str
    ticker: str
    prediccion_movimiento: str          # "ALCISTA" o "BAJISTA"
    certeza_pct: float
    prompt_enviado: str
    respuesta_modelo_texto: str
    accion_ejecutada_cartera: str
    noticias_analizadas: List[NoticiaMercado]
    analisis_tecnico: AnalisisTecnicoTicker
    tokens_consulta_input: int = 0
    tokens_consulta_output: int = 0
    tokens_consulta_thought: int = 0
    tokens_consulta_total: int = 0
    costo_consulta_usd: float = 0.0


@dataclass
class EstadisticasSupervisorIA:
    """Contadores en tiempo real para la línea `Supervisión AI` (antepenúltima línea) en `main.py`."""

    total_consultas: int = 0            # Llamadas reales enviadas a la API (cada lote cuenta 1)
    total_activos_evaluados: int = 0    # Total de tickers evaluados dentro de todas las consultas
    respuestas_alcistas: int = 0        # Veredictos alcistas / compra / long
    respuestas_bajistas: int = 0        # Veredictos bajistas / venta / short
    respuestas_neutrales: int = 0       # Veredictos sin alto potencial / neutral
    fallos_conexion: int = 0
    fallos_fondos_cuota: int = 0
    estado_agente: str = "LISTO"        # "ACTIVO" | "CONSULTANDO [...]" | "SIN RESPUESTA..."
    detalle_alerta: str = ""
    consultas_omitidas_gatekeeper: int = 0
    consultas_ahorradas_lote: int = 0
    ultimos_tickers_consulta: List[str] = field(default_factory=list)
    regimen_horario: str = ""
    consultas_pausadas: bool = False
    esperando_evento_reanudacion: bool = False
    evento_offline_pendiente: bool = False
    inicio_pausa_str: str = ""
    noticias_durante_pausa: int = 0
    eventos_tec_durante_pausa: int = 0
    total_pausas: int = 0
    tokens_ultima_input: int = 0
    tokens_ultima_output: int = 0
    tokens_ultima_thought: int = 0
    tokens_ultima_cached: int = 0
    tokens_ultima_total: int = 0
    tokens_acum_input: int = 0
    tokens_acum_output: int = 0
    tokens_acum_thought: int = 0
    tokens_acum_cached: int = 0
    tokens_acum_total: int = 0
    costo_ultima_usd: float = 0.0
    costo_acumulado_usd: float = 0.0


def _formatear_duracion_seg(segundos: float) -> str:
    """Convierte segundos a una cadena legible (ej. `45s`, `3m 12s`, `1h 05m 20s`)."""
    tot = max(0, int(round(segundos)))
    hrs, rem = divmod(tot, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


# ==============================================================================
# 2. INSTRUCCIÓN DE SISTEMA INSTITUCIONAL Y UTILIDADES DE ESQUEMA/CONEXIÓN
# ==============================================================================

SYSTEM_INSTRUCTION_FINANCIERO = """Eres un Analista Cuantitativo Institucional y Gestor de Cartera Swing Senior (VAGY-M1).
NO realizas day trading impulsivo por ruido aislado; buscas oportunidades de ALTA CONVICCIÓN respaldadas por confluencia multi-pilar.
Operas mediante MÉTODO DE CONSULTA MÚLTIPLE recibiendo en un solo paquete:
A) VENTANA GENERAL DE MERCADO (`FED`, `MACROECONOMIA` - avances en otros países, tensión geopolítica, movimientos bélicos, `DIVULGACIONES_CONGRESO` y Sectores).
B) ESTADO PORCENTUAL ACTUAL DE LA CARTERA VIRTUAL.
C) VENTANAS ESPECÍFICAS POR EMPRESA del lote: datos financieros, **Reporte de Resultados** conjunto (`fecha próximo reporte`, `ingresos esperados vs reportados`, EPS y sorpresa), **17 indicadores y patrones** (`ORB`, `Bandera`, `Rebote Sobreventa`, `Tendencia Diaria SMA20/50 + Squeeze`, flujo `RVOL`/`CMF`) y eventos de las **6 secciones** (`NOTICIAS`, `FILINGS_8K`, `REPORTES_RESULTADOS`, `DIVULGACIONES_CONGRESO`, `FED`, `MACROECONOMIA`).

REGLAS ESTRICTAS DE DECISIÓN:
1. Evalúa CADA ticker solicitado devolviendo una entrada en `decisiones` (1 por ticker evaluado) comparando fuerza relativa entre pares sectoriales.
2. Enfoque anti-ruido de altísima selectividad: marca `alto_potencial = true` ÚNICAMENTE si existe confluencia sólida técnico+catalizador o ventaja clara (`certeza_pct >= 68`). De lo contrario responde `alto_potencial = false`, `prediccion_movimiento = "NEUTRAL"` y `accion_cartera = "MANTENER"`.
3. Operación en BASE PORCENTUAL: expresa `porcentaje_capital_asignar` como % del patrimonio total (0% si es NEUTRAL, entre 5% y 25% si hay alto potencial).
4. Respeta el `modo_operacion` de la cartera:
   - `"spot"`: solo abre `COMPRAR_LONG` o cierra posiciones `LONG` existentes (`CERRAR_POSICION`). Si una acción abierta corre riesgo grave, marca `prediccion_movimiento = "BAJISTA"` con `alto_potencial = true` para alertar/cerrar; si no está en cartera déjala `MANTENER`.
   - `"short y long"`: puedes abrir libremente `COMPRAR_LONG` y `VENDER_SHORT`.
5. ROTACIÓN DE CARTERA: si falta liquidez o hay posiciones abiertas inferiores, puedes indicar `ROTAR_CARTERA` y `liquidaciones_rotacion`. Si la posición a rotar está en pérdidas (`pyg_pct < 0`), exige `certeza_pct >= 85`.
6. Responde EXCLUSIVAMENTE en JSON válido conforme al esquema solicitado y redacta los textos en español claro."""


def limpiar_api_key(raw_key: str) -> str:
    """Elimina comillas, espacios invisibles o prefijos accidentales como GEMINI_API_KEY=..."""
    limpia = (raw_key or "").strip().strip('"').strip("'")
    for pref in ("GEMINI_API_KEY=", "GOOGLE_API_KEY=", "ANTHROPIC_API_KEY=", "Bearer ", "bearer "):
        if limpia.startswith(pref):
            limpia = limpia[len(pref) :].strip().strip('"').strip("'")
    return limpia


def _aplanar_esquema_pydantic(modelo_pydantic: type[BaseModel]) -> Dict[str, Any]:
    """Resuelve `$defs` / `$ref` en línea para máxima compatibilidad estructurada en Gemini y Claude."""
    raw = modelo_pydantic.model_json_schema()
    defs = raw.pop("$defs", {})

    def _resolver(nodo: Any) -> Any:
        if isinstance(nodo, dict):
            if "$ref" in nodo:
                nombre_def = str(nodo["$ref"]).rsplit("/", 1)[-1]
                if nombre_def in defs:
                    return _resolver(dict(defs[nombre_def]))
            return {k: _resolver(v) for k, v in nodo.items() if k != "title"}
        if isinstance(nodo, list):
            return [_resolver(x) for x in nodo]
        return nodo

    return _resolver(raw)


def obtener_esquema_decision_plano() -> Dict[str, Any]:
    """Devuelve JSON Schema plano de `DecisionModeloIA` (compatibilidad unitaria)."""
    return _aplanar_esquema_pydantic(DecisionModeloIA)


def obtener_esquema_decision_lote_plano() -> Dict[str, Any]:
    """Devuelve JSON Schema plano de `DecisionLoteModeloIA` para el método de consulta múltiple."""
    return _aplanar_esquema_pydantic(DecisionLoteModeloIA)


def diagnosticar_error_autenticacion(exc: Exception) -> Tuple[str, str]:
    """Clasifica fallos en `CREDENCIAL`, `FONDOS` o `CONEXION` en español claro."""
    msg = str(exc)
    msg_low = msg.lower()

    if "access_token_type_unsupported" in msg_low:
        return (
            "CREDENCIAL",
            "Google rechazó esta clave `AQ.` (401 ACCESS_TOKEN_TYPE_UNSUPPORTED). "
            "Está vinculada a otro servicio interno; crea una clave en https://aistudio.google.com/apikey",
        )
    if "leaked" in msg_low or "blocked" in msg_low:
        return (
            "CREDENCIAL",
            "Google bloqueó esta API Key por seguridad. Genera una nueva en https://aistudio.google.com/apikey",
        )
    if any(
        k in msg_low
        for k in (
            "401",
            "403",
            "api_key_invalid",
            "api key not valid",
            "unauthenticated",
            "invalid x-api-key",
            "permission_denied",
            "insufficient credentials",
        )
    ):
        return (
            "CREDENCIAL",
            f"API Key inválida o sin permisos para este modelo/servicio: {msg[:130]}",
        )
    if any(
        k in msg_low
        for k in ("quota", "resource_exhausted", "429", "402", "billing", "credit balance", "payment", "exceeded", "funds")
    ):
        return (
            "FONDOS",
            f"Fondos o cuota agotada en la cuenta del proveedor: {msg[:130]}",
        )
    return ("CONEXION", f"Fallo de red o servicio no disponible: {msg[:130]}")


def _es_error_modelo_no_encontrado(exc_msg: str) -> bool:
    low = exc_msg.lower()
    return ("404" in low or "not_found" in low or "not found" in low) and "401" not in low and "403" not in low


def probar_conexion_modelo_ia(
    modelo_clave: str,
    api_key_raw: str,
    nivel_esfuerzo: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Realiza una prueba en vivo durante la inicialización validando modelo, clave y nivel de esfuerzo.
    Devuelve `(exito, modo_transporte, mensaje_detalle)`.
    """
    clave = limpiar_api_key(api_key_raw)
    clave_mod = resolver_clave_modelo(modelo_clave) or "flash3.8"
    cfg = MODELOS_IA_DISPONIBLES[clave_mod]
    esfuerzo = normalizar_esfuerzo_modelo(clave_mod, nivel_esfuerzo)
    proveedor = str(cfg["proveedor"])
    modelos_intento = [str(cfg["model_id"])] + [str(m) for m in cfg.get("fallback_ids", [])]

    if not clave:
        return False, "sin_clave", "No se ingresó ninguna API Key."

    if proveedor == "gemini":
        cliente_std = genai.Client(api_key=clave, vertexai=False)
        ultima_exc: Optional[Exception] = None

        for m_id in modelos_intento:
            try:
                resp = cliente_std.interactions.create(
                    model=m_id,
                    input="Responde únicamente: OK",
                    store=False,
                    generation_config={"thinking_level": esfuerzo},
                )
                if resp is not None:
                    return (
                        True,
                        "gemini_interactions",
                        f"Conectado a {m_id} (effort={esfuerzo}, Gemini Interactions API).",
                    )
            except Exception as exc_std:
                ultima_exc = exc_std
                if _es_error_modelo_no_encontrado(str(exc_std)) and m_id != modelos_intento[-1]:
                    continue
                # Si la clave empieza por AQ., intentar Vertex Express antes de fallar
                if clave.startswith("AQ."):
                    try:
                        cliente_vx = genai.Client(api_key=clave, vertexai=True)
                        resp_vx = cliente_vx.models.generate_content(
                            model=m_id,
                            contents="Responde únicamente: OK",
                        )
                        if resp_vx is not None:
                            return (
                                True,
                                "vertex_express",
                                f"Conectado a {m_id} (effort={esfuerzo}, Vertex AI Express).",
                            )
                    except Exception:
                        pass
                break

        _, explicacion = diagnosticar_error_autenticacion(ultima_exc or RuntimeError("Fallo de conexión Gemini."))
        return False, "gemini_interactions", explicacion

    # Proveedor Anthropic (Claude Sonnet / Opus / Fable)
    headers = {
        "x-api-key": clave,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    ultimo_err = ""
    for m_id in modelos_intento:
        for usar_effort in (True, False):
            payload: Dict[str, Any] = {
                "model": m_id,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Reply OK"}],
            }
            if usar_effort:
                payload["output_config"] = {"effort": esfuerzo}
            try:
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=18.0,
                )
                if r.status_code == 400 and usar_effort and ("output_config" in r.text or "effort" in r.text):
                    continue
                if r.status_code == 404 and m_id != modelos_intento[-1]:
                    break
                if r.status_code >= 400:
                    _, exp = diagnosticar_error_autenticacion(RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}"))
                    return False, "anthropic", exp
                return (
                    True,
                    "anthropic",
                    f"Conectado a {m_id} (effort={esfuerzo}, Anthropic API).",
                )
            except Exception as exc_a:
                _, ultimo_err = diagnosticar_error_autenticacion(exc_a)
                break

    return False, "anthropic", ultimo_err or "No fue posible conectar con Anthropic."


# ==============================================================================
# 3. ANALIZADOR PRINCIPAL CON CONSULTA MÚLTIPLE Y ANTI-NEUTRALES
# ==============================================================================

class AnalizadorAI:
    """
    Orquesta el filtrado inteligente anti-neutrales, agrupación por consulta múltiple en lote,
    llamadas a Gemini / Claude con el nivel de esfuerzo elegido y ejecución porcentual en cartera.
    """

    def __init__(
        self,
        modelo_clave: str,
        api_key: str,
        modo_operacion: str,
        infomarket: GestorInfoMarket,
        indicadores: GestorIndicadores,
        cartera: CarteraVirtual,
        historial: GestorHistorial,
        callback_alto_potencial: Optional[Callable[[EventoImpresionPrincipal], None]] = None,
        modo_transporte: str = "gemini_interactions",
        nivel_esfuerzo: Optional[str] = None,
    ) -> None:
        clave_canonica = resolver_clave_modelo(modelo_clave) or "flash3.8"
        self.config_modelo = MODELOS_IA_DISPONIBLES[clave_canonica]
        self.modelo_clave = clave_canonica
        self.nivel_esfuerzo = normalizar_esfuerzo_modelo(clave_canonica, nivel_esfuerzo)
        self.proveedor = str(self.config_modelo["proveedor"])
        self.model_id = str(self.config_modelo["model_id"])
        self.fallback_ids: List[str] = [str(m) for m in self.config_modelo.get("fallback_ids", [])]
        self.api_key = limpiar_api_key(api_key)
        self.modo_transporte = modo_transporte
        self.modo_operacion = "short y long" if "short" in modo_operacion.lower() else "spot"

        self.infomarket = infomarket
        self.analisis = infomarket
        self.indicadores = indicadores
        self.scanner = indicadores
        self.cartera = cartera
        self.historial = historial
        self.callback_alto_potencial = callback_alto_potencial

        self._gemini_client: Optional[genai.Client] = None
        self._inicializar_cliente_gemini()

        self._lock = threading.RLock()
        reg_init = obtener_regimen_horario_mercado()
        self._stats = EstadisticasSupervisorIA(
            estado_agente="SIN API KEY" if not self.api_key else "LISTO",
            regimen_horario=str(reg_init["etiqueta"]),
        )

        # Registro multi-pilar de últimas señales por ticker (6 secciones analisis + 17 ind/patrones scanner)
        self._senales_recientes: Dict[str, Dict[str, Any]] = {}
        # ticker -> (epoch, huella, precio, fue_neutral)
        self._ultimo_analisis_ticker: Dict[str, Tuple[float, str, float, bool]] = {}
        self._ventana_timestamps_minuto: Deque[float] = deque()
        self._ventana_timestamps_hora: Deque[float] = deque()
        # cola de candidatos listos para lote: tk -> (prioridad, origen, timestamp_ingreso)
        self._cola_candidatos: Dict[str, Tuple[float, str, float]] = {}
        self._ultimo_uso_tokens_invocacion: Dict[str, Any] = {}

        self._detener = threading.Event()
        self._evento_nuevo_candidato = threading.Event()
        self._hilo_worker: Optional[threading.Thread] = None

        # Control de modo PAUSADO / OFFLINE y marco de contexto de interrupción
        self._consultas_pausadas: bool = False
        self._esperando_evento_post_reanudacion: bool = False
        self._reanudacion_epoch: float = 0.0
        self._evento_despertar_post_pausa: Optional[str] = None
        self._offline_info: Optional[Dict[str, Any]] = None

    def _inicializar_cliente_gemini(self) -> None:
        if self.proveedor == "gemini" and self.api_key:
            try:
                usar_vx = self.modo_transporte == "vertex_express"
                self._gemini_client = genai.Client(api_key=self.api_key, vertexai=usar_vx)
            except Exception:
                self._gemini_client = None
        else:
            self._gemini_client = None

    def actualizar_api_key(self, nueva_api_key: str) -> Tuple[bool, str]:
        """Actualiza y verifica en caliente la API Key desde el comando interactivo `/apikey`."""
        ok, transporte, msg = probar_conexion_modelo_ia(
            self.modelo_clave, nueva_api_key, nivel_esfuerzo=self.nivel_esfuerzo
        )
        with self._lock:
            self.api_key = limpiar_api_key(nueva_api_key)
            if ok:
                self.modo_transporte = transporte
                self._inicializar_cliente_gemini()
                if self._consultas_pausadas:
                    self._stats.estado_agente = "PAUSADA (OFFLINE)"
                elif self._esperando_evento_post_reanudacion:
                    self._stats.estado_agente = "ESPERANDO EVENTO ALZA/BAJA"
                else:
                    self._stats.estado_agente = "CONECTADO Y ACTIVO"
                self._stats.detalle_alerta = ""
                self._ultimo_analisis_ticker.clear()
            else:
                self._stats.estado_agente = "API KEY RECHAZADA"
                self._stats.detalle_alerta = msg[:100]
        return ok, msg

    def actualizar_modelo_o_esfuerzo(
        self,
        modelo_clave: Optional[str] = None,
        nivel_esfuerzo: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Permite cambiar el modelo y/o nivel de esfuerzo en caliente con `/modelo`."""
        with self._lock:
            clave_nueva = resolver_clave_modelo(modelo_clave) if modelo_clave else self.modelo_clave
            if clave_nueva is None:
                return False, f"Modelo '{modelo_clave}' no reconocido."
            cfg_nueva = MODELOS_IA_DISPONIBLES[clave_nueva]
            esf_nuevo = normalizar_esfuerzo_modelo(clave_nueva, nivel_esfuerzo or self.nivel_esfuerzo)

            self.config_modelo = cfg_nueva
            self.modelo_clave = clave_nueva
            self.nivel_esfuerzo = esf_nuevo
            self.proveedor = str(cfg_nueva["proveedor"])
            self.model_id = str(cfg_nueva["model_id"])
            self.fallback_ids = [str(m) for m in cfg_nueva.get("fallback_ids", [])]
            self._inicializar_cliente_gemini()

        if self.api_key:
            ok, trans, msg = probar_conexion_modelo_ia(
                self.modelo_clave, self.api_key, nivel_esfuerzo=self.nivel_esfuerzo
            )
            with self._lock:
                if ok:
                    self.modo_transporte = trans
                    if self._consultas_pausadas:
                        self._stats.estado_agente = "PAUSADA (OFFLINE)"
                    elif self._esperando_evento_post_reanudacion:
                        self._stats.estado_agente = "ESPERANDO EVENTO ALZA/BAJA"
                    else:
                        self._stats.estado_agente = "ACTIVO"
                    self._stats.detalle_alerta = ""
                else:
                    self._stats.detalle_alerta = msg[:100]
            return ok, msg

        return True, f"Configurado {self.modelo_clave} ({self.model_id}) con effort={self.nivel_esfuerzo}."

    # ------------------------------------------------------------------
    # 0. CONTROL DE PAUSA / MODO OFFLINE Y REANUDACIÓN POR EVENTO
    # ------------------------------------------------------------------

    def esta_pausado(self) -> bool:
        with self._lock:
            return self._consultas_pausadas

    def esta_esperando_evento_post_reanudacion(self) -> bool:
        with self._lock:
            return self._esperando_evento_post_reanudacion

    def _resumir_snapshot_tecnico(self, snap: AnalisisTecnicoTicker) -> Dict[str, Any]:
        """Extrae un resumen compacto de datos financieros e indicadores clave para tracking offline."""
        df = snap.datos_financieros
        rvol = self._extraer_rvol_de_analisis(snap)
        ind_map: Dict[str, str] = {}
        for ind in snap.indicadores:
            nom_u = ind.nombre.upper()
            if "RSI" in nom_u:
                ind_map["RSI"] = f"{ind.texto_valor}({ind.evaluacion})"
            elif "MACD" in nom_u:
                ind_map["MACD"] = f"{ind.texto_valor}({ind.evaluacion})"
            elif "CMF" in nom_u:
                ind_map["CMF"] = f"{ind.texto_valor}({ind.evaluacion})"
            elif "ADX" in nom_u:
                ind_map["ADX"] = ind.texto_valor
            elif "SUPERTREND" in nom_u:
                ind_map["ST"] = ind.texto_valor
            elif "ESTRUCTURA" in nom_u:
                ind_map["SR"] = ind.texto_valor

        return {
            "epoch": snap.epoch,
            "hora": snap.fecha_hora_ampm,
            "precio": round(float(df.precio_actual), 4),
            "cambio_pct": round(float(df.cambio_diario_pct), 2),
            "pe": df.pe_ratio,
            "s1": round(float(df.soporte_1), 2),
            "r1": round(float(df.resistencia_1), 2),
            "vwap": round(float(df.vwap_sesion), 2),
            "rvol": round(float(rvol), 2),
            "evaluacion": snap.evaluacion_general,
            "direccion": snap.direccion_potencial,
            "potencial": round(float(snap.potencial_score), 1),
            "certeza": round(float(snap.certeza_pct), 1),
            "alarmas_compra": list(snap.alarmas_compra),
            "alarmas_venta": list(snap.alarmas_venta),
            "ind_clave": ind_map,
        }

    def _actualizar_extremo_offline_de_snapshot(
        self,
        info_off: Dict[str, Any],
        tk: str,
        res: Dict[str, Any],
    ) -> None:
        extremos: Dict[str, Dict[str, Any]] = info_off["extremos_ciclo"]
        precio = float(res.get("precio", 0.0))
        eval_act = str(res.get("evaluacion", "NEUTRAL"))
        ext = extremos.get(tk)

        if ext is None:
            ext = {
                "min_precio": precio,
                "max_precio": precio,
                "ultimo_precio": precio,
                "eval_ultimo": eval_act,
                "dir_ultimo": res.get("direccion", "NEUTRAL"),
                "pot_ultimo": float(res.get("potencial", 0.0)),
                "cert_ultimo": float(res.get("certeza", 30.0)),
                "rvol_ultimo": float(res.get("rvol", 1.0)),
                "s1_ultimo": float(res.get("s1", 0.0)),
                "r1_ultimo": float(res.get("r1", 0.0)),
                "vwap_ultimo": float(res.get("vwap", 0.0)),
                "ind_clave": dict(res.get("ind_clave", {})),
                "transiciones": [],
            }
            ini = info_off["snapshots_inicio"].get(tk)
            if ini and ini.get("evaluacion") != eval_act:
                ext["transiciones"].append(f"{ini['evaluacion']}->{eval_act}")
                info_off["tickers_actualizados"].add(tk)
            extremos[tk] = ext
            return

        if precio > 0:
            if ext["min_precio"] <= 0 or precio < ext["min_precio"]:
                ext["min_precio"] = precio
            if precio > ext["max_precio"]:
                ext["max_precio"] = precio
            ext["ultimo_precio"] = precio

        eval_prev = str(ext.get("eval_ultimo", "NEUTRAL"))
        if eval_act != eval_prev:
            tr_list = ext.setdefault("transiciones", [])
            tr_txt = f"{eval_prev}->{eval_act}"
            if not tr_list or tr_list[-1] != tr_txt:
                tr_list.append(tr_txt)
                if len(tr_list) > 6:
                    del tr_list[:-6]
            info_off["tickers_actualizados"].add(tk)

        ext["eval_ultimo"] = eval_act
        ext["dir_ultimo"] = res.get("direccion", "NEUTRAL")
        ext["pot_ultimo"] = float(res.get("potencial", 0.0))
        ext["cert_ultimo"] = float(res.get("certeza", 30.0))
        ext["rvol_ultimo"] = float(res.get("rvol", 1.0))
        ext["s1_ultimo"] = float(res.get("s1", 0.0))
        ext["r1_ultimo"] = float(res.get("r1", 0.0))
        ext["vwap_ultimo"] = float(res.get("vwap", 0.0))
        if res.get("ind_clave"):
            ext["ind_clave"] = dict(res["ind_clave"])

        ini = info_off["snapshots_inicio"].get(tk)
        if ini and ini.get("precio", 0) > 0 and precio > 0:
            var_pct = abs(precio - float(ini["precio"])) / float(ini["precio"]) * 100.0
            if var_pct >= 0.35:
                info_off["tickers_actualizados"].add(tk)

    def _registrar_evento_indicador_offline(self, analisis: AnalisisTecnicoTicker, n_alm: int) -> None:
        tk = analisis.ticker
        res = self._resumir_snapshot_tecnico(analisis)
        with self._lock:
            info_off = self._offline_info
            if info_off is None:
                return
            if tk not in info_off["snapshots_inicio"]:
                info_off["snapshots_inicio"][tk] = dict(res)
            self._actualizar_extremo_offline_de_snapshot(info_off, tk, res)

            ini_ev = info_off["snapshots_inicio"][tk].get("evaluacion")
            if analisis.es_evento_alto_potencial or n_alm >= 2 or (ini_ev != analisis.evaluacion_general):
                evs_tk = info_off["eventos_indicadores"].setdefault(tk, [])
                entrada = {
                    "hora": analisis.fecha_hora_ampm,
                    "epoch": res["epoch"],
                    "precio": res["precio"],
                    "cambio_pct": res["cambio_pct"],
                    "evaluacion": analisis.evaluacion_general,
                    "direccion": analisis.direccion_potencial,
                    "potencial": res["potencial"],
                    "certeza": res["certeza"],
                    "rvol": res["rvol"],
                    "alarmas": (
                        list(analisis.alarmas_compra)[:3]
                        if analisis.direccion_potencial == "ALCISTA"
                        else list(analisis.alarmas_venta)[:3]
                    ),
                    "ind_clave": res["ind_clave"],
                    "alto_potencial": analisis.es_evento_alto_potencial,
                }
                if (
                    not evs_tk
                    or evs_tk[-1]["evaluacion"] != entrada["evaluacion"]
                    or abs(entrada["precio"] - evs_tk[-1]["precio"]) / max(evs_tk[-1]["precio"], 1e-6) >= 0.004
                    or (entrada["epoch"] - evs_tk[-1]["epoch"]) >= 120.0
                ):
                    evs_tk.append(entrada)
                    if len(evs_tk) > 6:
                        del evs_tk[:-6]
                info_off["tickers_actualizados"].add(tk)
                self._stats.eventos_tec_durante_pausa = sum(
                    len(v) for v in info_off["eventos_indicadores"].values()
                )

    def registrar_ciclo_datos_mercado(self, mapa_precios: Dict[str, Dict[str, Any]]) -> None:
        """
        Registra variaciones de precios e indicadores al cierre de cada ciclo cuantitativo
        mientras las consultas IA estén pausadas (modo offline).
        """
        if not mapa_precios:
            return
        snaps_todos = self.indicadores.obtener_snapshots_todos()
        with self._lock:
            if not self._consultas_pausadas or self._offline_info is None:
                return
            info_off = self._offline_info
            info_off["ciclos_datos"] = int(info_off.get("ciclos_datos", 0)) + 1
            for tk_raw, datos in mapa_precios.items():
                tk = tk_raw.strip().upper()
                snap_full = snaps_todos.get(tk)
                if snap_full is not None:
                    res = self._resumir_snapshot_tecnico(snap_full)
                else:
                    res = {
                        "epoch": time.time(),
                        "hora": formatear_fecha_hora_ampm(),
                        "precio": round(float(datos.get("precio") or 0.0), 4),
                        "cambio_pct": 0.0,
                        "pe": datos.get("pe"),
                        "s1": round(float(datos.get("soporte") or 0.0), 2),
                        "r1": round(float(datos.get("resistencia") or 0.0), 2),
                        "vwap": 0.0,
                        "rvol": 1.0,
                        "evaluacion": str(datos.get("evaluacion") or "NEUTRAL"),
                        "direccion": str(datos.get("direccion") or "NEUTRAL"),
                        "potencial": 0.0,
                        "certeza": round(float(datos.get("certeza") or 30.0), 1),
                        "ind_clave": {},
                    }
                if tk not in info_off["snapshots_inicio"]:
                    info_off["snapshots_inicio"][tk] = dict(res)
                self._actualizar_extremo_offline_de_snapshot(info_off, tk, res)

    def pausar_consultas(self) -> Tuple[bool, str]:
        """
        Pausa todas las consultas al modelo de IA dejando activos InfoMarket, Indicadores,
        Cartera Virtual y las terminales `/v`, `/n`, `/d` operando en modo offline.
        """
        ahora = time.time()
        ahora_str = formatear_fecha_hora_ampm()
        snaps_act = self.indicadores.obtener_snapshots_todos()
        snap_cart = self.cartera.construir_snapshot_estado()

        snaps_resumen = {
            tk: self._resumir_snapshot_tecnico(sn)
            for tk, sn in snaps_act.items()
        }

        with self._lock:
            if self._consultas_pausadas:
                desde = self._offline_info["inicio_str"] if self._offline_info else ahora_str
                return False, f"Las consultas IA ya estaban pausadas (modo offline desde {desde})."

            if self._offline_info is None:
                self._offline_info = {
                    "inicio_epoch": ahora,
                    "inicio_str": ahora_str,
                    "seg_actual_inicio_epoch": ahora,
                    "seg_actual_inicio_str": ahora_str,
                    "fin_epoch": None,
                    "fin_str": None,
                    "segmentos_pausa": [],
                    "patrimonio_inicio_usd": float(snap_cart.get("patrimonio_actual_usd", self.cartera.capital_inicial_usd)),
                    "efectivo_inicio_pct": float(snap_cart.get("efectivo_pct", 100.0)),
                    "posiciones_inicio": [
                        f"{p['ticker']}({p['tipo']},{float(p['peso_pct']):.1f}%)"
                        for p in snap_cart.get("posiciones", [])
                    ],
                    "ciclos_datos": 0,
                    "snapshots_inicio": snaps_resumen,
                    "extremos_ciclo": {},
                    "eventos_indicadores": {},
                    "noticias_offline": {},
                    "tickers_actualizados": set(),
                }
                for tk_s, res_s in snaps_resumen.items():
                    self._actualizar_extremo_offline_de_snapshot(self._offline_info, tk_s, res_s)
            else:
                # Re-pausa antes de haberse consumido el contexto offline previo: conservar acumulado
                self._offline_info["seg_actual_inicio_epoch"] = ahora
                self._offline_info["seg_actual_inicio_str"] = ahora_str
                self._offline_info["fin_epoch"] = None
                self._offline_info["fin_str"] = None
                for tk_s, res_s in snaps_resumen.items():
                    if tk_s not in self._offline_info["snapshots_inicio"]:
                        self._offline_info["snapshots_inicio"][tk_s] = res_s
                    self._actualizar_extremo_offline_de_snapshot(self._offline_info, tk_s, res_s)

            self._cola_candidatos.clear()
            self._consultas_pausadas = True
            self._esperando_evento_post_reanudacion = False
            self._evento_despertar_post_pausa = None

            self._stats.consultas_pausadas = True
            self._stats.esperando_evento_reanudacion = False
            self._stats.evento_offline_pendiente = True
            self._stats.inicio_pausa_str = self._offline_info["inicio_str"]
            self._stats.total_pausas += 1
            self._stats.estado_agente = "PAUSADA (OFFLINE)"

        try:
            self.historial.actualizar_estado_sesion({
                "ia_pausada": True,
                "ia_esperando_evento": False,
                "ia_modo_offline_desde": ahora_str,
            })
        except Exception:
            pass

        return (
            True,
            f"Consultas IA pausadas ({ahora_str}). InfoMarket, Indicadores y Cartera siguen activos offline. Use /reanudar.",
        )

    def reanudar_consultas(self) -> Tuple[bool, str]:
        """
        Reanuda las consultas a la IA tras una pausa offline, exigiendo esperar un próximo
        evento potencialmente alcista o bajista antes de enviar la siguiente consulta (que
        incluirá el marco de contexto offline generado durante la interrupción).
        """
        ahora = time.time()
        ahora_str = formatear_fecha_hora_ampm()
        snaps_act = self.indicadores.obtener_snapshots_todos()

        with self._lock:
            if not self._consultas_pausadas:
                if self._esperando_evento_post_reanudacion:
                    return (
                        False,
                        "La IA ya está reanudada y esperando el próximo evento ALCISTA o BAJISTA para consultar.",
                    )
                return False, "Las consultas al modelo de IA ya están activas."

            info_off = self._offline_info
            seg_ini_ep = float(info_off.get("seg_actual_inicio_epoch", ahora)) if info_off else ahora
            seg_ini_str = str(info_off.get("seg_actual_inicio_str", ahora_str)) if info_off else ahora_str

        # Barrer noticias capturadas en InfoMarket durante el intervalo de pausa
        noticias_rango = self.infomarket.obtener_noticias_en_rango(
            inicio_epoch=seg_ini_ep,
            fin_epoch=ahora + 0.5,
            max_items=36,
        )

        with self._lock:
            if not self._consultas_pausadas:
                return False, "La IA ya fue reanudada."

            info_off = self._offline_info
            if info_off is not None:
                info_off["fin_epoch"] = ahora
                info_off["fin_str"] = ahora_str
                info_off["segmentos_pausa"].append((seg_ini_str, ahora_str))

                for n_r in noticias_rango:
                    if n_r.id_noticia not in info_off["noticias_offline"]:
                        info_off["noticias_offline"][n_r.id_noticia] = n_r
                    if n_r.ticker in self.indicadores.tickers:
                        info_off["tickers_actualizados"].add(n_r.ticker)

                for tk_s, sn in snaps_act.items():
                    res_s = self._resumir_snapshot_tecnico(sn)
                    if tk_s not in info_off["snapshots_inicio"]:
                        info_off["snapshots_inicio"][tk_s] = res_s
                    self._actualizar_extremo_offline_de_snapshot(info_off, tk_s, res_s)

                n_not_off = len(info_off["noticias_offline"])
                n_ev_off = sum(len(v) for v in info_off["eventos_indicadores"].values())
                n_tk_chg = len(info_off["tickers_actualizados"])
            else:
                n_not_off, n_ev_off, n_tk_chg = 0, 0, 0

            self._cola_candidatos.clear()
            self._consultas_pausadas = False
            self._esperando_evento_post_reanudacion = True
            self._reanudacion_epoch = ahora
            self._evento_despertar_post_pausa = None

            self._stats.consultas_pausadas = False
            self._stats.esperando_evento_reanudacion = True
            self._stats.evento_offline_pendiente = True
            self._stats.noticias_durante_pausa = n_not_off
            self._stats.eventos_tec_durante_pausa = n_ev_off
            self._stats.estado_agente = "ESPERANDO EVENTO ALZA/BAJA"

        try:
            self.historial.actualizar_estado_sesion({
                "ia_pausada": False,
                "ia_esperando_evento": True,
                "ia_modo_offline_desde": "",
            })
        except Exception:
            pass

        return (
            True,
            f"IA reanudada ({ahora_str}). Esperando evento ALCISTA/BAJISTA para consultar con contexto offline "
            f"({n_not_off} not., {n_ev_off} eventos ind., {n_tk_chg} emp. con cambios).",
        )

    # ------------------------------------------------------------------
    # 1. SMART GATEKEEPER ANTI-NEUTRALES + CONFLUENCIA MULTI-SEÑAL
    # ------------------------------------------------------------------

    @staticmethod
    def _calcular_huella_contexto(
        ticker: str,
        noticias_tk: List[NoticiaMercado],
        analisis_tec: AnalisisTecnicoTicker,
    ) -> str:
        ult_id_not = noticias_tk[0].id_noticia if noticias_tk else "sin_not"
        cubeta_certeza = int(analisis_tec.certeza_pct // 10)
        alarmas = ",".join(sorted(analisis_tec.alarmas_compra + analisis_tec.alarmas_venta))
        patrones = ",".join(sorted(getattr(analisis_tec, "patrones_detectados", [])))
        semilla = f"{ticker}|{ult_id_not}|{analisis_tec.evaluacion_general}|{cubeta_certeza}|{alarmas}|{patrones}"
        return hashlib.md5(semilla.encode("utf-8")).hexdigest()[:12]

    def _extraer_rvol_de_analisis(self, analisis: AnalisisTecnicoTicker) -> float:
        for ind in analisis.indicadores:
            if "RVOL" in ind.nombre.upper():
                try:
                    return float(ind.valor)
                except Exception:
                    pass
        if analisis.datos_financieros.volumen_promedio_20 > 0:
            return analisis.datos_financieros.volumen_barra / analisis.datos_financieros.volumen_promedio_20
        return 1.0

    def calcular_puntaje_confluencia(
        self,
        ticker: str,
        regimen: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str, bool]:
        """
        Smart Gatekeeper Anti-Day-Trading:
        Evalúa combinaciones multi-pilar reales (Técnico/Patrones + Noticias + 8-K/Resultados +
        Congreso + Sector/Fed/Macro + Régimen Horario).
        - NUNCA dispara por una noticia aislada (evita ruido de day trading).
        - Ventanas flexibles (`09:15-09:30`, `09:30-09:50`, `15:45-16:00 ET`): exige mínimo 2 pilares alineados.
        - Todo otro horario: ESTRICTO — exige combinación multi-pilar (3+ pilares o 2 pilares de máxima
          certeza con patrón estructural + RVOL >= 1.35 + catalizador 8-K/Resultados/Congreso)
          que prácticamente asegure ganancia, o alerta defensiva sobre posición ya abierta.
        """
        tk = ticker.strip().upper()
        reg = regimen or obtener_regimen_horario_mercado()
        ahora = time.time()
        vigencia_seg = 2700.0       # 45 min para noticias/técnico/macro
        vigencia_inst_seg = 7200.0  # 2h para 8-K / Reportes de Resultados / Congreso

        with self._lock:
            mem = dict(self._senales_recientes.get(tk, {}))

        en_cartera = tk in self.cartera.posiciones
        pos_cartera = self.cartera.posiciones.get(tk)

        # Pilar 1: Noticias directas (Sección NOTICIAS)
        tiene_not = (ahora - float(mem.get("not_ts", 0.0))) <= vigencia_seg
        not_score = float(mem.get("not_score", 0.0)) if tiene_not else 0.0
        not_sent = str(mem.get("not_sent", "NEUTRAL")) if tiene_not else "NEUTRAL"
        not_plat = str(mem.get("not_plat", "")) if tiene_not else ""

        # Pilar 2: Fundamental / Filings 8-K / Reportes de Resultados (Secciones 2 y 3)
        tiene_fund = (ahora - float(mem.get("rep_ts", 0.0))) <= vigencia_inst_seg
        rep_score = float(mem.get("rep_score", 0.0)) if tiene_fund else 0.0
        rep_sent = str(mem.get("rep_sent", "NEUTRAL")) if tiene_fund else "NEUTRAL"
        rep_sub = str(mem.get("rep_sub", "")) if tiene_fund else ""

        # Pilar 3: Divulgaciones del Congreso (Sección 4)
        tiene_cong = (ahora - float(mem.get("cong_ts", 0.0))) <= vigencia_inst_seg
        cong_score = float(mem.get("cong_score", 0.0)) if tiene_cong else 0.0
        cong_sent = str(mem.get("cong_sent", "NEUTRAL")) if tiene_cong else "NEUTRAL"
        cong_sub = str(mem.get("cong_sub", "")) if tiene_cong else ""

        # Pilar 4: Cuantitativo Scanner (17 indicadores + patrones ORB/Bandera/Rebote/Squeeze)
        tiene_tec = (ahora - float(mem.get("tec_ts", 0.0))) <= vigencia_seg
        snap_tec: Optional[AnalisisTecnicoTicker] = None
        if not tiene_tec:
            snap_tec, _ = self.indicadores.obtener_snapshot_e_historial(tk)

        if tiene_tec:
            tec_dir = str(mem.get("tec_dir", "NEUTRAL"))
            tec_pot = float(mem.get("tec_pot", 0.0))
            tec_cert = float(mem.get("tec_cert", 30.0))
            tec_rvol = float(mem.get("tec_rvol", 1.0))
            tec_alm = int(mem.get("tec_alm", 0))
            tec_ev = str(mem.get("tec_ev", "NEUTRAL"))
            tec_patrones = list(mem.get("tec_patrones", []))
            tec_contra = int(mem.get("tec_contra", 0))
        elif snap_tec is not None:
            tec_dir = snap_tec.direccion_potencial
            tec_pot = snap_tec.potencial_score
            tec_cert = snap_tec.certeza_pct
            tec_rvol = self._extraer_rvol_de_analisis(snap_tec)
            tec_alm = (
                len(snap_tec.alarmas_compra)
                if tec_dir == "ALCISTA"
                else (len(snap_tec.alarmas_venta) if tec_dir == "BAJISTA" else 0)
            )
            tec_contra = (
                len(snap_tec.alarmas_venta)
                if tec_dir == "ALCISTA"
                else (len(snap_tec.alarmas_compra) if tec_dir == "BAJISTA" else 0)
            )
            tec_ev = snap_tec.evaluacion_general
            tec_patrones = list(getattr(snap_tec, "patrones_detectados", []))
        else:
            tec_dir, tec_pot, tec_cert, tec_rvol, tec_alm, tec_ev, tec_patrones, tec_contra = (
                "NEUTRAL", 0.0, 30.0, 1.0, 0, "NEUTRAL", [], 0
            )

        # Pilar 5: Ripple Sectorial / FED / Macroeconomía (Secciones 5 y 6)
        tiene_rip = (ahora - float(mem.get("rip_ts", 0.0))) <= vigencia_seg
        rip_score = float(mem.get("rip_score", 0.0)) if tiene_rip else 0.0
        rip_sent = str(mem.get("rip_sent", "NEUTRAL")) if tiene_rip else "NEUTRAL"
        rip_orig = str(mem.get("rip_orig", "")) if tiene_rip else ""

        score = 0.0
        razones: List[str] = []
        pilares: Set[str] = set()

        # Evaluación de Pilar NOTICIAS
        if tiene_not and not_sent != "NEUTRAL" and not_score >= 58.0:
            score += not_score * 0.45
            pilares.add("NOTICIAS")
            razones.append(f"Noticia {not_sent} ({not_plat}, imp={not_score:.0f})")

        # Evaluación de Pilar FUNDAMENTAL (8-K / Reportes Resultados + Scanner)
        if tiene_fund and rep_score >= 64.0 and (rep_sent != "NEUTRAL" or "Inminente" in rep_sub):
            score += rep_score * 0.55
            pilares.add("FUNDAMENTAL")
            razones.append(f"{rep_sub or '8-K/Resultados'} ({rep_sent}, imp={rep_score:.0f})")

        # Evaluación de Pilar CONGRESO
        if tiene_cong and (cong_sent != "NEUTRAL" or cong_score >= 66.0):
            score += cong_score * 0.45
            pilares.add("CONGRESO")
            razones.append(f"Congreso[{cong_sub}] ({cong_sent})")

        # Evaluación de Pilar TECNICO (Indicadores + Patrones Swing + Volumen Real)
        bajista_no_operable_spot = (
            self.modo_operacion == "spot" and not en_cartera and tec_dir == "BAJISTA"
        )
        tiene_patron = len(tec_patrones) >= 1
        pilar_tec_valido = (
            not bajista_no_operable_spot
            and tec_dir != "NEUTRAL"
            and tec_cert >= 62.0
            and (tec_alm >= 2 or tiene_patron)
        )
        if pilar_tec_valido:
            pilares.add("TECNICO")
            aporte_tec = (tec_pot * 0.48) + (tec_cert * 0.24)
            if tec_alm >= 3:
                aporte_tec += 7.0
            if tiene_patron:
                aporte_tec += min(12.0, len(tec_patrones) * 6.0)
            if tec_rvol >= 1.35:
                aporte_tec += 10.0
            elif tec_rvol < 1.10 and not reg["es_permisivo"]:
                aporte_tec -= 8.0
            score += aporte_tec
            pat_str = f" [{tec_patrones[0]}]" if tec_patrones else ""
            razones.append(f"{tec_ev}{pat_str} (RVOL {tec_rvol:.2f}x)")

        # Evaluación de Pilar SECTOR_MACRO (Fed / Macroeconomía / Ripple entre pares)
        if tiene_rip and rip_sent != "NEUTRAL" and rip_score >= 60.0:
            score += rip_score * 0.35
            pilares.add("SECTOR_MACRO")
            if rip_orig:
                razones.append(f"Macro/Ripple {rip_orig}")

        # Bonos de alineación direccional entre pilares
        votos_alc = (
            (1 if (pilar_tec_valido and tec_dir == "ALCISTA") else 0)
            + (1 if (tiene_not and not_sent == "POSITIVA") else 0)
            + (1 if (tiene_fund and rep_sent == "POSITIVA") else 0)
            + (1 if (tiene_cong and cong_sent == "POSITIVA") else 0)
            + (1 if (tiene_rip and rip_sent == "POSITIVA") else 0)
        )
        votos_baj = (
            (1 if (pilar_tec_valido and tec_dir == "BAJISTA") else 0)
            + (1 if (tiene_not and not_sent == "NEGATIVA") else 0)
            + (1 if (tiene_fund and rep_sent == "NEGATIVA") else 0)
            + (1 if (tiene_cong and cong_sent == "NEGATIVA") else 0)
            + (1 if (tiene_rip and rip_sent == "NEGATIVA") else 0)
        )
        alineacion_max = max(votos_alc, votos_baj)
        contradiccion = (votos_alc >= 1 and votos_baj >= 1) and min(votos_alc, votos_baj) >= 1

        if alineacion_max >= 2 and not contradiccion:
            score += 22.0 + (alineacion_max - 2) * 14.0
            razones.append(f"Confluencia {alineacion_max} pilares")
        elif contradiccion and not en_cartera:
            score -= 18.0

        # Protección de cartera abierta
        alerta_defensiva_cartera = False
        if en_cartera and pos_cartera is not None:
            opuesto = (
                (pos_cartera.tipo == "LONG" and votos_baj >= 2)
                or (pos_cartera.tipo == "SHORT" and votos_alc >= 2)
            )
            pnl_pct = 0.0
            if snap_tec is not None:
                p_act = snap_tec.datos_financieros.precio_actual
                if pos_cartera.precio_entrada > 0 and p_act > 0:
                    if pos_cartera.tipo == "LONG":
                        pnl_pct = ((p_act - pos_cartera.precio_entrada) / pos_cartera.precio_entrada) * 100.0
                    else:
                        pnl_pct = ((pos_cartera.precio_entrada - p_act) / pos_cartera.precio_entrada) * 100.0
            if opuesto:
                score += 18.0
                razones.append(f"Protección Cartera {pos_cartera.tipo} ({pnl_pct:+.1f}%)")
                if abs(pnl_pct) >= 2.5 or alineacion_max >= 2:
                    alerta_defensiva_cartera = True
        elif self.modo_operacion == "spot":
            if votos_baj > 0 and votos_alc == 0:
                score -= 45.0

        # Regla dura anti-day-trading por horario y pilares:
        num_pilares = len(pilares)
        min_pilares = int(reg.get("min_pilares", 2 if reg["es_permisivo"] else 3))
        umbral = float(reg["umbral_confluencia"])

        if reg["es_permisivo"]:
            # Ventanas flexibles (09:15-09:30, 09:30-09:50, 15:45-16:00 ET):
            # Exige mínimo 2 pilares alineados (nunca 1 noticia suelta).
            combinacion_valida = (alineacion_max >= min_pilares and num_pilares >= min_pilares) or alerta_defensiva_cartera
        else:
            # Resto de horarios (estricto):
            # Exige >= 3 pilares alineados sin contradicción, O combinación de certeza extrema:
            # Pilar TECNICO fuerte (patrón + RVOL>=1.35 + certeza>=74 + contra<=1) + Pilar FUNDAMENTAL/CONGRESO (imp>=74) alineados.
            combo_extremo = (
                pilar_tec_valido
                and tiene_patron
                and tec_rvol >= 1.35
                and tec_cert >= 74.0
                and tec_contra <= 1
                and alineacion_max >= 2
                and not contradiccion
                and ((tiene_fund and rep_score >= 74.0) or (tiene_cong and cong_score >= 72.0))
            )
            combinacion_valida = (
                (alineacion_max >= min_pilares and num_pilares >= min_pilares and not contradiccion)
                or combo_extremo
                or alerta_defensiva_cartera
            )

        supera = bool(combinacion_valida and score >= umbral)
        desc = " + ".join(razones[:3]) if razones else f"Pilares insuficientes ({num_pilares}/{min_pilares})"
        return round(max(0.0, min(200.0, score)), 1), desc, supera

    def encolar_por_noticia(self, noticia: NoticiaMercado) -> None:
        """
        Registra eventos de cualquiera de las 6 secciones de `analisis.py` (`NOTICIAS`,
        `FILINGS_8K`, `REPORTES_RESULTADOS`, `DIVULGACIONES_CONGRESO`, `FED`, `MACROECONOMIA`)
        en la matriz multi-pilar, propaga ripples sectoriales y filtra ruido de day trading.
        """
        if self.cartera.modo_cierre_activo is not None:
            return

        with self._lock:
            pausada = self._consultas_pausadas
            esperando_post_pausa = self._esperando_evento_post_reanudacion
            if pausada and self._offline_info is not None:
                self._offline_info["noticias_offline"][noticia.id_noticia] = noticia
                if noticia.ticker in self.indicadores.tickers:
                    self._offline_info["tickers_actualizados"].add(noticia.ticker)
                self._stats.noticias_durante_pausa = len(self._offline_info["noticias_offline"])

        reg = obtener_regimen_horario_mercado()
        ahora = time.time()
        seccion = getattr(noticia, "seccion", "NOTICIAS") or "NOTICIAS"
        subcat = getattr(noticia, "subcategoria", "") or seccion
        candidatos_potenciales: List[Tuple[str, float, str]] = []

        # Caso A: Evento específico de un ticker de la cartera/watchlist
        if noticia.ticker in self.indicadores.tickers:
            tk = noticia.ticker
            with self._lock:
                mem = self._senales_recientes.setdefault(tk, {})
                if seccion in ("FILINGS_8K", "REPORTES_RESULTADOS"):
                    mem["rep_ts"] = ahora
                    mem["rep_score"] = noticia.impacto_score
                    mem["rep_sent"] = noticia.sentimiento
                    mem["rep_sub"] = subcat
                elif seccion == "DIVULGACIONES_CONGRESO":
                    mem["cong_ts"] = ahora
                    mem["cong_score"] = noticia.impacto_score
                    mem["cong_sent"] = noticia.sentimiento
                    mem["cong_sub"] = subcat
                else:
                    mem["not_ts"] = ahora
                    mem["not_score"] = noticia.impacto_score
                    mem["not_sent"] = noticia.sentimiento
                    mem["not_plat"] = noticia.plataforma

            sectores_clave = [s for s in noticia.sectores_relacionados if s not in ("GENERAL", "MACRO_GLOBAL")]
            if noticia.impacto_score >= 68.0 and noticia.sentimiento != "NEUTRAL" and sectores_clave:
                pares_rip = self._propagar_ripple_sectorial(
                    sectores=sectores_clave,
                    sentimiento=noticia.sentimiento,
                    impacto=noticia.impacto_score * 0.90,
                    etiqueta_origen=f"Par[{tk}]/{seccion}",
                    excluir_ticker=tk,
                    reg=reg,
                    solo_registrar_senales=True,
                )
                candidatos_potenciales.extend(pares_rip)

            score, motivo, ok = self.calcular_puntaje_confluencia(tk, reg)
            if ok:
                candidatos_potenciales.insert(0, (tk, score, motivo))
            elif not pausada and not esperando_post_pausa:
                with self._lock:
                    self._stats.consultas_omitidas_gatekeeper += 1

        # Caso B: Macroeconomía, FED, Divulgación Congreso sin ticker en lista o 8-K sectorial
        elif noticia.impacto_score >= 60.0:
            sectores_obj = [s for s in noticia.sectores_relacionados if s not in ("GENERAL", "MACRO_GLOBAL")]
            if not sectores_obj:
                abiertas = list(self.cartera.posiciones.keys())
                candidatos_macro = abiertas[:4] if abiertas else self.indicadores.tickers[:4]
                for tk_m in candidatos_macro:
                    with self._lock:
                        mem = self._senales_recientes.setdefault(tk_m, {})
                        mem["rip_ts"] = ahora
                        mem["rip_score"] = noticia.impacto_score
                        mem["rip_sent"] = noticia.sentimiento
                        mem["rip_orig"] = f"{seccion}/{noticia.categoria_mercado}"
                    sc, mot, ok_m = self.calcular_puntaje_confluencia(tk_m, reg)
                    if ok_m:
                        candidatos_potenciales.append((tk_m, sc, mot))
            else:
                pares_sec = self._propagar_ripple_sectorial(
                    sectores=sectores_obj,
                    sentimiento=noticia.sentimiento,
                    impacto=noticia.impacto_score,
                    etiqueta_origen=f"{seccion}({subcat})",
                    excluir_ticker=None,
                    reg=reg,
                    solo_registrar_senales=True,
                )
                candidatos_potenciales.extend(pares_sec)

        if pausada or not candidatos_potenciales:
            return

        if esperando_post_pausa:
            es_noticia_direccional = noticia.sentimiento in ("POSITIVA", "NEGATIVA")
            if not es_noticia_direccional:
                return
            with self._lock:
                if self._consultas_pausadas:
                    return
                if self._esperando_evento_post_reanudacion:
                    self._esperando_evento_post_reanudacion = False
                    dir_lbl = "ALCISTA" if noticia.sentimiento == "POSITIVA" else "BAJISTA"
                    self._evento_despertar_post_pausa = (
                        f"Evento {seccion} potencialmente {dir_lbl} [{noticia.ticker}] "
                        f"({noticia.plataforma} | {noticia.sentimiento} | Imp:{noticia.impacto_score:.0f}): "
                        f"{noticia.titulo[:90]}"
                    )
                    self._stats.esperando_evento_reanudacion = False
                    self._stats.estado_agente = "ACTIVO (REACTIVADA)"

            for tk_c, sc_c, mot_c in candidatos_potenciales:
                self._registrar_candidato(
                    tk_c, sc_c, f"Reactivación Post-Offline: {mot_c}", autorizado_post_pausa=True
                )
            return

        for tk_c, sc_c, mot_c in candidatos_potenciales:
            self._registrar_candidato(tk_c, sc_c, mot_c)

    def _propagar_ripple_sectorial(
        self,
        sectores: List[str],
        sentimiento: str,
        impacto: float,
        etiqueta_origen: str,
        excluir_ticker: Optional[str],
        reg: Dict[str, Any],
        solo_registrar_senales: bool = False,
    ) -> List[Tuple[str, float, str]]:
        pares = obtener_tickers_relacionados_por_sectores(sectores, universe=self.indicadores.tickers)
        snapshots = self.indicadores.obtener_snapshots_todos()
        ahora = time.time()

        rank_pares: List[Tuple[float, str]] = []
        for tk_par in pares:
            if excluir_ticker and tk_par == excluir_ticker:
                continue
            with self._lock:
                mem = self._senales_recientes.setdefault(tk_par, {})
                mem["rip_ts"] = ahora
                mem["rip_score"] = impacto
                mem["rip_sent"] = sentimiento
                mem["rip_orig"] = etiqueta_origen

            snap_par = snapshots.get(tk_par)
            sesgo_tec = snap_par.potencial_score if snap_par is not None else 20.0
            if tk_par in self.cartera.posiciones:
                sesgo_tec += 20.0
            rank_pares.append((sesgo_tec, tk_par))

        rank_pares.sort(reverse=True)
        seleccionados: List[Tuple[str, float, str]] = []
        for _, tk_par in rank_pares:
            sc, mot, ok_par = self.calcular_puntaje_confluencia(tk_par, reg)
            if ok_par:
                seleccionados.append((tk_par, sc, mot))
                if not solo_registrar_senales:
                    self._registrar_candidato(tk_par, sc, mot)
                if len(seleccionados) >= 4:
                    break
        return seleccionados

    def encolar_por_indicadores(self, analisis: AnalisisTecnicoTicker) -> None:
        """
        Registra ruptura/patrón cuantitativo de `scanner.py` + volumen real RVOL y evalúa
        confluencia multi-pilar contra `analisis.py` y el régimen horario antes de encolar lote IA.
        """
        if self.cartera.modo_cierre_activo is not None:
            return

        tk = analisis.ticker
        rvol = self._extraer_rvol_de_analisis(analisis)
        n_alm = (
            len(analisis.alarmas_compra)
            if analisis.direccion_potencial == "ALCISTA"
            else (len(analisis.alarmas_venta) if analisis.direccion_potencial == "BAJISTA" else 0)
        )
        n_contra = (
            len(analisis.alarmas_venta)
            if analisis.direccion_potencial == "ALCISTA"
            else (len(analisis.alarmas_compra) if analisis.direccion_potencial == "BAJISTA" else 0)
        )
        patrones = list(getattr(analisis, "patrones_detectados", []))
        ahora = time.time()

        with self._lock:
            mem = self._senales_recientes.setdefault(tk, {})
            mem["tec_ts"] = ahora
            mem["tec_dir"] = analisis.direccion_potencial
            mem["tec_ev"] = analisis.evaluacion_general
            mem["tec_pot"] = analisis.potencial_score
            mem["tec_cert"] = analisis.certeza_pct
            mem["tec_rvol"] = rvol
            mem["tec_alm"] = n_alm
            mem["tec_contra"] = n_contra
            mem["tec_patrones"] = patrones

            # Si el escáner tiene reporte próximo inminente (<=7d) o recién publicado con sorpresa, reflejarlo en pilar fundamental
            rep = getattr(analisis.datos_financieros, "reporte_resultados", None)
            if rep is not None:
                if (rep.dias_hasta_reporte is not None and 0 <= rep.dias_hasta_reporte <= 7) or (
                    rep.evaluacion_reporte in ("POSITIVO", "NEGATIVO")
                ):
                    mem.setdefault("rep_ts", ahora)
                    mem.setdefault("rep_score", 70.0 if rep.evaluacion_reporte in ("POSITIVO", "NEGATIVO") else 66.0)
                    mem.setdefault(
                        "rep_sent",
                        "POSITIVA" if rep.evaluacion_reporte == "POSITIVO" else ("NEGATIVA" if rep.evaluacion_reporte == "NEGATIVO" else "NEUTRAL"),
                    )
                    mem.setdefault("rep_sub", f"Reporte Scanner ({rep.evaluacion_reporte})")

            pausada = self._consultas_pausadas
            esperando_post_pausa = self._esperando_evento_post_reanudacion

        if pausada:
            self._registrar_evento_indicador_offline(analisis, n_alm)
            return

        if not analisis.es_evento_alto_potencial and n_alm < 2 and not patrones:
            return

        reg = obtener_regimen_horario_mercado()
        score, motivo, ok = self.calcular_puntaje_confluencia(tk, reg)

        if esperando_post_pausa:
            dir_p = analisis.direccion_potencial.strip().upper()
            if dir_p not in ("ALCISTA", "BAJISTA") or not ok:
                return
            with self._lock:
                if self._consultas_pausadas:
                    return
                if self._esperando_evento_post_reanudacion:
                    self._esperando_evento_post_reanudacion = False
                    self._evento_despertar_post_pausa = (
                        f"Evento cuantitativo {dir_p} en [{tk}] "
                        f"({analisis.evaluacion_general}, Pot={analisis.potencial_score:.1f}/100, "
                        f"Cert={analisis.certeza_pct:.0f}%)"
                    )
                    self._stats.esperando_evento_reanudacion = False
                    self._stats.estado_agente = "ACTIVO (REACTIVADA)"
            self._registrar_candidato(
                tk, score, f"Reactivación Post-Offline({dir_p}): {motivo}", autorizado_post_pausa=True
            )
            return

        if ok:
            self._registrar_candidato(tk, score, motivo)
        else:
            with self._lock:
                self._stats.consultas_omitidas_gatekeeper += 1

    def _registrar_candidato(
        self,
        ticker: str,
        prioridad: float,
        origen: str,
        autorizado_post_pausa: bool = False,
    ) -> None:
        tk = ticker.strip().upper()
        ahora = time.time()
        with self._lock:
            if self._consultas_pausadas:
                return
            if self._esperando_evento_post_reanudacion and not autorizado_post_pausa:
                return
            existente = self._cola_candidatos.get(tk)
            if existente is None:
                self._cola_candidatos[tk] = (prioridad, origen, ahora)
            elif prioridad > existente[0]:
                self._cola_candidatos[tk] = (prioridad, origen, existente[2])
        self._evento_nuevo_candidato.set()

    def _pasa_filtro_cooldown_lote(
        self,
        ticker: str,
        prioridad: float,
        noticias_tk: List[NoticiaMercado],
        analisis_tec: AnalisisTecnicoTicker,
        regimen: Dict[str, Any],
        vuelve_de_offline: bool = False,
        info_offline: Optional[Dict[str, Any]] = None,
        es_reactivacion_post_pausa: bool = False,
    ) -> bool:
        """Verifica cooldown swing ampliado, cooldown extendido post-NEUTRAL y cambio de huella."""
        ahora = time.time()
        huella = self._calcular_huella_contexto(ticker, noticias_tk, analisis_tec)
        precio_act = analisis_tec.datos_financieros.precio_actual

        with self._lock:
            prev = self._ultimo_analisis_ticker.get(ticker)
            if prev is not None:
                ts_prev, huella_prev, precio_prev, fue_neutral = prev
                transcurrido = ahora - ts_prev
                cambio_precio_pct = (
                    abs(precio_act - precio_prev) / max(precio_prev, 1e-6) * 100.0
                    if precio_prev > 0
                    else 0.0
                )

                imp_max_not = max((n.impacto_score for n in noticias_tk[:2] if (ahora - n.epoch_publicacion) < 600), default=0.0)
                es_shock_urgente = (prioridad >= 115.0 and imp_max_not >= 80.0) or cambio_precio_pct >= 2.25

                if vuelve_de_offline and info_offline is not None:
                    ini_pausa_ep = float(info_offline.get("inicio_epoch", ahora))
                    actualizados = info_offline.get("tickers_actualizados", set())
                    if ts_prev < ini_pausa_ep and (
                        es_reactivacion_post_pausa
                        or huella != huella_prev
                        or cambio_precio_pct >= 0.45
                        or ticker in actualizados
                        or imp_max_not >= 68.0
                    ):
                        return True

                if huella == huella_prev and cambio_precio_pct < 1.1:
                    self._stats.consultas_omitidas_gatekeeper += 1
                    return False

                cooldown_req = (
                    float(regimen.get("cooldown_neutral_seg", COOLDOWN_NEUTRAL_EXTENDIDO_SEG))
                    if fue_neutral
                    else float(regimen.get("enfriamiento_base_seg", COOLDOWN_CONSULTA_TICKER_SEG))
                )
                if transcurrido < cooldown_req and not es_shock_urgente:
                    self._stats.consultas_omitidas_gatekeeper += 1
                    return False

            return True

    def _consumir_cupo_rate_limiter_lote(self, regimen: Optional[Dict[str, Any]] = None) -> bool:
        """Aplica tope estricto por minuto y tope por hora según régimen horario."""
        ahora = time.time()
        reg = regimen or obtener_regimen_horario_mercado()
        max_hora = int(reg.get("max_consultas_hora", MAX_CONSULTAS_AI_POR_HORA_ESTRICTA))
        with self._lock:
            while self._ventana_timestamps_minuto and (ahora - self._ventana_timestamps_minuto[0]) > 60.0:
                self._ventana_timestamps_minuto.popleft()
            while self._ventana_timestamps_hora and (ahora - self._ventana_timestamps_hora[0]) > 3600.0:
                self._ventana_timestamps_hora.popleft()

            if len(self._ventana_timestamps_minuto) >= MAX_CONSULTAS_AI_POR_MINUTO:
                return False
            if len(self._ventana_timestamps_hora) >= max_hora:
                return False

            self._ventana_timestamps_minuto.append(ahora)
            self._ventana_timestamps_hora.append(ahora)
            return True

    # ------------------------------------------------------------------
    # 2. CONSTRUCCIÓN DE PROMPTS (6 SECCIONES ANALISIS + 17 INDICADORES SCANNER)
    # ------------------------------------------------------------------

    def _formatear_bloque_empresa(
        self,
        ticker: str,
        origen_disparo: str,
        noticias_tk: List[NoticiaMercado],
        analisis_actual: AnalisisTecnicoTicker,
        snapshots_previos: List[AnalisisTecnicoTicker],
    ) -> str:
        df = analisis_actual.datos_financieros
        sectores = ", ".join(obtener_sectores_ticker(ticker))

        if noticias_tk:
            bloque_not_tk = "\n".join(
                f"    - [{n.fecha_hora_ampm}] [{getattr(n, 'seccion', 'NOTICIAS')}{('|' + n.subcategoria) if getattr(n, 'subcategoria', '') else ''}] "
                f"({n.plataforma} | {n.sentimiento} | Imp:{n.impacto_score:.0f}): {n.titulo}"
                + (f" — {n.resumen[:150]}" if n.resumen else "")
                for n in noticias_tk[:6]
            )
        else:
            bloque_not_tk = "    - Sin eventos directos nuevos para este ticker (revisar contexto sectorial/macro y scanner)."

        bloque_ind = "\n".join(
            f"    - {ind.categoria} | {ind.nombre}: {ind.texto_valor} -> {ind.evaluacion}"
            + (f" [ALARMA {ind.alarma}]" if ind.alarma else "")
            for ind in analisis_actual.indicadores
        )

        if snapshots_previos:
            bloque_prev = "\n".join(
                f"    - [{sp.fecha_hora_ampm}] ${sp.datos_financieros.precio_actual:.2f} | "
                f"Eval={sp.evaluacion_general} | Cert={sp.certeza_pct:.0f}% | Pot={sp.potencial_score:.1f}"
                for sp in snapshots_previos[-2:]
            )
        else:
            bloque_prev = "    - Primer corte técnico de la sesión."

        pe_s = f"{df.pe_ratio:.2f}" if df.pe_ratio is not None else "N/D"
        fpe_s = f"{df.pe_forward:.2f}" if df.pe_forward is not None else "N/D"
        mcap_s = f"${df.cap_mercado_b:.2f}B" if df.cap_mercado_b is not None else "N/D"
        r52 = (
            f"${df.min_52_semanas:.2f}-${df.max_52_semanas:.2f}"
            if (df.min_52_semanas is not None and df.max_52_semanas is not None)
            else "N/D"
        )

        rep = getattr(df, "reporte_resultados", None)
        if rep is not None:
            rep_txt = (
                f"Próx:{rep.fecha_proximo_reporte or 'N/D'} ({rep.dias_hasta_reporte if rep.dias_hasta_reporte is not None else '-'}d) | "
                f"Ing Esp:{formatear_monto_compacto(rep.ingresos_esperados)} vs Rep:{formatear_monto_compacto(rep.ingresos_reportados)} "
                f"(YoY:{f'{rep.ingresos_crecimiento_yoy_pct:+.1f}%' if rep.ingresos_crecimiento_yoy_pct is not None else 'N/D'}) | "
                f"EPS Esp:{rep.eps_esperado if rep.eps_esperado is not None else 'N/D'} / Rep:{rep.eps_reportado if rep.eps_reportado is not None else 'N/D'} "
                f"-> [{rep.evaluacion_reporte}]"
            )
        else:
            rep_txt = "Pendiente de actualización en caché fundamental."

        pats = ", ".join(getattr(analisis_actual, "patrones_detectados", [])) or "Ninguno activo"

        return (
            f"--- EMPRESA [{ticker}] | Sectores: {sectores} | Activación Gatekeeper: {origen_disparo} ---\n"
            f"  Datos Financieros: Precio=${df.precio_actual:.2f} ({df.cambio_diario_pct:+.2f}%) | "
            f"P/E={pe_s} (Fwd={fpe_s}) | Cap={mcap_s} | Beta={df.beta or 'N/D'} | "
            f"S1/S2=${df.soporte_1:.2f}/${df.soporte_2:.2f} | R1/R2=${df.resistencia_1:.2f}/${df.resistencia_2:.2f} | "
            f"VWAP=${df.vwap_sesion:.2f} | ATR=${df.atr_14:.3f} | 52s={r52} | "
            f"Vol={df.volumen_barra:,.0f}/Prom={df.volumen_promedio_20:,.0f}\n"
            f"  Reporte de Resultados (Scanner): {rep_txt}\n"
            f"  Consenso Scanner ({len(analisis_actual.indicadores)} Indicadores): {analisis_actual.evaluacion_general} "
            f"(Dir={analisis_actual.direccion_potencial}, Pot={analisis_actual.potencial_score:.1f}/100, "
            f"Certeza={analisis_actual.certeza_pct:.1f}%) | Patrones Swing: {pats}\n"
            f"  Indicadores y Patrones:\n{bloque_ind}\n"
            f"  Evolución Reciente Indicadores:\n{bloque_prev}\n"
            f"  Eventos de 6 Secciones [{ticker}] (Noticias / 8-K / Resultados / Congreso):\n{bloque_not_tk}"
        )

    def _obtener_snapshot_offline_para_prompt(self) -> Optional[Dict[str, Any]]:
        """Devuelve una copia segura del marco de contexto offline si está pendiente de entrega."""
        with self._lock:
            if self._offline_info is None or self._consultas_pausadas:
                return None
            return {
                "inicio_epoch": float(self._offline_info["inicio_epoch"]),
                "inicio_str": str(self._offline_info["inicio_str"]),
                "fin_epoch": float(self._offline_info["fin_epoch"] or time.time()),
                "fin_str": str(self._offline_info["fin_str"] or formatear_fecha_hora_ampm()),
                "segmentos_pausa": list(self._offline_info["segmentos_pausa"]),
                "patrimonio_inicio_usd": float(self._offline_info["patrimonio_inicio_usd"]),
                "efectivo_inicio_pct": float(self._offline_info["efectivo_inicio_pct"]),
                "posiciones_inicio": list(self._offline_info["posiciones_inicio"]),
                "ciclos_datos": int(self._offline_info["ciclos_datos"]),
                "snapshots_inicio": dict(self._offline_info["snapshots_inicio"]),
                "extremos_ciclo": dict(self._offline_info["extremos_ciclo"]),
                "eventos_indicadores": {
                    k: list(v) for k, v in self._offline_info["eventos_indicadores"].items()
                },
                "noticias_offline": dict(self._offline_info["noticias_offline"]),
                "tickers_actualizados": set(self._offline_info["tickers_actualizados"]),
                "evento_despertar": self._evento_despertar_post_pausa or "",
            }

    def _construir_marco_contexto_offline(
        self,
        lote_datos: List[Tuple[str, str, List[NoticiaMercado], AnalisisTecnicoTicker, List[AnalisisTecnicoTicker]]],
        info_offline: Dict[str, Any],
    ) -> str:
        """
        Construye el marco de contexto (datos financieros, indicadores y noticias) generado
        durante la interrupción offline para inyectarlo en la primera consulta al reanudar.
        """
        tickers_lote = [item[0] for item in lote_datos]
        tickers_lote_set = set(tickers_lote)
        ini_ep = float(info_offline["inicio_epoch"])
        fin_ep = float(info_offline["fin_epoch"])
        dur_str = _formatear_duracion_seg(fin_ep - ini_ep)
        ini_str = str(info_offline["inicio_str"])
        fin_str = str(info_offline["fin_str"])
        segs = info_offline.get("segmentos_pausa", [])
        seg_txt = f" ({len(segs)} tramos de pausa acumulados)" if len(segs) > 1 else ""

        pat_ini = float(info_offline.get("patrimonio_inicio_usd", self.cartera.capital_inicial_usd))
        pat_act = self.cartera.patrimonio_total_usd()
        var_pat_pct = ((pat_act - pat_ini) / max(pat_ini, 1e-6)) * 100.0
        ef_ini = float(info_offline.get("efectivo_inicio_pct", 100.0))
        ef_act = self.cartera.efectivo_disponible_pct()

        lineas: List[str] = [
            f"D) MARCO DE CONTEXTO GENERADO DURANTE INTERRUPCIÓN OFFLINE ({ini_str} -> {fin_str} | Duración: {dur_str}{seg_txt}):",
            f"  • Operación en segundo plano: InfoMarket, Indicadores y Cartera siguieron activos ({info_offline['ciclos_datos']} ciclos de mercado).",
            f"  • Cartera durante la interrupción: Patrimonio ${pat_ini:,.2f} -> ${pat_act:,.2f} ({var_pat_pct:+.2f}%) | Efectivo {ef_ini:.1f}% -> {ef_act:.1f}%.",
        ]
        despertar = str(info_offline.get("evento_despertar", "")).strip()
        if despertar:
            lineas.append(f"  • Evento Alcista/Bajista post-reanudación que activó esta consulta: {despertar}")

        # 1) Datos Financieros e Indicadores del Lote durante la interrupción
        snaps_ini: Dict[str, Dict[str, Any]] = info_offline.get("snapshots_inicio", {})
        extremos: Dict[str, Dict[str, Any]] = info_offline.get("extremos_ciclo", {})
        evs_ind: Dict[str, List[Dict[str, Any]]] = info_offline.get("eventos_indicadores", {})

        lineas.append("  [1. Evolución de Datos e Indicadores del Lote durante la Interrupción Offline]:")
        for tk, _, _, snap_act, _ in lote_datos:
            df = snap_act.datos_financieros
            p_act = float(df.precio_actual)
            rvol_act = self._extraer_rvol_de_analisis(snap_act)
            s_ini = snaps_ini.get(tk)
            ext = extremos.get(tk)

            if s_ini and float(s_ini.get("precio", 0.0)) > 0:
                p_ini = float(s_ini["precio"])
                var_p = ((p_act - p_ini) / p_ini) * 100.0
                tramo_p = f"${p_ini:.2f} -> ${p_act:.2f} ({var_p:+.2f}%)"
                eval_tr = f"{s_ini.get('evaluacion', 'NEUTRAL')}({float(s_ini.get('certeza', 30)):.0f}%) -> {snap_act.evaluacion_general}({snap_act.certeza_pct:.0f}%)"
                rvol_tr = f"RVOL {float(s_ini.get('rvol', 1.0)):.2f}x -> {rvol_act:.2f}x"
                inds_i = s_ini.get("ind_clave", {})
                ind_ini_s = ", ".join(f"{k}:{v}" for k, v in list(inds_i.items())[:3]) if inds_i else "N/D"
            else:
                tramo_p = f"${p_act:.2f} (sin corte previo al inicio de pausa)"
                eval_tr = f"{snap_act.evaluacion_general}({snap_act.certeza_pct:.0f}%)"
                rvol_tr = f"RVOL {rvol_act:.2f}x"
                ind_ini_s = "N/D"

            rango_ext = ""
            trans_s = ""
            if ext:
                p_mn = float(ext.get("min_precio", 0.0))
                p_mx = float(ext.get("max_precio", 0.0))
                if p_mn > 0 and p_mx > 0:
                    rango_ext = f" | Rango Pausa: [${p_mn:.2f}..${p_mx:.2f}]"
                trs = ext.get("transiciones", [])
                if trs:
                    trans_s = f" | Transiciones: {' | '.join(trs[-4:])}"

            lineas.append(
                f"    * [{tk}] Datos: {tramo_p}{rango_ext} | S1=${df.soporte_1:.2f} R1=${df.resistencia_1:.2f} VWAP=${df.vwap_sesion:.2f} | "
                f"Indicadores: {eval_tr} ({snap_act.direccion_potencial}) | {rvol_tr} | Base inicio: [{ind_ini_s}]{trans_s}"
            )

            evs_tk = evs_ind.get(tk, [])
            for ev_i in evs_tk[-3:]:
                alm_txt = ", ".join(ev_i.get("alarmas", [])[:3]) or "-"
                lineas.append(
                    f"      - Hito offline [{ev_i['hora']}]: ${ev_i['precio']:.2f} | {ev_i['evaluacion']} "
                    f"({ev_i['direccion']}, Pot={ev_i['potencial']:.0f}, Cert={ev_i['certeza']:.0f}%, RVOL={ev_i['rvol']:.2f}x) | Alarmas:[{alm_txt}]"
                )

        # 2) Cambios relevantes de otras acciones y eventos /d ocurridos offline
        snaps_todos = self.indicadores.obtener_snapshots_todos()
        movimientos_universo: List[Tuple[float, str, float, float, float, str, str]] = []
        for tk_u, s_i in snaps_ini.items():
            if tk_u in tickers_lote_set:
                continue
            p_i = float(s_i.get("precio", 0.0))
            if p_i <= 0:
                continue
            ext_u = extremos.get(tk_u, {})
            sn_u = snaps_todos.get(tk_u)
            p_f = float(sn_u.datos_financieros.precio_actual) if sn_u else float(ext_u.get("ultimo_precio", p_i))
            ev_f = sn_u.evaluacion_general if sn_u else str(ext_u.get("eval_ultimo", s_i.get("evaluacion", "NEUTRAL")))
            d_pct = ((p_f - p_i) / p_i) * 100.0
            ev_i = str(s_i.get("evaluacion", "NEUTRAL"))
            if abs(d_pct) >= 0.45 or ev_f != ev_i:
                movimientos_universo.append((abs(d_pct) + (2.0 if ev_f != ev_i else 0.0), tk_u, p_i, p_f, d_pct, ev_i, ev_f))

        movimientos_universo.sort(reverse=True)
        if movimientos_universo:
            res_movs = ", ".join(
                f"{tk_m}:${p1:.2f}->${p2:.2f}({dp:+.2f}%, {e1}->{e2})"
                for _, tk_m, p1, p2, dp, e1, e2 in movimientos_universo[:6]
            )
            lineas.append(f"  [2. Otros Movimientos Relevantes de Datos/Evaluación Offline en Watchlist]: {res_movs}")

        otros_evs: List[Tuple[float, str, Dict[str, Any]]] = []
        for tk_e, lista_e in evs_ind.items():
            if tk_e in tickers_lote_set:
                continue
            for ev_o in lista_e:
                otros_evs.append((float(ev_o.get("potencial", 0.0)), tk_e, ev_o))
        otros_evs.sort(key=lambda x: (x[0], x[2].get("epoch", 0.0)), reverse=True)
        if otros_evs:
            lineas.append("  [3. Eventos de Indicadores de Alto Potencial (/d) en Otras Empresas durante la Pausa]:")
            for _, tk_e, ev_o in otros_evs[:6]:
                alm_o = ", ".join(ev_o.get("alarmas", [])[:2]) or "-"
                lineas.append(
                    f"    - [{ev_o['hora']}] [{tk_e}] ${ev_o['precio']:.2f} | {ev_o['evaluacion']} "
                    f"({ev_o['direccion']}, Pot={ev_o['potencial']:.0f}, Cert={ev_o['certeza']:.0f}%) | Alarmas:[{alm_o}]"
                )

        # 4) Noticias generadas/capturadas durante la interrupción offline
        pool_not: Dict[str, NoticiaMercado] = dict(info_offline.get("noticias_offline", {}))
        for n_rng in self.infomarket.obtener_noticias_en_rango(
            ini_ep, fin_ep + 0.5, tickers_prioritarios=tickers_lote, max_items=24
        ):
            pool_not[n_rng.id_noticia] = n_rng

        # Priorizar noticias del lote, sectoriales e impacto alto, conservando orden cronológico al renderizar
        sectores_lote: Set[str] = set()
        for tk_l in tickers_lote:
            sectores_lote.update(obtener_sectores_ticker(tk_l))

        rank_not: List[Tuple[Tuple[int, int, int, float], NoticiaMercado]] = []
        for n_item in pool_not.values():
            en_lote = 1 if n_item.ticker in tickers_lote_set else 0
            en_sector = 1 if (set(n_item.sectores_relacionados) & sectores_lote) else 0
            es_dir = 1 if n_item.sentimiento in ("POSITIVA", "NEGATIVA") else 0
            ts_n = n_item.epoch_captura if n_item.epoch_captura > 0 else n_item.epoch_publicacion
            rank_not.append(((en_lote, en_sector, es_dir, float(n_item.impacto_score) + ts_n * 1e-12), n_item))

        rank_not.sort(key=lambda x: x[0], reverse=True)
        noticias_sel = [par[1] for par in rank_not[:16]]
        noticias_sel.sort(key=lambda n: n.epoch_captura if n.epoch_captura > 0 else n.epoch_publicacion)

        lineas.append(
            f"  [4. Noticias Capturadas durante la Interrupción Offline ({len(pool_not)} total, {len(noticias_sel)} destacadas)]:"
        )
        if noticias_sel:
            for n_off in noticias_sel:
                sec_lbl = getattr(n_off, "seccion", "NOTICIAS")
                sub_lbl = getattr(n_off, "subcategoria", "")
                etq_sec = f"{sec_lbl}/{sub_lbl}" if sub_lbl else sec_lbl
                lineas.append(
                    f"    - [{n_off.fecha_hora_ampm}] [{n_off.ticker}|{etq_sec}] "
                    f"({n_off.plataforma} | {n_off.sentimiento} | Imp:{n_off.impacto_score:.0f}): {n_off.titulo}"
                )
        else:
            lineas.append("    - Sin eventos nuevos adicionales capturados durante este intervalo offline.")

        lineas.append(
            "  NOTA DE CONTINUIDAD OFFLINE: Evalúa esta transición acumulada (datos, indicadores y eventos "
            "de la sección D) junto con el estado actual de C y el evento post-reanudación antes de decidir."
        )
        return "\n".join(lineas)

    def _construir_prompt_lote(
        self,
        lote_datos: List[Tuple[str, str, List[NoticiaMercado], AnalisisTecnicoTicker, List[AnalisisTecnicoTicker]]],
        noticias_generales: List[NoticiaMercado],
        regimen_horario: Dict[str, Any],
        info_offline: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construye el prompt de Consulta Múltiple con:
        1) Ventana unificada de Contexto General (FED, MACROECONOMIA, DIVULGACIONES_CONGRESO, Sectores).
        2) Estado porcentual actual de la Cartera Virtual.
        3) Ventana específica (17 indicadores + patrones + Reporte Resultados + eventos) por empresa.
        4) Marco de Contexto Offline si aplica.
        """
        tickers_str = ", ".join(item[0] for item in lote_datos)
        cartera_ctx = self.cartera.obtener_resumen_contexto_ia()

        if noticias_generales:
            bloque_general = "\n".join(
                f"  - [{m.fecha_hora_ampm}] [{getattr(m, 'seccion', m.categoria_mercado)}"
                f"{('|' + m.subcategoria) if getattr(m, 'subcategoria', '') else ''}] "
                f"({m.plataforma} | {m.sentimiento} | Sectores:{','.join(m.sectores_relacionados[:3])}): {m.titulo}"
                for m in noticias_generales[:12]
            )
        else:
            bloque_general = "  - Sin alertas macroeconómicas, Fed, Congreso o sectoriales extraordinarias en este corte."

        bloques_empresas = "\n\n".join(
            self._formatear_bloque_empresa(tk, orig, n_tk, snap_act, snaps_prev)
            for (tk, orig, n_tk, snap_act, snaps_prev) in lote_datos
        )

        off_snap = info_offline if info_offline is not None else self._obtener_snapshot_offline_para_prompt()
        bloque_offline = (
            "\n\n" + self._construir_marco_contexto_offline(lote_datos, off_snap)
            if off_snap is not None
            else ""
        )

        return (
            f"=== CONSULTA MÚLTIPLE SWING DE ALTA CONVICCIÓN VAGY-M1 ({len(lote_datos)} EMPRESAS EN LOTE) ===\n"
            f"Fecha/Hora Local: {formatear_fecha_hora_ampm()} | Régimen Horario Wall Street: {regimen_horario['etiqueta']} ({regimen_horario['hora_et']})\n"
            f"Tickers en esta Consulta Múltiple: [{tickers_str}]\n\n"
            f"A) VENTANA DE CONTEXTO GENERAL (FED/TASAS, MACROECONOMÍA/GEOPOLÍTICA/BÉLICO, CONGRESO, SECTORES):\n"
            f"{bloque_general}\n\n"
            f"B) ESTADO ACTUAL DE LA CARTERA VIRTUAL (BASE PORCENTUAL):\n"
            f"  {json.dumps(cartera_ctx, ensure_ascii=False)}\n\n"
            f"C) VENTANAS DE CONTEXTO ESPECÍFICAS POR EMPRESA (SCANNER 17 IND + REPORTES + 6 SECCIONES ANALISIS):\n"
            f"{bloques_empresas}{bloque_offline}\n\n"
            f"Analiza cada empresa en `decisiones` evaluando confluencia multi-pilar (sin day trading por ruido). "
            f"Marca `alto_potencial=true` SOLO en oportunidades claras (`certeza_pct >= {int(UMBRAL_CERTEZA_AI_OPERACION)}`); las demás déjalas `NEUTRAL`."
        )

    def _construir_prompt_completo(
        self,
        ticker: str,
        origen_disparo: str,
        noticias_tk: List[NoticiaMercado],
        noticias_macro: List[NoticiaMercado],
        analisis_actual: AnalisisTecnicoTicker,
        snapshots_previos: List[AnalisisTecnicoTicker],
    ) -> str:
        """Constructor individual (compatibilidad directa y pruebas unitarias)."""
        reg = obtener_regimen_horario_mercado()
        return self._construir_prompt_lote(
            lote_datos=[(ticker, origen_disparo, noticias_tk, analisis_actual, snapshots_previos)],
            noticias_generales=noticias_macro,
            regimen_horario=reg,
        )

    # ------------------------------------------------------------------
    # 3. INVOCACIÓN A PROVEEDORES IA, TOKENS Y COSTO USD
    # ------------------------------------------------------------------

    @staticmethod
    def _clasificar_error_api(exc: Exception) -> Tuple[str, str]:
        return diagnosticar_error_autenticacion(exc)

    @staticmethod
    def _extraer_json_limpio(texto: str) -> str:
        limpio = texto.strip()
        if limpio.startswith("```"):
            limpio = re.sub(r"^```(?:json)?\s*", "", limpio, flags=re.IGNORECASE)
            limpio = re.sub(r"\s*```$", "", limpio)
        inicio = limpio.find("{")
        fin = limpio.rfind("}")
        if inicio != -1 and fin != -1 and fin > inicio:
            return limpio[inicio : fin + 1]
        return limpio

    def _construir_uso_tokens(
        self,
        tokens_input: int,
        tokens_output: int,
        tokens_thought: int = 0,
        tokens_cached: int = 0,
        tokens_total: int = 0,
    ) -> Dict[str, Any]:
        in_t = max(0, int(tokens_input or 0))
        out_t = max(0, int(tokens_output or 0))
        th_t = max(0, int(tokens_thought or 0))
        ca_t = max(0, int(tokens_cached or 0))
        tot_t = max(in_t + out_t, int(tokens_total or 0))
        costo = calcular_costo_consulta_usd(
            self.modelo_clave,
            input_tokens=in_t,
            output_tokens=out_t,
            cached_tokens=ca_t,
            thought_tokens=th_t,
            total_tokens_reportados=tot_t,
        ) or 0.0
        uso = {
            "input": in_t,
            "output": out_t,
            "thought": th_t,
            "cached": ca_t,
            "total": tot_t,
            "costo_usd": costo,
        }
        with self._lock:
            self._ultimo_uso_tokens_invocacion = uso
        return uso

    def _extraer_uso_gemini_interaccion(self, interaccion: Any) -> Dict[str, Any]:
        usage = getattr(interaccion, "usage", None)
        if usage is None:
            return self._construir_uso_tokens(0, 0, 0, 0, 0)
        return self._construir_uso_tokens(
            tokens_input=int(getattr(usage, "total_input_tokens", 0) or 0),
            tokens_output=int(getattr(usage, "total_output_tokens", 0) or 0),
            tokens_thought=int(getattr(usage, "total_thought_tokens", 0) or 0),
            tokens_cached=int(getattr(usage, "total_cached_tokens", 0) or 0),
            tokens_total=int(getattr(usage, "total_tokens", 0) or 0),
        )

    def _extraer_uso_vertex(self, resp_vx: Any) -> Dict[str, Any]:
        meta = getattr(resp_vx, "usage_metadata", None)
        if meta is None:
            return self._construir_uso_tokens(0, 0, 0, 0, 0)
        return self._construir_uso_tokens(
            tokens_input=int(getattr(meta, "prompt_token_count", 0) or 0),
            tokens_output=int(getattr(meta, "candidates_token_count", 0) or 0),
            tokens_thought=int(getattr(meta, "thoughts_token_count", 0) or 0),
            tokens_cached=int(getattr(meta, "cached_content_token_count", 0) or 0),
            tokens_total=int(getattr(meta, "total_token_count", 0) or 0),
        )

    def _extraer_uso_anthropic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        usage = data.get("usage") if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        in_t = int(usage.get("input_tokens") or 0)
        cache_r = int(usage.get("cache_read_input_tokens") or 0)
        cache_w = int(usage.get("cache_creation_input_tokens") or 0)
        out_t = int(usage.get("output_tokens") or 0)
        in_total = in_t + cache_r + cache_w
        return self._construir_uso_tokens(
            tokens_input=in_total,
            tokens_output=out_t,
            tokens_thought=0,
            tokens_cached=cache_r,
            tokens_total=in_total + out_t,
        )

    def _invocar_json_modelo(self, prompt: str, esquema_cls: type[BaseModel], esquema_plano: Dict[str, Any]) -> Any:
        """Invoca el modelo configurado pasando `thinking_level` (Gemini) o `output_config.effort` (Anthropic) y registra uso de tokens."""
        if not self.api_key:
            raise PermissionError("API Key vacía o no proporcionada (401 insufficient credentials).")

        with self._lock:
            self._ultimo_uso_tokens_invocacion = {}
        modelos_intento = [self.model_id] + [m for m in self.fallback_ids if m != self.model_id]

        if self.proveedor == "gemini":
            if self._gemini_client is None:
                self._inicializar_cliente_gemini()
            if self._gemini_client is None:
                raise ConnectionError("No se pudo inicializar el cliente Gemini.")

            ultima_exc: Optional[Exception] = None
            for m_id in modelos_intento:
                try:
                    if self.modo_transporte == "vertex_express":
                        resp_vx = self._gemini_client.models.generate_content(
                            model=m_id,
                            contents=prompt,
                            config={
                                "system_instruction": SYSTEM_INSTRUCTION_FINANCIERO,
                                "response_mime_type": "application/json",
                                "response_json_schema": esquema_plano,
                                "thinking_config": {"thinking_level": self.nivel_esfuerzo},
                            },
                        )
                        texto_vx = resp_vx.text or ""
                        if not texto_vx.strip():
                            raise ConnectionError("Vertex AI Express devolvió respuesta vacía.")
                        self._extraer_uso_vertex(resp_vx)
                        return esquema_cls.model_validate_json(self._extraer_json_limpio(texto_vx))

                    # SDK moderno google-genai Interactions API (sin temperature/top_p/top_k/thinking_budget)
                    interaccion = self._gemini_client.interactions.create(
                        model=m_id,
                        system_instruction=SYSTEM_INSTRUCTION_FINANCIERO,
                        input=prompt,
                        store=False,
                        generation_config={"thinking_level": self.nivel_esfuerzo},
                        response_format={
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": esquema_plano,
                        },
                    )
                    texto_salida = getattr(interaccion, "output_text", None) or ""
                    if not texto_salida.strip():
                        for out_it in getattr(interaccion, "outputs", None) or []:
                            if getattr(out_it, "type", "") == "text" and getattr(out_it, "text", None):
                                texto_salida += str(out_it.text)
                    if not texto_salida.strip():
                        raise ConnectionError("El modelo Gemini devolvió respuesta vacía.")
                    self._extraer_uso_gemini_interaccion(interaccion)
                    return esquema_cls.model_validate_json(self._extraer_json_limpio(texto_salida))
                except Exception as exc_g:
                    ultima_exc = exc_g
                    if _es_error_modelo_no_encontrado(str(exc_g)) and m_id != modelos_intento[-1]:
                        continue
                    raise

            raise RuntimeError(str(ultima_exc or "Error desconocido en Gemini."))

        # Proveedor Anthropic (Claude Sonnet / Opus / Fable)
        esquema_json = json.dumps(esquema_plano, ensure_ascii=False)
        sistema_claude = (
            f"{SYSTEM_INSTRUCTION_FINANCIERO}\n\n"
            f"ESQUEMA JSON OBLIGATORIO DE SALIDA:\n{esquema_json}"
        )
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        max_toks = 4096 if self.nivel_esfuerzo in ("high", "xhigh", "max") else 2400
        ultimo_error: Optional[Exception] = None
        for m_id in modelos_intento:
            for usar_effort in (True, False):
                payload: Dict[str, Any] = {
                    "model": m_id,
                    "max_tokens": max_toks,
                    "system": sistema_claude,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if usar_effort:
                    payload["output_config"] = {"effort": self.nivel_esfuerzo}

                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=55.0,
                )
                if resp.status_code == 400 and usar_effort and ("output_config" in resp.text or "effort" in resp.text):
                    continue
                if resp.status_code == 404 and m_id != modelos_intento[-1]:
                    ultimo_error = RuntimeError(f"HTTP 404 en {m_id}")
                    break
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                data = resp.json()
                self._extraer_uso_anthropic(data)
                bloques = data.get("content", [])
                texto = "".join(b.get("text", "") for b in bloques if isinstance(b, dict))
                return esquema_cls.model_validate_json(self._extraer_json_limpio(texto))

        raise RuntimeError(str(ultimo_error or "Error desconocido en proveedor Anthropic."))

    def _invocar_modelo_ia(self, prompt: str) -> DecisionModeloIA:
        return self._invocar_json_modelo(prompt, DecisionModeloIA, obtener_esquema_decision_plano())

    def _invocar_modelo_lote_ia(self, prompt: str) -> DecisionLoteModeloIA:
        return self._invocar_json_modelo(prompt, DecisionLoteModeloIA, obtener_esquema_decision_lote_plano())

    # ------------------------------------------------------------------
    # 4. PROCESAMIENTO DE LOTE MÚLTIPLE, EJECUCIÓN Y TELEMETRÍA
    # ------------------------------------------------------------------

    def _ejecutar_decision_en_cartera(
        self,
        tk: str,
        pred: str,
        alto_potencial: bool,
        certeza_pct: float,
        accion_cartera: str,
        pct_asignar: float,
        liquidaciones_rotacion: List[LiquidacionRotacionIA],
        explicacion: str,
        df_precio: float,
        df_pe: Optional[float],
        df_s1: float,
        df_r1: float,
    ) -> str:
        resumen_operacion = "Sin operación en cartera (espera/observación)"
        if alto_potencial and certeza_pct >= UMBRAL_CERTEZA_AI_OPERACION and self.cartera.modo_cierre_activo is None:
            logs_op: List[str] = []

            if liquidaciones_rotacion and (
                accion_cartera == "ROTAR_CARTERA" or self.cartera.efectivo_disponible_pct() < pct_asignar
            ):
                logs_rot = self.cartera.ejecutar_rotacion_ia(
                    [item.model_dump() for item in liquidaciones_rotacion],
                    certeza_nueva_oportunidad=certeza_pct,
                    ticker_destino=tk,
                )
                logs_op.extend(logs_rot)

            pct_asig = min(max(5.0, pct_asignar), MAX_ASIGNACION_POR_POSICION_PCT)
            if accion_cartera == "CERRAR_POSICION":
                ok_c, msg_c = self.cartera.liquidar_posicion(
                    tk, 100.0, df_precio, f"Cierre dictaminado por IA: {explicacion[:100]}"
                )
                if ok_c:
                    logs_op.append(msg_c)
            elif pred == "ALCISTA" and accion_cartera in ("COMPRAR_LONG", "ROTAR_CARTERA"):
                _, msg_a = self.cartera.abrir_o_ajustar_posicion(
                    tk, "LONG", pct_asig, df_precio,
                    pe=df_pe, soporte=df_s1, resistencia=df_r1,
                    potencial_score=certeza_pct, motivo=explicacion,
                )
                logs_op.append(msg_a)
            elif pred == "BAJISTA" and accion_cartera in ("VENDER_SHORT", "ROTAR_CARTERA", "CERRAR_POSICION"):
                _, msg_v = self.cartera.abrir_o_ajustar_posicion(
                    tk, "SHORT", pct_asig, df_precio,
                    pe=df_pe, soporte=df_s1, resistencia=df_r1,
                    potencial_score=certeza_pct, motivo=explicacion,
                )
                logs_op.append(msg_v)

            if logs_op:
                resumen_operacion = " | ".join(logs_op)

        return resumen_operacion

    def analizar_lote_completo(
        self,
        candidatos: List[Tuple[str, str, float]],
    ) -> Optional[DecisionLoteModeloIA]:
        """
        Ejecuta el MÉTODO DE CONSULTA MÚLTIPLE para un lote de `(ticker, origen_disparo, prioridad)`:
        - Respeta modo pausado/offline sin consultar API.
        - Filtra enfriamientos individuales / post-neutral.
        - Recopila contexto específico + contexto general + contexto de interrupción offline si aplica.
        - Realiza UNA sola consulta al modelo IA y actualiza todos los tickers del lote.
        """
        if not candidatos:
            return None

        with self._lock:
            if self._consultas_pausadas:
                return None

        info_offline_snap = self._obtener_snapshot_offline_para_prompt()
        vuelve_de_offline = info_offline_snap is not None

        regimen = obtener_regimen_horario_mercado()
        tickers_ordenados: List[str] = []
        prioridades_map: Dict[str, Tuple[str, float]] = {}
        snapshots_map: Dict[str, Tuple[AnalisisTecnicoTicker, List[AnalisisTecnicoTicker]]] = {}

        # Paso 1: reunir snapshots y verificar cooldown individual antes de consumir cuota
        for tk_raw, orig, prio in candidatos:
            tk = tk_raw.strip().upper()
            if tk in prioridades_map:
                continue
            snap_actual, snaps_previos = self.indicadores.obtener_snapshot_e_historial(tk)
            if snap_actual is None:
                snap_actual = self.indicadores.consultar_ticker_puntual(tk)
            if snap_actual is None:
                continue

            noticias_tk_chk, _ = self.infomarket.obtener_contexto_ticker(tk, max_ticker=5, max_macro=4)
            if not self._pasa_filtro_cooldown_lote(
                tk,
                prio,
                noticias_tk_chk,
                snap_actual,
                regimen,
                vuelve_de_offline=vuelve_de_offline,
                info_offline=info_offline_snap,
                es_reactivacion_post_pausa=orig.startswith("Reactivación Post-Offline"),
            ):
                continue

            tickers_ordenados.append(tk)
            prioridades_map[tk] = (orig, prio)
            snapshots_map[tk] = (snap_actual, snaps_previos)

        if not tickers_ordenados:
            if vuelve_de_offline:
                with self._lock:
                    if self._offline_info is not None and not self._consultas_pausadas:
                        self._esperando_evento_post_reanudacion = True
                        self._stats.esperando_evento_reanudacion = True
                        self._stats.estado_agente = "ESPERANDO EVENTO ALZA/BAJA"
            return None

        with self._lock:
            if self._consultas_pausadas:
                return None

        if not self._consumir_cupo_rate_limiter_lote(regimen):
            with self._lock:
                self._stats.consultas_omitidas_gatekeeper += len(tickers_ordenados)
                if vuelve_de_offline and self._offline_info is not None and not self._consultas_pausadas:
                    self._esperando_evento_post_reanudacion = True
                    self._stats.esperando_evento_reanudacion = True
                    self._stats.estado_agente = "ESPERANDO EVENTO ALZA/BAJA"
            return None

        # Paso 2: obtener ventana específica de cada empresa + ventana unificada de las 6 secciones
        mapa_especificas, noticias_generales = self.infomarket.obtener_contexto_lote(
            tickers_ordenados, max_por_ticker=6, max_mercado_general=12
        )

        lote_datos: List[Tuple[str, str, List[NoticiaMercado], AnalisisTecnicoTicker, List[AnalisisTecnicoTicker]]] = []
        for tk in tickers_ordenados:
            snap_act, snaps_prev = snapshots_map[tk]
            orig, _ = prioridades_map[tk]
            lote_datos.append((tk, orig, mapa_especificas.get(tk, []), snap_act, snaps_prev))

        info_offline_snap = self._obtener_snapshot_offline_para_prompt()
        prompt = self._construir_prompt_lote(
            lote_datos, noticias_generales, regimen, info_offline=info_offline_snap
        )
        ahora = time.time()

        with self._lock:
            if self._consultas_pausadas:
                return None
            self._ultimo_uso_tokens_invocacion = {}
            self._stats.regimen_horario = str(regimen["etiqueta"])
            self._stats.ultimos_tickers_consulta = list(tickers_ordenados)
            self._stats.estado_agente = f"CONSULTANDO {len(tickers_ordenados)} EMP."

        # Paso 3: UNA única llamada al modelo para evaluar todo el lote
        try:
            resp_lote = self._invocar_modelo_lote_ia(prompt)
        except requests.exceptions.RequestException as exc_red:
            with self._lock:
                self._stats.total_consultas += 1
                self._stats.fallos_conexion += 1
                self._stats.estado_agente = "SIN RESPUESTA: PÉRDIDA DE CONEXIÓN"
                self._stats.detalle_alerta = str(exc_red)[:90]
            return None
        except Exception as exc_gen:
            tipo_err, detalle = self._clasificar_error_api(exc_gen)
            with self._lock:
                self._stats.total_consultas += 1
                if tipo_err == "FONDOS":
                    self._stats.fallos_fondos_cuota += 1
                    self._stats.estado_agente = "SIN RESPUESTA: FONDOS/CUOTA INSUFICIENTES"
                elif tipo_err == "CREDENCIAL":
                    self._stats.fallos_fondos_cuota += 1
                    self._stats.estado_agente = "SIN RESPUESTA: API KEY NO VÁLIDA (401)"
                else:
                    self._stats.fallos_conexion += 1
                    self._stats.estado_agente = "SIN RESPUESTA: PÉRDIDA DE CONEXIÓN"
                self._stats.detalle_alerta = detalle
            return None

        # Paso 4: registrar telemetría de tokens/costo + lote, cooldowns y contexto offline
        with self._lock:
            uso_cons = dict(self._ultimo_uso_tokens_invocacion or {})
        tok_in = int(uso_cons.get("input", 0) or 0)
        tok_out = int(uso_cons.get("output", 0) or 0)
        tok_th = int(uso_cons.get("thought", 0) or 0)
        tok_ca = int(uso_cons.get("cached", 0) or 0)
        tok_tot = int(uso_cons.get("total", 0) or 0)
        costo_usd = float(uso_cons.get("costo_usd", 0.0) or 0.0)

        self.cartera.panorama_ia_global = resp_lote.panorama_mercado_global
        dec_por_tk: Dict[str, DecisionTickerLote] = {
            d.ticker.strip().upper(): d for d in resp_lote.decisiones
        }

        cnt_alc = 0
        cnt_baj = 0
        cnt_neu = 0
        with self._lock:
            self._stats.total_consultas += 1
            self._stats.total_activos_evaluados += len(tickers_ordenados)
            if len(tickers_ordenados) > 1:
                self._stats.consultas_ahorradas_lote += len(tickers_ordenados) - 1

            self._stats.tokens_ultima_input = tok_in
            self._stats.tokens_ultima_output = tok_out
            self._stats.tokens_ultima_thought = tok_th
            self._stats.tokens_ultima_cached = tok_ca
            self._stats.tokens_ultima_total = tok_tot
            self._stats.costo_ultima_usd = round(costo_usd, 6)

            self._stats.tokens_acum_input += tok_in
            self._stats.tokens_acum_output += tok_out
            self._stats.tokens_acum_thought += tok_th
            self._stats.tokens_acum_cached += tok_ca
            self._stats.tokens_acum_total += tok_tot
            self._stats.costo_acumulado_usd = round(self._stats.costo_acumulado_usd + costo_usd, 6)

            self._stats.estado_agente = "PAUSADA (OFFLINE)" if self._consultas_pausadas else "ACTIVO"
            self._stats.detalle_alerta = ""

            if info_offline_snap is not None:
                self._offline_info = None
                self._evento_despertar_post_pausa = None
                self._esperando_evento_post_reanudacion = False
                self._stats.esperando_evento_reanudacion = False
                self._stats.evento_offline_pendiente = False
                self._stats.noticias_durante_pausa = 0
                self._stats.eventos_tec_durante_pausa = 0
                self._stats.inicio_pausa_str = ""

        if info_offline_snap is not None:
            try:
                self.historial.actualizar_estado_sesion({
                    "ia_pausada": self._consultas_pausadas,
                    "ia_esperando_evento": False,
                })
            except Exception:
                pass

        eventos_alto_potencial: List[EventoImpresionPrincipal] = []

        for tk in tickers_ordenados:
            snap_act, _ = snapshots_map[tk]
            n_tk = mapa_especificas.get(tk, [])
            huella = self._calcular_huella_contexto(tk, n_tk, snap_act)
            dec = dec_por_tk.get(tk)

            if dec is None:
                cnt_neu += 1
                with self._lock:
                    self._ultimo_analisis_ticker[tk] = (ahora, huella, snap_act.datos_financieros.precio_actual, True)
                continue

            pred = dec.prediccion_movimiento.strip().upper()
            es_alto = bool(dec.alto_potencial and pred in ("ALCISTA", "BAJISTA") and dec.certeza_pct >= UMBRAL_CERTEZA_AI_OPERACION)
            fue_neutral = not es_alto

            with self._lock:
                self._ultimo_analisis_ticker[tk] = (ahora, huella, snap_act.datos_financieros.precio_actual, fue_neutral)

            if es_alto and pred == "ALCISTA":
                cnt_alc += 1
            elif es_alto and pred == "BAJISTA":
                cnt_baj += 1
            else:
                cnt_neu += 1

            df = snap_act.datos_financieros
            resumen_operacion = self._ejecutar_decision_en_cartera(
                tk=tk,
                pred=pred,
                alto_potencial=es_alto,
                certeza_pct=dec.certeza_pct,
                accion_cartera=dec.accion_cartera,
                pct_asignar=dec.porcentaje_capital_asignar,
                liquidaciones_rotacion=dec.liquidaciones_rotacion,
                explicacion=dec.explicacion_ejecutiva,
                df_precio=df.precio_actual,
                df_pe=df.pe_ratio,
                df_s1=df.soporte_1,
                df_r1=df.resistencia_1,
            )

            self.historial.registrar_consulta_ai({
                "fecha_hora": formatear_fecha_hora_ampm(),
                "ticker": tk,
                "lote_tickers": tickers_ordenados,
                "modelo": self.model_id,
                "effort": self.nivel_esfuerzo,
                "alto_potencial": es_alto,
                "prediccion": pred,
                "certeza_pct": dec.certeza_pct,
                "accion_cartera": dec.accion_cartera,
                "ejecucion": resumen_operacion,
                "post_offline": bool(info_offline_snap is not None),
                "tokens_input": tok_in,
                "tokens_output": tok_out,
                "tokens_thought": tok_th,
                "tokens_cached": tok_ca,
                "tokens_total": tok_tot,
                "costo_usd": costo_usd,
            })

            # REGLA DE VISUALIZACIÓN: Solo imprimir bloque en la terminal principal si tiene ALTO POTENCIAL
            if es_alto and self.callback_alto_potencial is not None:
                noticias_comb = n_tk[:4] + noticias_generales[:3]
                texto_resp = (
                    f"{dec.explicacion_ejecutiva}\n"
                    f"  [Contexto Macro Lote: {resp_lote.resumen_macro_sectorial} | "
                    f"Certeza IA: {dec.certeza_pct:.1f}% | Acción: {dec.accion_cartera} ({dec.porcentaje_capital_asignar:.1f}% cap.) | "
                    f"Panorama: {resp_lote.panorama_mercado_global}]\n"
                    f"  [Ejecución Cartera Virtual]: {resumen_operacion}"
                )
                eventos_alto_potencial.append(
                    EventoImpresionPrincipal(
                        fecha_hora_ampm=formatear_fecha_hora_ampm(),
                        ticker=tk,
                        prediccion_movimiento=pred,
                        certeza_pct=dec.certeza_pct,
                        prompt_enviado=prompt,
                        respuesta_modelo_texto=texto_resp,
                        accion_ejecutada_cartera=resumen_operacion,
                        noticias_analizadas=noticias_comb,
                        analisis_tecnico=snap_act,
                        tokens_consulta_input=tok_in,
                        tokens_consulta_output=tok_out,
                        tokens_consulta_thought=tok_th,
                        tokens_consulta_total=tok_tot,
                        costo_consulta_usd=costo_usd,
                    )
                )

        with self._lock:
            self._stats.respuestas_alcistas += cnt_alc
            self._stats.respuestas_bajistas += cnt_baj
            self._stats.respuestas_neutrales += cnt_neu

        for ev_ui in eventos_alto_potencial:
            try:
                if self.callback_alto_potencial is not None:
                    self.callback_alto_potencial(ev_ui)
            except Exception:
                pass

        return resp_lote

    def analizar_ticker_completo(
        self,
        ticker: str,
        origen_disparo: str,
        prioridad: float = 80.0,
    ) -> Optional[DecisionModeloIA]:
        """Wrapper compatible de análisis único apoyado en la tubería común."""
        res_lote = self.analizar_lote_completo([(ticker, origen_disparo, prioridad)])
        if res_lote is None or not res_lote.decisiones:
            return None
        d0 = res_lote.decisiones[0]
        return DecisionModeloIA(
            alto_potencial=d0.alto_potencial,
            prediccion_movimiento=d0.prediccion_movimiento,
            certeza_pct=d0.certeza_pct,
            accion_cartera=d0.accion_cartera,
            porcentaje_capital_asignar=d0.porcentaje_capital_asignar,
            liquidaciones_rotacion=d0.liquidaciones_rotacion,
            panorama_mercado=res_lote.panorama_mercado_global,
            explicacion_ejecutiva=d0.explicacion_ejecutiva,
        )

    def evaluar_panorama_cierre_perdidas(self, tickers_en_perdida: List[str]) -> Dict[str, str]:
        """Consultado en `/quit` opciones 1 y 2 para evaluar si esperar recuperación o liquidar pérdidas."""
        resultado: Dict[str, str] = {}
        if not tickers_en_perdida or not self.api_key:
            return {tk: self.cartera.panorama_ia_global for tk in tickers_en_perdida}

        for tk in tickers_en_perdida[:3]:
            snap, _ = self.indicadores.obtener_snapshot_e_historial(tk)
            if snap is None:
                resultado[tk] = self.cartera.panorama_ia_global
                continue
            pos = self.cartera.posiciones.get(tk)
            if snap.evaluacion_general == "VENTA FUERTE" and pos and pos.tipo == "LONG":
                resultado[tk] = "MALO"
            elif snap.evaluacion_general == "COMPRA FUERTE" and pos and pos.tipo == "SHORT":
                resultado[tk] = "MALO"
            else:
                resultado[tk] = self.cartera.panorama_ia_global
        return resultado

    # ------------------------------------------------------------------
    # 5. BUCLE WORKER DE AGRUPACIÓN MULTI-TICKER Y SUPERVISOR
    # ------------------------------------------------------------------

    def iniciar_worker(self) -> None:
        if self._hilo_worker and self._hilo_worker.is_alive():
            return
        self._detener.clear()
        self._hilo_worker = threading.Thread(
            target=self._bucle_cola,
            name="AnalisisAIWorker",
            daemon=True,
        )
        self._hilo_worker.start()

    def detener(self) -> None:
        self._detener.set()
        self._evento_nuevo_candidato.set()

    def _extraer_lote_sector_afines(self) -> List[Tuple[str, str, float]]:
        """
        Extrae hasta `MAX_TICKERS_LOTE_AI` candidatos de la cola, priorizando agrupar
        empresas del mismo sector alrededor del candidato principal para análisis comparativo en lote.
        """
        with self._lock:
            if not self._cola_candidatos:
                return []

            ordenados = sorted(
                self._cola_candidatos.items(),
                key=lambda kv: kv[1][0],
                reverse=True,
            )
            lider_tk, (lider_prio, lider_orig, _) = ordenados[0]
            sectores_lider: Set[str] = set(obtener_sectores_ticker(lider_tk))

            elegidos: List[Tuple[str, str, float]] = [(lider_tk, lider_orig, lider_prio)]
            self._cola_candidatos.pop(lider_tk, None)

            afines: List[Tuple[str, str, float]] = []
            otros: List[Tuple[str, str, float]] = []
            for tk, (prio, orig, _) in ordenados[1:]:
                if sectores_lider & set(obtener_sectores_ticker(tk)):
                    afines.append((tk, orig, prio))
                else:
                    otros.append((tk, orig, prio))

            for tk, orig, prio in afines + otros:
                if len(elegidos) >= MAX_TICKERS_LOTE_AI:
                    break
                elegidos.append((tk, orig, prio))
                self._cola_candidatos.pop(tk, None)

            return elegidos

    def _bucle_cola(self) -> None:
        while not self._detener.is_set():
            self._evento_nuevo_candidato.wait(timeout=4.0)
            self._evento_nuevo_candidato.clear()
            if self._detener.is_set():
                break

            reg = obtener_regimen_horario_mercado()
            ventana_lote = float(reg.get("ventana_lote_seg", 18.0))
            t_inicio_espera = time.time()

            while not self._detener.is_set():
                with self._lock:
                    if self._consultas_pausadas or self._esperando_evento_post_reanudacion:
                        self._cola_candidatos.clear()
                        n_cola = 0
                    else:
                        n_cola = len(self._cola_candidatos)
                    max_prio = max((v[0] for v in self._cola_candidatos.values()), default=0.0)
                    ts_mas_viejo = min((v[2] for v in self._cola_candidatos.values()), default=time.time())

                if n_cola == 0:
                    break

                transcurrido_cola = time.time() - ts_mas_viejo
                listo_lote = (
                    n_cola >= MAX_TICKERS_LOTE_AI
                    or (n_cola >= MIN_TICKERS_LOTE_AI and (max_prio >= 112.0 or transcurrido_cola >= ventana_lote * 0.65))
                    or (time.time() - t_inicio_espera >= ventana_lote)
                )

                if not listo_lote:
                    time.sleep(0.45)
                    continue

                lote = self._extraer_lote_sector_afines()
                if not lote:
                    break
                try:
                    self.analizar_lote_completo(lote)
                except Exception:
                    pass
                t_inicio_espera = time.time()
                time.sleep(1.5)

    def obtener_estadisticas_supervisor(self) -> Dict[str, Any]:
        """Devuelve telemetría en vivo (incluyendo tokens y costo USD) para `Supervisión AI`."""
        with self._lock:
            reg = obtener_regimen_horario_mercado()
            self._stats.regimen_horario = str(reg["etiqueta"])
            tot_calls = self._stats.total_consultas
            tot_evals = max(
                1,
                self._stats.respuestas_alcistas + self._stats.respuestas_bajistas + self._stats.respuestas_neutrales,
            ) if (self._stats.respuestas_alcistas + self._stats.respuestas_bajistas + self._stats.respuestas_neutrales) > 0 else 0

            pct_alc = (self._stats.respuestas_alcistas / tot_evals * 100.0) if tot_evals > 0 else 0.0
            pct_baj = (self._stats.respuestas_bajistas / tot_evals * 100.0) if tot_evals > 0 else 0.0
            pct_neu = (self._stats.respuestas_neutrales / tot_evals * 100.0) if tot_evals > 0 else 0.0

            return {
                "total_consultas": tot_calls,
                "total_activos_evaluados": self._stats.total_activos_evaluados,
                "alcistas": self._stats.respuestas_alcistas,
                "alcistas_pct": round(pct_alc, 1),
                "bajistas": self._stats.respuestas_bajistas,
                "bajistas_pct": round(pct_baj, 1),
                "neutrales": self._stats.respuestas_neutrales,
                "neutrales_pct": round(pct_neu, 1),
                "fallos_conexion": self._stats.fallos_conexion,
                "fallos_fondos": self._stats.fallos_fondos_cuota,
                "estado_agente": self._stats.estado_agente,
                "detalle_alerta": self._stats.detalle_alerta,
                "filtradas_gatekeeper": self._stats.consultas_omitidas_gatekeeper,
                "ahorradas_lote": self._stats.consultas_ahorradas_lote,
                "en_cola": len(self._cola_candidatos),
                "tickers_cola": list(self._cola_candidatos.keys()),
                "ultimos_tickers_consulta": list(self._stats.ultimos_tickers_consulta),
                "regimen_horario": self._stats.regimen_horario,
                "esfuerzo": self.nivel_esfuerzo,
                "modelo_clave": self.modelo_clave,
                "consultas_pausadas": self._consultas_pausadas,
                "esperando_evento_reanudacion": self._esperando_evento_post_reanudacion,
                "evento_offline_pendiente": self._offline_info is not None,
                "inicio_pausa_str": self._stats.inicio_pausa_str,
                "noticias_durante_pausa": self._stats.noticias_durante_pausa,
                "eventos_tec_durante_pausa": self._stats.eventos_tec_durante_pausa,
                "total_pausas": self._stats.total_pausas,
                "tokens_ultima_input": self._stats.tokens_ultima_input,
                "tokens_ultima_output": self._stats.tokens_ultima_output,
                "tokens_ultima_thought": self._stats.tokens_ultima_thought,
                "tokens_ultima_cached": self._stats.tokens_ultima_cached,
                "tokens_ultima_total": self._stats.tokens_ultima_total,
                "tokens_acum_input": self._stats.tokens_acum_input,
                "tokens_acum_output": self._stats.tokens_acum_output,
                "tokens_acum_thought": self._stats.tokens_acum_thought,
                "tokens_acum_cached": self._stats.tokens_acum_cached,
                "tokens_acum_total": self._stats.tokens_acum_total,
                "costo_ultima_usd": self._stats.costo_ultima_usd,
                "costo_acumulado_usd": self._stats.costo_acumulado_usd,
            }
