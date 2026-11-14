"""
Módulo analisis_ai.py
Segundo y Tercer Nivel del Sistema de Monitoreo Bursátil:

- NIVEL 2 (Filtro de Correlación Local y Contexto):
  Enlaza las noticias (`infomarket.py`), los datos fundamentales/flujo de mercado y los
  indicadores técnicos con filtro anti-errático (`indicadores.py`).
  Cuando detecta una combinación válida de eventos (Noticias + Datos + Indicadores),
  genera el objeto `evento_potencial_para_analisis` junto con su `contexto` histórico
  completo (todas las noticias previas de esa acción, noticias macro/Fed y todos los
  eventos técnicos previos y actuales de esa empresa).

- NIVEL 3 (Enlace con Modelos de IA):
  Envía el `evento_potencial_para_analisis` y su `contexto` a un modelo de IA
  (Google Gemini: `gemini-3.8-flash`, `gemini-4-argon`, `gemini-4-carbon`, o
  Anthropic Claude: `claude-sonnet-4-6`, `claude-opus-4-6`) usando el SDK oficial actual
  `google-genai` (`client.interactions.create`) para analizar a profundidad si las
  noticias son positivas o negativas, su impacto fundamental en la acción y la validez
  conjunta con los datos e indicadores.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import colorama
import requests

from indicadores import (
    GestorIndicadores,
    InstantaneaEventoTecnico,
    resaltar_turquesa,
)
from infomarket import (
    MAPA_NOMBRES_EMPRESAS,
    GestorInfoMarket,
    Noticia,
    formatear_fecha,
    resaltar_azul,
)

colorama.init(autoreset=True)


# =====================================================================
# COLORES ANSI PARA EL MÓDULO DE ANÁLISIS Y EVENTO POTENCIAL
# =====================================================================
class ColoresAI:
    RESET = "\033[0m"
    # Resaltado dorado/ámbar para identificar "evento_potencial_para_analisis"
    EVENTO_ANALISIS_BG = "\033[1;30;48;5;214m"
    # Resaltado índigo/violeta para la respuesta del Modelo de IA
    MODELO_AI_BG = "\033[1;97;48;5;63m"
    VERDE_BG = "\033[1;97;42m"
    ROJO_BG = "\033[1;97;41m"
    CIAN = "\033[1;36m"
    AMARILLO = "\033[1;33m"


# =====================================================================
# CATÁLOGO DE MODELOS DE IA SOPORTADOS
# =====================================================================
CATALOGO_MODELOS_AI: Dict[str, Dict[str, str]] = {
    # Modelos Google Gemini (SDK actual `google-genai` -> client.interactions.create)
    "flash": {
        "proveedor": "gemini",
        "model_id": "gemini-3.8-flash",
        "descripcion": "Gemini 3.8 Flash (1M tokens, rápido y balanceado para análisis en tiempo real)",
    },
    "argon": {
        "proveedor": "gemini",
        "model_id": "gemini-4-argon",
        "descripcion": "Gemini 4 Argon (1M tokens, razonamiento avanzado eficiente para bucles de producción)",
    },
    "carbon": {
        "proveedor": "gemini",
        "model_id": "gemini-4-carbon",
        "descripcion": "Gemini 4 Carbon (1M tokens, máxima capacidad de razonamiento profundo)",
    },
    "hydrogen": {
        "proveedor": "gemini",
        "model_id": "gemini-3.5-flash-lite",
        "descripcion": "Gemini 3.5 Flash-Lite (ultra-rápido y ligero para alta frecuencia)",
    },
    # Modelos Anthropic Claude
    "sonnet": {
        "proveedor": "claude",
        "model_id": "claude-sonnet-4-6",
        "descripcion": "Claude Sonnet 4.6 (rápido y preciso para análisis financiero)",
    },
    "opus": {
        "proveedor": "claude",
        "model_id": "claude-opus-4-6",
        "descripcion": "Claude Opus 4.6 (razonamiento profundo multimodal y financiero)",
    },
}


# =====================================================================
# ESTRUCTURAS DE DATOS: CONTEXTO Y EVENTO_POTENCIAL_PARA_ANALISIS
# =====================================================================
@dataclass
class ContextoAccion:
    """
    Agrupa toda la información histórica y actual de la acción específica que generó el evento:
    - Noticias previas y actuales de la acción (con pre-clasificación léxica de sentimiento).
    - Noticias macroeconómicas / Fed / geopolíticas relevantes del periodo.
    - Historial cronológico de todos los eventos técnicos previos de la acción.
    - Datos actuales de mercado, flujo de volumen (FRVP), estabilidad anti-errática, P/E y S&R.
    """
    simbolo: str
    nombre_empresa: str
    noticias_accion: List[Dict[str, Any]]
    noticias_macro_contexto: List[Dict[str, Any]]
    eventos_tecnicos_previos: List[Dict[str, Any]]
    datos_mercado_actuales: Dict[str, Any]
    indicadores_actuales: List[Dict[str, Any]]
    evaluacion_tecnica_actual: Dict[str, Any]

    def a_dict(self) -> Dict[str, Any]:
        return {
            "simbolo": self.simbolo,
            "nombre_empresa": self.nombre_empresa,
            "resumen_historial": {
                "total_noticias_accion": len(self.noticias_accion),
                "total_noticias_macro": len(self.noticias_macro_contexto),
                "total_eventos_tecnicos_previos": len(self.eventos_tecnicos_previos),
            },
            "noticias_accion": self.noticias_accion,
            "noticias_macro_contexto": self.noticias_macro_contexto,
            "eventos_tecnicos_previos": self.eventos_tecnicos_previos,
            "datos_mercado_actuales": self.datos_mercado_actuales,
            "indicadores_actuales": self.indicadores_actuales,
            "evaluacion_tecnica_actual": self.evaluacion_tecnica_actual,
        }


@dataclass
class EventoPotencialParaAnalisis:
    """
    Representa el `evento_potencial_para_analisis` generado en el Segundo Nivel
    cuando convergen Noticias + Datos de Mercado + Indicadores Técnicos estables.
    """
    id_evento: str
    timestamp: str
    ciclo: int
    simbolo: str
    tipo_combinacion: str
    direccion_preliminar: str  # "compra" o "venta"
    confianza_local_pct: float
    certeza_estabilidad_pct: float
    sentimiento_noticias_local: str
    resumen_catalizador: str
    contexto: ContextoAccion
    dictamen_ai: Optional[Dict[str, Any]] = None

    def a_dict(self) -> Dict[str, Any]:
        return {
            "evento": "evento_potencial_para_analisis",
            "id_evento": self.id_evento,
            "timestamp": self.timestamp,
            "ciclo": self.ciclo,
            "simbolo": self.simbolo,
            "tipo_combinacion": self.tipo_combinacion,
            "direccion_preliminar": self.direccion_preliminar,
            "confianza_local_pct": round(self.confianza_local_pct, 1),
            "certeza_estabilidad_pct": round(self.certeza_estabilidad_pct, 1),
            "sentimiento_noticias_local": self.sentimiento_noticias_local,
            "resumen_catalizador": self.resumen_catalizador,
            "contexto": self.contexto.a_dict(),
            "dictamen_ai": self.dictamen_ai,
        }


# =====================================================================
# PRE-ANALIZADOR LÉXICO LOCAL DE SENTIMIENTO DE NOTICIAS
# =====================================================================
PALABRAS_POSITIVAS = {
    "beat", "beats", "surge", "surges", "soars", "jump", "jumps", "upgrade",
    "upgrades", "buy", "outperform", "bullish", "record", "growth", "profit",
    "contract", "award", "partnership", "ai", "launch", "launches", "approval",
    "dividend", "buyback", "raises", "guidance", "strong", "rally", "breakthrough",
    "cut", "rate cut", "expansion", "deal", "acquires", "positive", "supera",
    "sube", "crecimiento", "récord", "acuerdo", "nuevo", "procesadores",
}

PALABRAS_NEGATIVAS = {
    "miss", "misses", "drop", "drops", "plunge", "plunges", "fall", "falls",
    "downgrade", "downgrades", "sell", "underperform", "bearish", "loss",
    "lawsuit", "investigation", "probe", "recall", "delay", "cuts guidance",
    "slump", "crash", "warning", "weak", "tariff", "sanctions", "war",
    "conflict", "tension", "inflation", "rate hike", "default", "layoffs",
    "cae", "baja", "pérdida", "tensión", "riesgo", "investigación", "recorte",
}


def estimar_sentimiento_titular(titulo: str) -> Tuple[str, int]:
    """
    Evalúa de forma local rápida si el titular tiene sesgo positivo, negativo o neutro
    para enriquecer el filtro de Nivel 2 antes de enviarlo al modelo de IA.
    """
    if not titulo:
        return "neutro", 0
    t_low = titulo.lower()
    tokens = set(re.findall(r"[a-z0-9áéíóúñ]+", t_low))
    pos = len(tokens.intersection(PALABRAS_POSITIVAS))
    neg = len(tokens.intersection(PALABRAS_NEGATIVAS))
    for frase_pos in ("rate cut", "raises guidance", "beats estimates"):
        if frase_pos in t_low:
            pos += 2
    for frase_neg in ("rate hike", "cuts guidance", "misses estimates", "middle east"):
        if frase_neg in t_low:
            neg += 2

    balance = pos - neg
    if balance >= 1:
        return "positivo", balance
    if balance <= -1:
        return "negativo", balance
    return "neutro", 0


# =====================================================================
# NIVEL 3: CONECTOR CON MODELOS DE IA (GEMINI / CLAUDE)
# =====================================================================
ESQUEMA_RESPUESTA_AI: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "simbolo": {"type": "string"},
        "sentimiento_noticias": {
            "type": "string",
            "enum": ["POSITIVO", "NEGATIVO", "NEUTRO", "MIXTO"],
        },
        "impacto_noticias_en_accion": {"type": "string"},
        "evaluacion_datos_e_indicadores": {"type": "string"},
        "veredicto_final": {
            "type": "string",
            "enum": [
                "COMPRA_FUERTE",
                "COMPRA",
                "MANTENER_ESPERAR",
                "VENTA",
                "VENTA_FUERTE",
                "DESCARTAR_RIESGO",
            ],
        },
        "probabilidad_exito_pct": {"type": "number"},
        "precio_objetivo_sugerido": {"type": "number"},
        "stop_loss_sugerido": {"type": "number"},
        "horizonte_estimado": {"type": "string"},
        "razonamiento_detallado": {"type": "string"},
    },
    "required": [
        "simbolo",
        "sentimiento_noticias",
        "impacto_noticias_en_accion",
        "evaluacion_datos_e_indicadores",
        "veredicto_final",
        "probabilidad_exito_pct",
        "precio_objetivo_sugerido",
        "stop_loss_sugerido",
        "horizonte_estimado",
        "razonamiento_detallado",
    ],
}

INSTRUCCION_SISTEMA_AI = (
    "Eres un analista cuantitativo e institucional senior (Portfolio Manager & Quant Researcher). "
    "Recibirás un objeto JSON `evento_potencial_para_analisis` junto con su `contexto` completo "
    "(historial de noticias de la acción, noticias macroeconómicas/Fed, historial de eventos técnicos, "
    "los 11 indicadores actuales, perfil de volumen de rango fijo FRVP POC/VAH/VAL, Chaikin Money Flow CMF, "
    "métricas de estabilidad Kaufman ER / Choppiness CHOP / R², relación P/E y niveles de Soporte/Resistencia). "
    "Tu tarea es:\n"
    "1. Analizar a profundidad las noticias de la empresa y macroeconómicas: determinar si son positivas, "
    "negativas o neutras y qué impacto fundamental directo o indirecto tienen en el flujo de caja, "
    "valoración o sentimiento institucional sobre la acción.\n"
    "2. Evaluar si los datos de mercado (P/E, FRVP, CMF, Soporte/Resistencia) y los indicadores técnicos "
    "confirman o contradicen el impacto de las noticias.\n"
    "3. Emitir un veredicto de alta precisión indicando la probabilidad de éxito (0-100%), el precio "
    "objetivo, el stop-loss y el razonamiento detallado en español."
)


class ConectorModeloAI:
    """
    Gestiona la conexión con modelos de IA:
    - Google Gemini (`gemini-3.8-flash`, `gemini-4-argon`, `gemini-4-carbon`, `gemini-3.5-flash-lite`)
      usando el SDK oficial `google-genai` (`client.interactions.create`).
    - Anthropic Claude (`claude-sonnet-4-6`, `claude-opus-4-6`).
    """

    def __init__(self, modelo_alias: str = "flash", timeout: int = 30):
        alias_norm = modelo_alias.strip().lower()
        info = CATALOGO_MODELOS_AI.get(alias_norm, CATALOGO_MODELOS_AI["flash"])
        self.alias = alias_norm
        self.proveedor = info["proveedor"]
        self.model_id = info["model_id"]
        self.descripcion = info["descripcion"]
        self.timeout = timeout

        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    def tiene_credenciales_activas(self) -> bool:
        if self.proveedor == "gemini":
            return bool(self.gemini_api_key)
        if self.proveedor == "claude":
            return bool(self.anthropic_api_key)
        return False

    def analizar_evento(
        self, evento: EventoPotencialParaAnalisis
    ) -> Dict[str, Any]:
        """
        Envía el `evento_potencial_para_analisis` y todo su `contexto` al modelo de IA.
        Si no hay API Key configurada en el entorno, genera un dictamen cuantitativo local
        estructurado indicando cómo activar la API Key.
        """
        payload_prompt = json.dumps(evento.a_dict(), ensure_ascii=False, indent=2)

        if self.proveedor == "gemini" and self.gemini_api_key:
            return self._llamar_gemini_interactions(payload_prompt, evento)
        if self.proveedor == "claude" and self.anthropic_api_key:
            return self._llamar_claude_messages(payload_prompt, evento)

        return self._generar_dictamen_local_fallback(evento)

    def _llamar_gemini_interactions(
        self, payload_prompt: str, evento: EventoPotencialParaAnalisis
    ) -> Dict[str, Any]:
        """
        Llama a la API de Google Gemini usando el SDK oficial actual `google-genai`
        con `client.interactions.create(...)` y salida estructurada JSON.
        """
        prompt_usuario = (
            f"Analiza el siguiente `evento_potencial_para_analisis` para la acción {evento.simbolo} "
            f"y devuelve el dictamen profundo en JSON:\n\n{payload_prompt}"
        )
        try:
            from google import genai

            client = genai.Client(api_key=self.gemini_api_key)
            interaction = client.interactions.create(
                model=self.model_id,
                system_instruction=INSTRUCCION_SISTEMA_AI,
                input=prompt_usuario,
                response_mime_type="application/json",
                response_format=ESQUEMA_RESPUESTA_AI,
            )
            texto = interaction.output_text or ""
            if texto:
                datos = json.loads(texto)
                datos["modelo_utilizado"] = self.model_id
                datos["modo"] = "API_GEMINI_INTERACTIONS"
                return datos
        except Exception as exc:
            # Si falla o si algún binario local está restringido, intentar fallback o reportar
            fallback = self._generar_dictamen_local_fallback(evento)
            fallback["nota_conexion_ai"] = f"Aviso al invocar {self.model_id}: {exc}"
            return fallback

        return self._generar_dictamen_local_fallback(evento)

    def _llamar_claude_messages(
        self, payload_prompt: str, evento: EventoPotencialParaAnalisis
    ) -> Dict[str, Any]:
        """
        Llama a Anthropic Claude (`claude-sonnet-4-6` o `claude-opus-4-6`) vía API.
        """
        prompt_usuario = (
            f"Analiza el siguiente `evento_potencial_para_analisis` para la acción {evento.simbolo} "
            f"y responde únicamente con un objeto JSON válido que cumpla las claves solicitadas "
            f"(simbolo, sentimiento_noticias, impacto_noticias_en_accion, evaluacion_datos_e_indicadores, "
            f"veredicto_final, probabilidad_exito_pct, precio_objetivo_sugerido, stop_loss_sugerido, "
            f"horizonte_estimado, razonamiento_detallado):\n\n{payload_prompt}"
        )
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model_id,
                    "max_tokens": 1024,
                    "system": INSTRUCCION_SISTEMA_AI,
                    "messages": [{"role": "user", "content": prompt_usuario}],
                },
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                contenido = resp.json().get("content", [])
                texto = "".join(
                    bloque.get("text", "")
                    for bloque in contenido
                    if bloque.get("type") == "text"
                ).strip()
                # Extraer bloque JSON si viene envuelto en markdown
                match = re.search(r"\{.*\}", texto, flags=re.DOTALL)
                if match:
                    datos = json.loads(match.group(0))
                    datos["modelo_utilizado"] = self.model_id
                    datos["modo"] = "API_CLAUDE"
                    return datos
        except Exception as exc:
            fallback = self._generar_dictamen_local_fallback(evento)
            fallback["nota_conexion_ai"] = f"Aviso al invocar {self.model_id}: {exc}"
            return fallback

        return self._generar_dictamen_local_fallback(evento)

    def _generar_dictamen_local_fallback(
        self, evento: EventoPotencialParaAnalisis
    ) -> Dict[str, Any]:
        """
        Pre-diagnóstico cuantitativo local cuando aún no se ha definido GEMINI_API_KEY
        o ANTHROPIC_API_KEY en las variables de entorno.
        """
        eval_tec = evento.contexto.evaluacion_tecnica_actual
        datos_mkt = evento.contexto.datos_mercado_actuales
        est = datos_mkt.get("estabilidad", {})
        frvp = datos_mkt.get("frvp", {})

        sent = evento.sentimiento_noticias_local.upper()
        dir_pre = evento.direccion_preliminar.upper()

        # Ajustar convicción según alineación entre el sentimiento de la noticia y la señal técnica
        alineado = (dir_pre == "COMPRA" and sent == "POSITIVO") or (
            dir_pre == "VENTA" and sent == "NEGATIVO"
        )
        contradicho = (dir_pre == "COMPRA" and sent == "NEGATIVO") or (
            dir_pre == "VENTA" and sent == "POSITIVO"
        )

        prob = evento.confianza_local_pct
        if alineado:
            prob = min(96.0, prob + 7.5)
            veredicto = f"{dir_pre}_FUERTE"
            impacto_not = (
                f"Catalizador noticioso {sent} alineado con la estructura técnica ({evento.resumen_catalizador})."
            )
        elif contradicho:
            prob = max(45.0, prob - 12.0)
            veredicto = "MANTENER_ESPERAR"
            impacto_not = (
                f"Divergencia entre noticia ({sent}) y señal técnica ({dir_pre}); prudencia recomendada."
            )
        else:
            veredicto = dir_pre
            impacto_not = (
                f"Contexto informativo {sent}; el movimiento está guiado principalmente por flujo institucional y estructura técnica."
            )

        return {
            "simbolo": evento.simbolo,
            "modelo_configurado": f"{self.model_id} (Modo Pre-Diagnóstico Local - defina GEMINI_API_KEY para inferencia en la nube)",
            "sentimiento_noticias": sent,
            "impacto_noticias_en_accion": impacto_not,
            "evaluacion_datos_e_indicadores": (
                f"Estabilidad {est.get('indice_certeza', 0):.0f}% (ER={est.get('efficiency_ratio', 0):.2f}, "
                f"CHOP={est.get('choppiness_index', 0):.1f}, R²={est.get('r2_tendencia', 0):.2f}) | "
                f"FRVP: {frvp.get('estado_volumen', '')} | CMF={datos_mkt.get('cmf_actual', 0):+.2f}"
            ),
            "veredicto_final": veredicto,
            "probabilidad_exito_pct": round(prob, 1),
            "precio_objetivo_sugerido": eval_tec.get("take_profit", datos_mkt.get("precio_actual", 0)),
            "stop_loss_sugerido": eval_tec.get("stop_loss", datos_mkt.get("precio_actual", 0)),
            "horizonte_estimado": "Intradía / Swing Corto (1 a 5 sesiones)",
            "razonamiento_detallado": (
                f"Combinación detectada ({evento.tipo_combinacion}): {eval_tec.get('indicadores_a_favor', 0)}/"
                f"{eval_tec.get('total_indicadores', 11)} indicadores a favor con Score "
                f"{eval_tec.get('score', 0):+.1f}/100 y relación R:R {eval_tec.get('ratio_riesgo_beneficio', 1):.2f}x."
            ),
        }


# =====================================================================
# NIVEL 2: GESTOR DE `evento_potencial_para_analisis` Y CONTEXTO
# =====================================================================
class GestorEventosAnalisis:
    """
    Orquesta el Segundo Nivel del proyecto:
    1. Vigila los eventos de Nivel 1 producidos por `infomarket.py` (noticias) e
       `indicadores.py` (datos fundamentales, flujo FRVP/CMF e indicadores estables).
    2. Evalúa si existe una combinación de eventos (Noticias + Datos + Indicadores)
       descartando acciones erráticas.
    3. Construye el `evento_potencial_para_analisis` y su `contexto` (agrupando todos
       los eventos y noticias previas de esa acción específica).
    4. Invoca al `ConectorModeloAI` (Nivel 3) y muestra el resultado en la terminal.
    """

    ARCHIVO_SALIDA_EVENTOS = (
        Path(__file__).resolve().parent / "eventos_potenciales_analisis.json"
    )

    def __init__(
        self,
        gestor_noticias: GestorInfoMarket,
        gestor_indicadores: GestorIndicadores,
        modelo_ai: str = "flash",
    ):
        self.gestor_noticias = gestor_noticias
        self.gestor_indicadores = gestor_indicadores
        self.conector_ai = ConectorModeloAI(modelo_alias=modelo_ai)

        self.contador_eventos: int = 0
        self.eventos_generados: List[EventoPotencialParaAnalisis] = []
        # Control para no duplicar el mismo evento en el mismo ciclo/ventana corta
        self.ultimo_ciclo_emitido_por_simbolo: Dict[str, int] = {}

    def limpiar_historial(self) -> None:
        """Auto-gestión de historial: elimina el archivo previo de eventos de análisis."""
        self.contador_eventos = 0
        self.eventos_generados.clear()
        self.ultimo_ciclo_emitido_por_simbolo.clear()
        if self.ARCHIVO_SALIDA_EVENTOS.exists():
            try:
                self.ARCHIVO_SALIDA_EVENTOS.unlink()
            except OSError:
                pass

    def _guardar_eventos_disco(self) -> None:
        try:
            datos = [ev.a_dict() for ev in self.eventos_generados[-100:]]
            self.ARCHIVO_SALIDA_EVENTOS.write_text(
                json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def _construir_contexto_accion(
        self, simbolo: str, snap_actual: InstantaneaEventoTecnico
    ) -> Tuple[ContextoAccion, str, str]:
        """
        Agrupa todos los eventos previos de la acción específica que generó el nuevo evento:
        - Todas las noticias previas y recientes de la acción.
        - Noticias macroeconómicas / Fed de contexto.
        - Todos los eventos técnicos previos registrados para la acción.
        - Estado actual de mercado, flujo FRVP, estabilidad e indicadores.
        """
        simbolo_up = simbolo.upper()
        nombre_emp = MAPA_NOMBRES_EMPRESAS.get(simbolo_up, simbolo_up)

        # 1. Noticias propias de la acción acumuladas en la sesión
        noticias_propias_obj: List[Noticia] = list(
            self.gestor_noticias.noticias_por_simbolo.get(simbolo_up, [])
        )
        noticias_propias_obj.sort(key=lambda n: n.fecha_publicacion, reverse=True)

        noticias_accion_list: List[Dict[str, Any]] = []
        puntaje_sentimiento_total = 0
        for n in noticias_propias_obj[:10]:
            sent_str, pts = estimar_sentimiento_titular(n.titulo)
            puntaje_sentimiento_total += pts
            noticias_accion_list.append(
                {
                    "fecha_publicacion": formatear_fecha(n.fecha_publicacion),
                    "fecha_encontrada": formatear_fecha(n.fecha_encontrada),
                    "plataforma": n.plataforma,
                    "titulo": n.titulo,
                    "link": n.link,
                    "sentimiento_lexico_local": sent_str,
                }
            )

        # 2. Noticias Macro / Fed recientes para contexto global
        noticias_macro_obj: List[Noticia] = sorted(
            self.gestor_noticias.noticias_macro,
            key=lambda n: n.fecha_publicacion,
            reverse=True,
        )[:5]
        noticias_macro_list: List[Dict[str, Any]] = []
        for m in noticias_macro_obj:
            sent_m, pts_m = estimar_sentimiento_titular(m.titulo)
            if not noticias_accion_list:
                puntaje_sentimiento_total += pts_m
            noticias_macro_list.append(
                {
                    "topico": m.simbolo,
                    "fecha_publicacion": formatear_fecha(m.fecha_publicacion),
                    "plataforma": m.plataforma,
                    "titulo": m.titulo,
                    "link": m.link,
                    "sentimiento_lexico_local": sent_m,
                }
            )

        if puntaje_sentimiento_total > 0:
            sentimiento_global = "positivo"
        elif puntaje_sentimiento_total < 0:
            sentimiento_global = "negativo"
        else:
            sentimiento_global = "neutro"

        if noticias_accion_list:
            ultimo_titular = noticias_accion_list[0].get("titulo") or noticias_accion_list[0].get("link")
            resumen_cat = f"Noticia directa [{noticias_accion_list[0]['plataforma']}]: {ultimo_titular}"
        elif noticias_macro_list:
            ultimo_macro = noticias_macro_list[0].get("titulo") or noticias_macro_list[0].get("link")
            resumen_cat = f"Contexto Macro [{noticias_macro_list[0]['topico']}/{noticias_macro_list[0]['plataforma']}]: {ultimo_macro}"
        else:
            resumen_cat = "Catalizador por flujo institucional (FRVP/CMF) y ruptura técnica limpia"

        # 3. Historial de eventos técnicos previos de esta acción
        hist_snaps = self.gestor_indicadores.historial_eventos_por_simbolo.get(simbolo_up, [])
        eventos_previos_list: List[Dict[str, Any]] = []
        for h_snap in hist_snaps[:-1][-8:]:
            eventos_previos_list.append(
                {
                    "timestamp": h_snap.timestamp,
                    "ciclo": h_snap.ciclo,
                    "precio": round(h_snap.flujo.precio_actual, 2),
                    "direccion": h_snap.evaluacion.direccion,
                    "score": round(h_snap.evaluacion.score, 2),
                    "confianza_pct": round(h_snap.evaluacion.confianza_pct, 1),
                    "certeza_estabilidad_pct": round(h_snap.flujo.estabilidad.indice_certeza, 1),
                }
            )

        contexto = ContextoAccion(
            simbolo=simbolo_up,
            nombre_empresa=nombre_emp,
            noticias_accion=noticias_accion_list,
            noticias_macro_contexto=noticias_macro_list,
            eventos_tecnicos_previos=eventos_previos_list,
            datos_mercado_actuales=snap_actual.flujo.a_dict(),
            indicadores_actuales=[ind.a_dict() for ind in snap_actual.indicadores],
            evaluacion_tecnica_actual=snap_actual.evaluacion.a_dict(),
        )
        return contexto, sentimiento_global, resumen_cat

    def _imprimir_evento_analisis(self, ev: EventoPotencialParaAnalisis) -> None:
        simbolo_turquesa = resaltar_turquesa(ev.simbolo)
        etiqueta_ev = (
            f"{ColoresAI.EVENTO_ANALISIS_BG} EVENTO_POTENCIAL_PARA_ANALISIS {ColoresAI.RESET}"
        )
        color_dir = (
            ColoresAI.VERDE_BG if ev.direccion_preliminar == "compra" else ColoresAI.ROJO_BG
        )
        ctx = ev.contexto
        n_not_acc = len(ctx.noticias_accion)
        n_not_mac = len(ctx.noticias_macro_contexto)
        n_ev_prev = len(ctx.eventos_tecnicos_previos)

        print(
            f"{ev.timestamp} | {simbolo_turquesa} | {etiqueta_ev} | "
            f"ID: {ev.id_evento} | Tipo: {ev.tipo_combinacion} | "
            f"Señal Base: {color_dir} {ev.direccion_preliminar.upper()} {ColoresAI.RESET} "
            f"(Confianza: {ev.confianza_local_pct:.1f}% | Certeza Estabilidad: {ev.certeza_estabilidad_pct:.0f}%) | "
            f"Contexto Agrupado: [Noticias Acción: {n_not_acc} | Macro/Fed: {n_not_mac} | "
            f"Eventos Técnicos Previos: {n_ev_prev}] | {ev.resumen_catalizador}"
        )

        if ev.dictamen_ai:
            d = ev.dictamen_ai
            etiqueta_ai = f"{ColoresAI.MODELO_AI_BG} DICTAMEN AI ({self.conector_ai.model_id}) {ColoresAI.RESET}"
            veredicto = str(d.get("veredicto_final", "N/D"))
            prob = d.get("probabilidad_exito_pct", 0.0)
            obj = d.get("precio_objetivo_sugerido", 0.0)
            sl = d.get("stop_loss_sugerido", 0.0)
            sent_not = d.get("sentimiento_noticias", "N/D")
            impacto = d.get("impacto_noticias_en_accion", "")
            razon = d.get("razonamiento_detallado", "")

            print(
                f"{ev.timestamp} | {simbolo_turquesa} | {etiqueta_ai} | "
                f"Veredicto: {ColoresAI.AMARILLO}{veredicto}{ColoresAI.RESET} "
                f"(Prob. Éxito: {prob}%) | Sentimiento Noticia: {sent_not} | "
                f"Objetivo AI: ${obj:.2f} | Stop AI: ${sl:.2f} | "
                f"Impacto: {impacto} | Análisis: {razon}"
            )

    def evaluar_y_generar_eventos(
        self, nuevas_noticias_ciclo: List[Noticia]
    ) -> List[EventoPotencialParaAnalisis]:
        """
        Ejecuta la correlación de Segundo Nivel al finalizar cada ciclo de escaneo:
        1. Revisa los eventos técnicos de alto potencial validados en `indicadores.py`
           (que ya descartaron acciones erráticas o volátiles).
        2. Revisa si entraron nuevas noticias corporativas en `infomarket.py` cuya acción
           presente respaldo técnico y estabilidad estructural.
        3. Si existe combinación de eventos, genera el `evento_potencial_para_analisis` con
           todo su `contexto`, lo envía al modelo de IA y muestra el dictamen en terminal.
        """
        ciclo = self.gestor_indicadores.ciclo_actual
        ahora_str = formatear_fecha(dt.datetime.now().astimezone())
        eventos_ciclo: List[EventoPotencialParaAnalisis] = []

        # En el ciclo 1 (carga de contexto histórico de 2 días y tabla inicial de 89 tickers),
        # seleccionamos solo las 2 mejores combinaciones que tengan TANTO noticia propia de la
        # acción COMO evento técnico de alto potencial validado, para no saturar la primera corrida.
        simbolos_candidatos: Dict[str, Tuple[InstantaneaEventoTecnico, str]] = {}

        if ciclo == 1:
            for snap in self.gestor_indicadores.ultimos_eventos_potenciales:
                if (
                    snap.simbolo in self.gestor_noticias.noticias_por_simbolo
                    and not snap.flujo.estabilidad.es_erratica
                ):
                    simbolos_candidatos[snap.simbolo] = (
                        snap,
                        "CONFLUENCIA_COMPLETA (Noticia Empresa + Indicadores Alto Potencial + Datos)",
                    )
        else:
            # 1) Acciones que dispararon un Evento Técnico de Alto Potencial en este ciclo
            for snap in self.gestor_indicadores.ultimos_eventos_potenciales:
                if snap.flujo.estabilidad.es_erratica:
                    continue
                tiene_not_propia = bool(
                    self.gestor_noticias.noticias_por_simbolo.get(snap.simbolo)
                )
                tipo_comb = (
                    "CONFLUENCIA_COMPLETA (Noticia Empresa + Indicadores Alto Potencial + Datos)"
                    if tiene_not_propia
                    else "CONFLUENCIA_TECNICA_Y_FLUJO_CON_CONTEXTO_MACRO"
                )
                simbolos_candidatos[snap.simbolo] = (snap, tipo_comb)

            # 2) Acciones que recibieron una NUEVA noticia en este ciclo y tienen respaldo técnico estable
            for noticia in nuevas_noticias_ciclo:
                sym = noticia.simbolo.upper()
                if sym in simbolos_candidatos:
                    continue
                snap_sym = self.gestor_indicadores.ultima_instantanea_por_simbolo.get(sym)
                if snap_sym is None or snap_sym.flujo.estabilidad.es_erratica:
                    continue
                # Verificar si la acción tiene confluencia técnica o movimiento relevante
                if (
                    snap_sym.evaluacion.es_alto_potencial
                    or abs(snap_sym.evaluacion.score) >= 24.0
                ):
                    simbolos_candidatos[sym] = (
                        snap_sym,
                        "NUEVA_NOTICIA_CON_RESPALDO_TECNICO_Y_ESTABILIDAD",
                    )

        # Ordenar candidatos por mayor confianza + certeza de estabilidad
        lista_ordenada = sorted(
            simbolos_candidatos.values(),
            key=lambda par: (
                par[0].evaluacion.confianza_pct
                + par[0].flujo.estabilidad.indice_certeza * 0.4
            ),
            reverse=True,
        )

        max_por_ciclo = 2 if ciclo == 1 else 3
        for snap, tipo_comb in lista_ordenada[:max_por_ciclo]:
            sym = snap.simbolo
            ult_ciclo = self.ultimo_ciclo_emitido_por_simbolo.get(sym)
            if ult_ciclo is not None and (ciclo - ult_ciclo) < 6:
                continue

            self.contador_eventos += 1
            self.ultimo_ciclo_emitido_por_simbolo[sym] = ciclo

            contexto, sent_local, resumen_cat = self._construir_contexto_accion(sym, snap)
            direccion = (
                snap.evaluacion.direccion
                if snap.evaluacion.direccion
                else ("compra" if snap.evaluacion.score >= 0 else "venta")
            )

            id_ev = f"EPA-{ciclo:04d}-{self.contador_eventos:04d}-{sym}"
            evento = EventoPotencialParaAnalisis(
                id_evento=id_ev,
                timestamp=ahora_str,
                ciclo=ciclo,
                simbolo=sym,
                tipo_combinacion=tipo_comb,
                direccion_preliminar=direccion,
                confianza_local_pct=snap.evaluacion.confianza_pct,
                certeza_estabilidad_pct=snap.flujo.estabilidad.indice_certeza,
                sentimiento_noticias_local=sent_local,
                resumen_catalizador=resumen_cat,
                contexto=contexto,
            )

            # Nivel 3: Analizar con el Modelo de IA configurado
            evento.dictamen_ai = self.conector_ai.analizar_evento(evento)

            self.eventos_generados.append(evento)
            eventos_ciclo.append(evento)
            self._imprimir_evento_analisis(evento)

        if eventos_ciclo:
            self._guardar_eventos_disco()
        elif ciclo > 1:
            etiqueta_azul = resaltar_azul("evento_potencial_para_analisis")
            print(f"{ahora_str} | sin combinaciones nuevas [{etiqueta_azul}]")

        return eventos_ciclo
