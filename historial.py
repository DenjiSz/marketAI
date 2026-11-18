# -*- coding: utf-8 -*-
"""
historial.py
============
Módulo de higiene del espacio de trabajo, bus IPC en disco para las 4 terminales
y persistencia del historial de sesión y de posiciones cerradas.

Funciones clave:
1. Al iniciar el proyecto, revisa la raíz del espacio de trabajo:
   - Protege estrictamente scripts (.py), documentación (.md como AGENTS.md) y carpetas del sistema.
   - Mueve cualquier residuo o archivo generado por sesiones previas a la carpeta `hystmp/`,
     manteniendo el directorio limpio e identificando claramente qué carpeta puede borrarse sin riesgo.
2. Verifica (y regenera si faltase) el archivo `AGENTS.md` compatible con Gemini Carbon/Argon y Claude Opus/Fable.
3. Administra el directorio `.sesion_activa/` donde la terminal principal publica en tiempo real
   la cartera virtual, el flujo secuencial de noticias y los eventos de indicadores para `/car-v`, `/mkinfo` y `/datos`.
4. Genera al cierre (/quit) dos archivos independientes en la raíz del proyecto:
   - Resumen completo de la sesión (`resumen_sesion_<timestamp>.md` y `.json`).
   - Historial detallado de posiciones (`historial_posiciones_<timestamp>.csv`).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    NOMBRE_DIR_HYSTMP,
    NOMBRE_DIR_SESION_ACTIVA,
    formatear_fecha_hora_ampm,
)

# Contenido base de respaldo en caso de que AGENTS.md haya sido eliminado manualmente
PLANTILLA_RESPALDO_AGENTS_MD = """# AGENTS.md — Contexto y Reglas de Arquitectura del Proyecto VAGY-M1

Sistema modular en Python para monitoreo bursátil multi-terminal (4 cuadrantes), escaneo cuantitativo vectorizado, análisis informativo multi-fuente en 6 secciones y toma de decisiones porcentuales sobre cartera virtual con IA (**Método de Consulta Múltiple por Lotes Sectoriales**: Gemini Flash 3.6/3.7/3.8, Argon, Hydrogen, Carbon, Pro 3.1 y Claude Sonnet/Opus/Fable con telemetría de tokens y costo USD).

## Módulos Principales
- `main.py`: Orquestador principal, asistente de inicio (10 modelos + esfuerzos), interfaz TUI de 4 zonas con `GestorComandosInteractivos` no bloqueante, comandos interactivos (`/v`, `/n`, `/d`, `/pausar`, `/reanudar`, `/modelo`, `/lote`, `/apikey`, `/help`, `/quit`), leyendas `Supervisión AI` (morado, con tokens/consulta, acumulados y costo USD) y `Supervisión Noticias` (azul).
- `analisis.py`: Motor de análisis de información dividido en **6 secciones**: (1) `NOTICIAS`, (2) `FILINGS_8K` (SEC EDGAR + `yfinance`), (3) `REPORTES_RESULTADOS` (evaluación conjunta con `scanner.py`: fecha próximo reporte, ingresos/EPS esperados vs reportados), (4) `DIVULGACIONES_CONGRESO` (6 categorías oficiales: Ingresos ganados/no ganados, Activos/propiedades, Deudas/pasivos, Regalos/viajes, Cargos externos/acuerdos, Finanzas familiares), (5) `FED` (tasas, FOMC, CPI/PCE) y (6) `MACROECONOMIA` (avances en otros países, tensión geopolítica, movimientos bélicos).
- `scanner.py` (`datos_indicadores.py`): Descarga vectorizada por lotes y análisis técnico/cuantitativo con flujo de volumen real (`RVOL`, `OBV`, `CMF`), 17 indicadores incluyendo **Rupturas de Rango de Apertura (ORB)**, **Patrones Bandera**, **Rebotes por Sobreventa / Rechazos**, **Tendencia Diaria SMA20/50 + Squeeze**, y extracción de **Reportes de Resultados** (`fecha_proximo_reporte`, `ingresos_esperados`, `ingresos_reportados`).
- `analisis_ai.py`: Smart Gatekeeper anti-day-trading con matriz multi-pilar, sensibilidad por horario ET (`09:15-09:30` pre-market, `09:30-09:50` apertura y `15:45-16:00` cierre flexibles; estricto en cualquier otro horario exigiendo combinación de pilares), consulta múltiple en una sola llamada (`DecisionLoteModeloIA`) y contabilidad de tokens/costo USD.
- `cartera.py`: Gestor de cartera virtual porcentual (`spot` vs `short y long`) y rotación por certeza.
- `terminales.py`: Ventanas secundarias dependientes (`/v`, `/n`, `/d`) lanzadas en cuadrantes sin robo de foco (`SW_SHOWNOACTIVATE`) y sin redimensionar Windows Terminal (`CASCADIA_HOSTING_WINDOW_CLASS`).
- `historial.py` y `config.py`: Limpieza automática integral de residuos (`__pycache__`, `.md` de resumen, `.csv`, `.json`, `.sesion_activa/`) hacia `hystmp/`, bus IPC y catálogos/tarifas.

## Reglas Esenciales para Agentes
1. Mantener toda operación de cartera en base porcentual (`%` del patrimonio).
2. En Gemini usar exclusivamente `from google import genai` y `client.interactions.create(...)` con `generation_config={"thinking_level": ...}`, sin `temperature`/`top_p`/`top_k` ni `thinking_budget`.
3. Respetar los colores ANSI exactos: nunca colorear los corchetes `[` `]` alrededor de `[ticker]`, usar leyenda morada para `Supervisión AI` y leyenda azul para `Supervisión Noticias`.
4. Nunca redimensionar la ventana padre de Windows Terminal (`CASCADIA_HOSTING_WINDOW_CLASS`) ni desactivar `ENABLE_VIRTUAL_TERMINAL_INPUT` (`0x0200`).
5. Al iniciar sesión, mover todo residuo (`__pycache__`, `.sesion_activa/`, `resumen_sesion_*`, `.csv`, `.json`, `.log`, `.tmp`, `.bak`) a `hystmp/` protegiendo scripts `.py`, documentación canónica (`AGENTS.md`, `README.md`) y `hystmp/`.
"""


class GestorHistorial:
    """Administra limpieza inicial (`hystmp`), IPC de sesión activa y guardado de reportes finales."""

    # Extensiones de código y configuración que NUNCA deben moverse a hystmp
    EXTENSIONES_PROTEGIDAS = {".py", ".toml", ".cfg", ".ini", ".bat", ".ps1", ".yaml", ".yml"}
    NOMBRES_PROTEGIDOS = {
        "AGENTS.md",
        "GEMINI.md",
        "CLAUDE.md",
        "README.md",
        "requirements.txt",
        ".gitignore",
        ".env",
        NOMBRE_DIR_HYSTMP,
        "old",
        ".git",
        ".agents",
        ".agent",
        "venv",
        ".venv",
    }
    CARPETAS_RESIDUALES = {
        NOMBRE_DIR_SESION_ACTIVA,
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "tmp",
        "logs",
    }
    EXTENSIONES_RESIDUALES = {
        ".log",
        ".tmp",
        ".jsonl",
        ".csv",
        ".json",
        ".bak",
        ".pyc",
        ".pyo",
        ".md",
        ".txt",
    }

    def __init__(self, directorio_base: Optional[ str | Path ] = None) -> None:
        self.base_dir = Path(directorio_base or Path(__file__).resolve().parent)
        self.hystmp_dir = self.base_dir / NOMBRE_DIR_HYSTMP
        self.sesion_dir = self.base_dir / NOMBRE_DIR_SESION_ACTIVA
        self.pids_dir = self.sesion_dir / "pids"
        self._io_lock = threading.RLock()

        # Rutas de comunicación IPC entre main.py y las 3 terminales secundarias
        self.archivo_estado_sesion = self.sesion_dir / "session_state.json"
        self.archivo_cartera = self.sesion_dir / "cartera_state.json"
        self.archivo_noticias_jsonl = self.sesion_dir / "mkinfo_noticias.jsonl"
        self.archivo_datos_jsonl = self.sesion_dir / "datos_eventos.jsonl"
        self.archivo_log_ai_jsonl = self.sesion_dir / "ai_consultas.jsonl"

    # ------------------------------------------------------------------
    # 1. LIMPIEZA INICIAL Y VERIFICACIÓN DE AGENTS.md
    # ------------------------------------------------------------------

    def preparar_espacio_trabajo(self) -> Dict[str, Any]:
        """
        Ejecuta la rutina de inicio limpio:
        1. Asegura la existencia del directorio `hystmp/`.
        2. Detecta archivos y carpetas residuales de ejecuciones previas (`__pycache__`,
           `.sesion_activa/`, `resumen_sesion_*.md/.json`, `historial_posiciones_*.csv`,
           logs, temporales, etc.) y los mueve a `hystmp/sesion_previa_<timestamp>/`
           protegiendo estrictamente scripts `.py` y documentación canónica (`AGENTS.md`, `README.md`).
        3. Verifica que `AGENTS.md` exista con el formato correcto.
        4. Inicializa un directorio `.sesion_activa/` limpio para la nueva ejecución.
        """
        self.hystmp_dir.mkdir(parents=True, exist_ok=True)
        movidos: List[str] = []
        marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
        destino_lote: Optional[Path] = None

        for item in list(self.base_dir.iterdir()):
            nombre = item.name

            # Proteger archivos/carpetas canónicos del proyecto y entornos virtuales
            if nombre in self.NOMBRES_PROTEGIDOS:
                continue
            if item.is_file() and item.suffix.lower() in self.EXTENSIONES_PROTEGIDAS:
                continue
            if nombre.startswith(".git") or nombre.startswith(".agent"):
                continue

            # Identificar si es residuo/historial de una sesión anterior
            es_residuo = (
                nombre in self.CARPETAS_RESIDUALES
                or nombre.startswith("resumen_sesion_")
                or nombre.startswith("historial_posiciones_")
                or nombre.startswith("sesion_")
                or nombre.startswith("reporte_")
                or (item.is_file() and item.suffix.lower() in self.EXTENSIONES_RESIDUALES)
            )

            if es_residuo:
                if destino_lote is None:
                    destino_lote = self.hystmp_dir / f"sesion_previa_{marca_tiempo}"
                    destino_lote.mkdir(parents=True, exist_ok=True)
                destino_final = destino_lote / nombre
                if destino_final.exists():
                    destino_final = destino_lote / f"{nombre}_{int(time.time() * 1000) % 10000}"
                try:
                    shutil.move(str(item), str(destino_final))
                    movidos.append(nombre)
                except Exception:
                    # En Windows, si algún subarchivo de __pycache__/.sesion_activa estuviera ocupado,
                    # intentamos mover archivo por archivo y limpiar lo sobrante.
                    if item.is_dir():
                        try:
                            destino_final.mkdir(parents=True, exist_ok=True)
                            alg_movido = False
                            for sub in list(item.iterdir()):
                                try:
                                    shutil.move(str(sub), str(destino_final / sub.name))
                                    alg_movido = True
                                except Exception:
                                    pass
                            if alg_movido:
                                movidos.append(f"{nombre}/*")
                            shutil.rmtree(item, ignore_errors=True)
                        except Exception:
                            pass

        agents_creado = self.asegurar_agents_md()

        # Crear estructura fresca para la sesión actual
        self.sesion_dir.mkdir(parents=True, exist_ok=True)
        self.pids_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar archivos vacíos de flujo para las terminales secundarias
        self.archivo_noticias_jsonl.write_text("", encoding="utf-8")
        self.archivo_datos_jsonl.write_text("", encoding="utf-8")
        self.archivo_log_ai_jsonl.write_text("", encoding="utf-8")

        self.actualizar_estado_sesion({
            "activa": True,
            "cerrando": False,
            "modo_cierre": None,
            "parent_pid": os.getpid(),
            "inicio_sesion": formatear_fecha_hora_ampm(),
            "inicio_epoch": time.time(),
        })

        return {
            "archivos_movidos_a_hystmp": movidos,
            "carpeta_lote_hystmp": str(destino_lote) if destino_lote else None,
            "agents_md_verificado": True,
            "agents_md_recien_creado": agents_creado,
        }

    def asegurar_agents_md(self) -> bool:
        """
        Garantiza que `AGENTS.md` exista en la raíz del proyecto sin frontmatter YAML,
        optimizado para Gemini Carbon/Argon y Claude Opus/Fable.
        Devuelve True si tuvo que crearlo, False si ya existía correctamente.
        """
        ruta_agents = self.base_dir / "AGENTS.md"
        if ruta_agents.exists() and ruta_agents.stat().st_size > 100:
            return False
        ruta_agents.write_text(PLANTILLA_RESPALDO_AGENTS_MD, encoding="utf-8")
        return True

    # ------------------------------------------------------------------
    # 2. ESCRITURA ATÓMICA Y STREAMS JSONL PARA LAS 4 TERMINALES
    # ------------------------------------------------------------------

    def _escribir_json_atomico(self, ruta: Path, datos: Dict[str, Any]) -> None:
        """Escribe un archivo JSON de forma atómica evitando lecturas parciales en terminales secundarias."""
        with self._io_lock:
            self.sesion_dir.mkdir(parents=True, exist_ok=True)
            ruta_tmp = ruta.with_suffix(f".tmp.{threading.get_ident()}")
            try:
                contenido = json.dumps(datos, ensure_ascii=False, indent=2)
                ruta_tmp.write_text(contenido, encoding="utf-8")
                os.replace(str(ruta_tmp), str(ruta))
            except Exception:
                try:
                    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                finally:
                    if ruta_tmp.exists():
                        try:
                            ruta_tmp.unlink()
                        except Exception:
                            pass

    def actualizar_estado_sesion(self, cambios: Dict[str, Any]) -> None:
        """Actualiza metadatos globales de la sesión activa (heartbeat, cierre, configuración)."""
        with self._io_lock:
            estado_actual: Dict[str, Any] = {}
            if self.archivo_estado_sesion.exists():
                try:
                    estado_actual = json.loads(self.archivo_estado_sesion.read_text(encoding="utf-8"))
                except Exception:
                    estado_actual = {}
            estado_actual.update(cambios)
            estado_actual["heartbeat_epoch"] = time.time()
            self._escribir_json_atomico(self.archivo_estado_sesion, estado_actual)

    def publicar_estado_cartera(self, snapshot_cartera: Dict[str, Any]) -> None:
        """Publica el estado instantáneo de la cartera virtual para la terminal `/car-v`."""
        self._escribir_json_atomico(self.archivo_cartera, snapshot_cartera)

    def registrar_noticia_stream(self, noticia_dict: Dict[str, Any]) -> None:
        """Agrega secuencialmente una noticia al stream leído por la terminal `/mkinfo`."""
        with self._io_lock:
            try:
                linea = json.dumps(noticia_dict, ensure_ascii=False)
                with self.archivo_noticias_jsonl.open("a", encoding="utf-8") as f:
                    f.write(linea + "\n")
            except Exception:
                pass

    def registrar_evento_datos_stream(self, evento_datos_dict: Dict[str, Any]) -> None:
        """Agrega un evento técnico/financiero de alto potencial al stream de la terminal `/datos`."""
        with self._io_lock:
            try:
                linea = json.dumps(evento_datos_dict, ensure_ascii=False)
                with self.archivo_datos_jsonl.open("a", encoding="utf-8") as f:
                    f.write(linea + "\n")
            except Exception:
                pass

    def registrar_consulta_ai(self, registro_ai: Dict[str, Any]) -> None:
        """Guarda auditoría interna de cada consulta enviada al modelo de IA durante la sesión."""
        with self._io_lock:
            try:
                linea = json.dumps(registro_ai, ensure_ascii=False)
                with self.archivo_log_ai_jsonl.open("a", encoding="utf-8") as f:
                    f.write(linea + "\n")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 3. EXPORTACIÓN FINAL AL EJECUTAR /quit (RESUMEN + HISTORIAL APARTE)
    # ------------------------------------------------------------------

    def exportar_cierre_sesion(
        self,
        resumen_datos: Dict[str, Any],
        historial_posiciones: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Guarda en archivos separados:
        1. El resumen ejecutivo de la sesión (`resumen_sesion_<ts>.md` y `.json`).
        2. El historial completo de posiciones (`historial_posiciones_<ts>.csv`), incluyendo
           apertura, ticker, tipo, tamaño porcentual y USD, cierre y PyG (% y USD).
        (En la próxima ejecución del sistema, estos archivos serán archivados automáticamente en `hystmp/`).
        """
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_resumen_md = self.base_dir / f"resumen_sesion_{marca}.md"
        ruta_resumen_json = self.base_dir / f"resumen_sesion_{marca}.json"
        ruta_historial_csv = self.base_dir / f"historial_posiciones_{marca}.csv"

        # 1. Guardar JSON estructurado del resumen
        ruta_resumen_json.write_text(
            json.dumps(resumen_datos, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 2. Construir Resumen legible en Markdown
        lineas_md = [
            f"# Resumen de Cierre de Sesión — VAGY-M1 ({resumen_datos.get('fecha_cierre', '')})",
            "",
            "## 1. Configuración y Modo de Cierre",
            f"- **Modelo de IA**: `{resumen_datos.get('modelo_ai', '')}` (`{resumen_datos.get('model_id', '')}`)",
            f"- **Nivel de Esfuerzo (Thinking/Effort)**: `{resumen_datos.get('nivel_esfuerzo', 'medium')}`",
            f"- **Tipo de Operación**: `{resumen_datos.get('modo_operacion', '')}`",
            f"- **Lista Seleccionada**: `{resumen_datos.get('lista_nombre', '')}` ({resumen_datos.get('total_tickers', 0)} acciones)",
            f"- **Modalidad de Cierre Ejecutada**: `{resumen_datos.get('modo_cierre', '')}`",
            f"- **Inicio de Sesión**: {resumen_datos.get('fecha_inicio', '')}",
            f"- **Fin de Sesión**: {resumen_datos.get('fecha_cierre', '')}",
            "",
            "## 2. Rendimiento de Cartera Virtual (Base Porcentual)",
            f"- **Capital Virtual Inicial**: `${resumen_datos.get('capital_inicial_usd', 0.0):,.2f} USD` (100.00%)",
            f"- **Patrimonio Virtual Final**: `${resumen_datos.get('patrimonio_final_usd', 0.0):,.2f} USD`",
            f"- **Rendimiento Neto Porcentual (PyG %)**: `{resumen_datos.get('rendimiento_total_pct', 0.0):+.2f}%`",
            f"- **PyG Total Equivalente (USD)**: `${resumen_datos.get('pyg_total_usd', 0.0):+,.2f} USD`",
            f"- **Total Operaciones Cerradas**: `{resumen_datos.get('operaciones_cerradas', 0)}`",
            f"- **Tasa de Acierto (Win Rate)**: `{resumen_datos.get('win_rate_pct', 0.0):.1f}%`",
            "",
            "## 3. Estadísticas del Supervisor de Agente IA, Tokens y Costo",
            f"- **Total Consultas (Lotes) a IA**: `{resumen_datos.get('ai_total_consultas', 0)}` (`{resumen_datos.get('ai_activos_evaluados', 0)}` empresas evaluadas)",
            f"- **Tokens Acumulados**: `{resumen_datos.get('ai_tokens_total', 0):,}` (Input: `{resumen_datos.get('ai_tokens_input', 0):,}` | Output: `{resumen_datos.get('ai_tokens_output', 0):,}` | Thought: `{resumen_datos.get('ai_tokens_thought', 0):,}` | Cache: `{resumen_datos.get('ai_tokens_cached', 0):,}`)",
            f"- **Costo Estimado Total IA**: `${resumen_datos.get('ai_costo_total_usd', 0.0):.5f} USD`",
            f"- **Consultas Ahorradas (Agrupación Lotes / Gatekeeper)**: `{resumen_datos.get('ai_consultas_ahorradas_lote', 0)}` por lote / `{resumen_datos.get('ai_consultas_filtradas', 0)}` filtradas",
            f"- **Predicciones Alcistas (Compra/Long)**: `{resumen_datos.get('ai_alcistas', 0)}`",
            f"- **Predicciones Bajistas (Venta/Short)**: `{resumen_datos.get('ai_bajistas', 0)}`",
            f"- **Respuestas Neutrales (Descartadas)**: `{resumen_datos.get('ai_neutrales', 0)}`",
            f"- **Pausas IA (Modo Offline)**: `{resumen_datos.get('ai_pausas_offline', 0)}`",
            f"- **Fallos de Conexión / Fondos IA**: `{resumen_datos.get('ai_errores', 0)}`",
            f"- **Total Eventos Informativos Procesados**: `{resumen_datos.get('total_noticias', 0)}`",
            "",
            f"> Historial detallado de posiciones guardado en: `{ruta_historial_csv.name}`",
        ]
        ruta_resumen_md.write_text("\n".join(lineas_md) + "\n", encoding="utf-8")

        # 3. Guardar archivo separado de Historial de Posiciones en CSV
        columnas = [
            "id_posicion",
            "fecha_apertura",
            "ticker",
            "tipo_operacion",
            "tamano_porcentual_cartera",
            "tamano_usd",
            "precio_apertura",
            "fecha_cierre",
            "precio_cierre",
            "pyg_porcentual",
            "pyg_usd",
            "motivo_cierre",
        ]
        with ruta_historial_csv.open("w", newline="", encoding="utf-8-sig") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=columnas)
            writer.writeheader()
            for pos in historial_posiciones:
                writer.writerow({
                    "id_posicion": pos.get("id_posicion", ""),
                    "fecha_apertura": pos.get("fecha_apertura", ""),
                    "ticker": pos.get("ticker", ""),
                    "tipo_operacion": pos.get("tipo", ""),
                    "tamano_porcentual_cartera": f"{float(pos.get('tamano_pct', 0.0)):.2f}%",
                    "tamano_usd": f"{float(pos.get('tamano_usd', 0.0)):.2f}",
                    "precio_apertura": f"{float(pos.get('precio_entrada', 0.0)):.4f}",
                    "fecha_cierre": pos.get("fecha_cierre", ""),
                    "precio_cierre": f"{float(pos.get('precio_cierre', 0.0)):.4f}",
                    "pyg_porcentual": f"{float(pos.get('pyg_pct', 0.0)):+.2f}%",
                    "pyg_usd": f"{float(pos.get('pyg_usd', 0.0)):+.2f}",
                    "motivo_cierre": pos.get("motivo_cierre", ""),
                })

        return {
            "resumen_md": str(ruta_resumen_md),
            "resumen_json": str(ruta_resumen_json),
            "historial_csv": str(ruta_historial_csv),
        }
