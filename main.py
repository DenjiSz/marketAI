# -*- coding: utf-8 -*-
"""
main.py
=======
Orquestador principal del proyecto VAGY-M1.

Responsabilidades:
1. Higiene inicial del espacio de trabajo (`historial.py` -> mueve residuos previos a `hystmp/`
   sin tocar scripts `.py` ni documentos `.md`, y garantiza `AGENTS.md`).
2. Menú interactivo de inicialización ampliado:
   - Modelo de IA (`flash3.6`, `flash3.7`, `flash3.8`, `argon`, `hydrogen`, `carbon`,
     `pro3.1`, `sonnet`, `opus`, `fable`)
   - Nivel de esfuerzo (`thinking_level` / `effort`: low, medium, high, xhigh, max según modelo)
   - API Key verificada en vivo contra el proveedor y nivel de esfuerzo elegido
   - Tipo de operación (`spot` / `sport` o `short y long`)
   - Lista de acciones (`Lista 1` o `Lista 2`) y monto inicial virtual en USD
   - Resumen completo de configuración inicial
3. Apertura automática sin robo de foco de las 3 terminales secundarias (`/v`, `/n`, `/d`).
4. Interfaz principal TUI rediseñada de 4 zonas con comandos interactivos instantáneos:
   - Zona 1 (Superior desplazable): Solo hallazgos de alto potencial (alcista/bajista).
   - Zona 2 (Antepenúltima línea): Leyenda `Supervisión AI` en morado + métricas, errores y tickers por consulta.
   - Zona 3 (Penúltima línea): Leyenda `Supervisión Noticias` en azul + salud, hora AM/PM, `[ticker]`, fuente.
   - Zona 4 (Última línea): Consola interactiva asíncrona con historial (`↑↓`), autocompletado (`Tab`),
     pegado (`Ctrl+V`) y edición en línea (incluye `/pausar` y `/reanudar` modo offline IA).
"""

from __future__ import annotations

import atexit
import os
import queue
import shutil
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

if os.name == "nt":
    import msvcrt

from analisis_ai import (
    AnalizadorAI,
    EventoImpresionPrincipal,
    limpiar_api_key,
    probar_conexion_modelo_ia,
)
from cartera import CarteraVirtual
from config import (
    LISTA_ACCIONES_1,
    LISTA_ACCIONES_2,
    MODELOS_IA_DISPONIBLES,
    Colores,
    activar_ansi_windows,
    colorear_evaluacion,
    colorear_nombre_indicador,
    colorear_plataforma_noticia,
    configurar_consola_tui_interactiva,
    formatear_fecha_hora_ampm,
    leer_portapapeles_windows,
    leyenda_supervision_ai,
    leyenda_supervision_noticias,
    longitud_visible_ansi,
    normalizar_esfuerzo_modelo,
    obtener_regimen_horario_mercado,
    recortar_ansi_visible,
    resolver_clave_modelo,
    restaurar_consola_windows,
    ticker_azul,
    ticker_morado,
)
from historial import GestorHistorial
from indicadores import GestorIndicadores
from infomarket import GestorInfoMarket
from terminales import GestorTerminalesSecundarias


# ==============================================================================
# 1. INTERFAZ TUI DE 4 ZONAS ANTI-PARPADEO Y SIN DESBORDE DE PANTALLA
# ==============================================================================

class ConsolaCuatroZonas:
    """
    Divide la terminal principal en 4 secciones estables:
    - Filas 1 .. (alto - 4): Salida de análisis IA de alto potencial (scroll VT100 natural).
    - Fila (alto - 3)      : Barra divisoria + avisos breves / régimen horario.
    - Fila (alto - 2)      : Antepenúltima línea -> `Supervisión AI` (morado) + tickers por consulta.
    - Fila (alto - 1)      : Penúltima línea     -> `Supervisión Noticias` (azul) + flujo/latencia.
    - Fila (alto)          : Última línea        -> Consola interactiva (`/v`, `/n`, `/d`, `/quit`, etc.).
    """

    PREFIJO_CMD_LARGO = "CMD (/v /n /d /pausar /reanudar /modelo /lote /apikey /help /quit) > "
    PREFIJO_CMD_MEDIO = "CMD (/v /n /d /pausar /reanudar /apikey /help /quit) > "
    PREFIJO_CMD_CORTO = "CMD (/v /n /d /pausar /reanudar /quit /help) > "
    PREFIJO_CMD_MINI = "CMD (/pausar /reanudar /help /quit) > "

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cols, self._rows = self._medir_terminal()
        self._buffer_comando: str = ""
        self._cursor_pos: int = 0
        self._aviso_sistema: str = "Listo. Escriba comando y pulse Enter (/v /n /d /pausar /reanudar /help /quit)."
        self._ts_aviso_manual: float = time.time()
        self._modo_tui_activo: bool = False
        self._cerrando_tui: bool = False
        self._ultima_firma_pie: str = ""
        self._dirty_input: bool = True

    @staticmethod
    def _medir_terminal() -> Tuple[int, int]:
        tam = shutil.get_terminal_size((100, 28))
        return max(44, tam.columns), max(12, tam.lines)

    def _prefijo_actual(self) -> str:
        if self._cols >= 128:
            return self.PREFIJO_CMD_LARGO
        if self._cols >= 98:
            return self.PREFIJO_CMD_MEDIO
        if self._cols >= 72:
            return self.PREFIJO_CMD_CORTO
        return self.PREFIJO_CMD_MINI

    @staticmethod
    def _recortar_visible(texto_plana: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(texto_plana) <= max_len:
            return texto_plana
        return texto_plana[: max(0, max_len - 1)] + "…"

    def iniciar_layout(self) -> None:
        """Limpia pantalla inicial y protege las 4 filas inferiores con DECSTBM."""
        with self._lock:
            self._cols, self._rows = self._medir_terminal()
            techo_scroll = max(5, self._rows - 4)
            sys.stdout.write(f"\x1b[?25h\x1b[2J\x1b[1;{techo_scroll}r\x1b[1;1H")
            sys.stdout.flush()
            self._modo_tui_activo = True
            self._cerrando_tui = False
            self._ultima_firma_pie = ""
            self._dirty_input = True

    def bloquear_para_cierre(self) -> None:
        """Impide que hilos en segundo plano vuelvan a escribir en la TUI al iniciar `/quit`."""
        with self._lock:
            self._cerrando_tui = True

    def restaurar_layout_normal(self, limpiar_pantalla: bool = False) -> None:
        """
        Restablece de forma segura e idempotente toda la consola (DECSTBM, auto-wrap,
        cursor visible, colores ANSI y modos nativos Win32).
        - Si `limpiar_pantalla=True` (flujo `/quit`), limpia el viewport después de `\x1b[r`
          para que nada previo se sobreimprima bajo el menú y resumen final.
        - Si `limpiar_pantalla=False` (`Ctrl+C`), limpia solo las 4 filas del pie manteniendo Zona 1.
        """
        with self._lock:
            estaba_activo = self._modo_tui_activo
            self._modo_tui_activo = False
            self._cerrando_tui = True
            try:
                if estaba_activo:
                    if limpiar_pantalla:
                        secuencia = f"{Colores.RESET}\x1b[?7h\x1b[r\x1b[2J\x1b[H\x1b[?25h"
                    else:
                        fila_pie = max(1, self._rows - 3)
                        secuencia = f"{Colores.RESET}\x1b[?7h\x1b[r\x1b[{fila_pie};1H\x1b[J\x1b[?25h"
                else:
                    secuencia = f"{Colores.RESET}\x1b[?7h\x1b[?25h"
                sys.stdout.write(secuencia)
                sys.stdout.flush()
            except Exception:
                pass
            try:
                restaurar_consola_windows(resetear_vt=False)
            except Exception:
                pass

    def establecer_aviso(self, mensaje: str, es_feedback_usuario: bool = True) -> None:
        with self._lock:
            self._aviso_sistema = mensaje
            if es_feedback_usuario:
                self._ts_aviso_manual = time.time()
            self._dirty_input = True

    # ------------------------------------------------------------------
    # Operaciones seguras sobre el buffer de comandos y posición del cursor
    # ------------------------------------------------------------------

    def obtener_buffer(self) -> str:
        with self._lock:
            return self._buffer_comando

    def establecer_buffer(self, nuevo_texto: str) -> None:
        with self._lock:
            self._buffer_comando = nuevo_texto[:220]
            self._cursor_pos = len(self._buffer_comando)
            self._dirty_input = True

    def agregar_texto_input(self, texto: str) -> None:
        with self._lock:
            if not texto:
                return
            restante = max(0, 220 - len(self._buffer_comando))
            if restante <= 0:
                return
            insercion = texto[:restante]
            pos = max(0, min(self._cursor_pos, len(self._buffer_comando)))
            self._buffer_comando = (
                self._buffer_comando[:pos] + insercion + self._buffer_comando[pos:]
            )
            self._cursor_pos = pos + len(insercion)
            self._dirty_input = True

    def borrar_ultimo_caracter(self) -> None:
        with self._lock:
            pos = max(0, min(self._cursor_pos, len(self._buffer_comando)))
            if pos > 0:
                self._buffer_comando = self._buffer_comando[: pos - 1] + self._buffer_comando[pos:]
                self._cursor_pos = pos - 1
                self._dirty_input = True

    def borrar_caracter_adelante(self) -> None:
        with self._lock:
            pos = max(0, min(self._cursor_pos, len(self._buffer_comando)))
            if pos < len(self._buffer_comando):
                self._buffer_comando = self._buffer_comando[:pos] + self._buffer_comando[pos + 1 :]
                self._dirty_input = True

    def mover_cursor(self, delta: int) -> None:
        with self._lock:
            self._cursor_pos = max(0, min(len(self._buffer_comando), self._cursor_pos + delta))
            self._dirty_input = True

    def mover_cursor_extremo(self, al_inicio: bool) -> None:
        with self._lock:
            self._cursor_pos = 0 if al_inicio else len(self._buffer_comando)
            self._dirty_input = True

    def limpiar_buffer_input(self) -> None:
        with self._lock:
            self._buffer_comando = ""
            self._cursor_pos = 0
            self._dirty_input = True

    def confirmar_comando_actual(self) -> str:
        with self._lock:
            cmd = self._buffer_comando.strip()
            self._buffer_comando = ""
            self._cursor_pos = 0
            self._dirty_input = True
            return cmd

    def agregar_caracter_input(self, ch: str) -> Optional[str]:
        if ch in ("\r", "\n"):
            return self.confirmar_comando_actual()
        if ch in ("\x08", "\x7f"):
            self.borrar_ultimo_caracter()
            return None
        if ch == "\x1b":
            self.limpiar_buffer_input()
            return None
        if ch.isprintable():
            self.agregar_texto_input(ch)
        return None

    # ------------------------------------------------------------------
    # Impresión Zona 1 y Renderizado Diferencial del Pie
    # ------------------------------------------------------------------

    def _sincronizar_tamano_si_cambio(self) -> bool:
        """
        Actualiza `_cols, _rows` sin borrar el contenido de Zona 1 (evita pantallas negras
        y mantiene visible la línea de comandos ante cualquier movimiento de ventana).
        """
        cols, rows = self._medir_terminal()
        if (cols, rows) == (self._cols, self._rows):
            return False

        old_rows = self._rows
        self._cols, self._rows = cols, rows
        techo = max(5, rows - 4)
        # Limpiar las filas viejas del pie si la ventana creció hacia abajo
        limpiar_extras = ""
        if rows > old_rows:
            for r in range(max(1, old_rows - 3), rows + 1):
                limpiar_extras += f"\x1b[{r};1H\x1b[K"
        sys.stdout.write(f"\x1b[1;{techo}r{limpiar_extras}")
        self._ultima_firma_pie = ""
        self._dirty_input = True
        return True

    def imprimir_bloque_zona_1(self, texto_multilinea: str) -> None:
        """
        Imprime bloques de alto potencial en Zona 1 preservando intactas las 4 filas inferiores
        y la posición real del cursor en la línea interactiva.
        """
        with self._lock:
            if self._cerrando_tui:
                return
            if not self._modo_tui_activo:
                print(texto_multilinea, flush=True)
                return

            self._sincronizar_tamano_si_cambio()
            ancho_z1 = max(38, self._cols - 1)
            fila_scroll = max(5, self._rows - 4)
            sys.stdout.write(f"\x1b[{fila_scroll};1H")
            for linea in texto_multilinea.splitlines():
                ln_segura = recortar_ansi_visible(linea, ancho_z1)
                sys.stdout.write(f"\n{ln_segura}\x1b[K")
            sys.stdout.write("\n")
            self._posicionar_cursor_en_input()
            sys.stdout.flush()

    def _formatear_linea_input_y_offset(self, ancho: int) -> Tuple[str, int]:
        prefijo_cmd = self._prefijo_actual()
        disp_buf = max(8, ancho - len(prefijo_cmd))
        buf = self._buffer_comando
        pos = max(0, min(self._cursor_pos, len(buf)))

        if len(buf) <= disp_buf:
            buf_vis = buf
            pos_vis = pos
        else:
            # Ventana deslizante centrada alrededor del cursor del usuario
            inicio = max(0, pos - disp_buf + 4)
            if inicio + disp_buf > len(buf):
                inicio = max(0, len(buf) - disp_buf)
            fragmento = buf[inicio : inicio + disp_buf]
            if inicio > 0 and len(fragmento) > 1:
                buf_vis = "…" + fragmento[1:]
            else:
                buf_vis = fragmento
            pos_vis = max(0, min(len(buf_vis), pos - inicio))

        linea_prompt = recortar_ansi_visible(
            f"{Colores.BG_PROMPT}{Colores.CIAN}{Colores.BOLD}{prefijo_cmd}{Colores.RESET}"
            f"{Colores.BG_PROMPT}{Colores.BLANCO}{buf_vis}{Colores.RESET}",
            ancho,
        )
        col_cursor = min(self._cols, len(prefijo_cmd) + pos_vis + 1)
        return linea_prompt, col_cursor

    def renderizar_solo_input(self) -> None:
        """Refresca instantáneamente únicamente la última fila al teclear (0 parpadeo, <1ms)."""
        with self._lock:
            if not self._modo_tui_activo or self._cerrando_tui:
                return
            self._sincronizar_tamano_si_cambio()
            ancho = max(40, self._cols - 1)
            linea_prompt, col_cursor = self._formatear_linea_input_y_offset(ancho)
            r_cmd = self._rows
            sys.stdout.write(
                f"\x1b[?7l\x1b[{r_cmd};1H{linea_prompt}\x1b[K\x1b[?7h\x1b[?25h\x1b[{r_cmd};{col_cursor}H"
            )
            sys.stdout.flush()
            self._dirty_input = False

    def renderizar_pie_tres_lineas(
        self,
        stats_ai: Dict[str, Any],
        sup_noticias: Dict[str, Any],
        forzar: bool = False,
    ) -> None:
        """
        Actualiza el bloque inferior únicamente cuando los datos visibles cambiaron o se fuerza:
        - `rows - 3`: Divisor + aviso interactivo / régimen horario
        - `rows - 2`: `Supervisión AI` (resaltado morado) + métricas + tickers evaluados por consulta
        - `rows - 1`: `Supervisión Noticias` (resaltado azul) + salud + último titular
        - `rows`    : Consola interactiva de comandos
        """
        with self._lock:
            if not self._modo_tui_activo or self._cerrando_tui:
                return

            cambio_tam = self._sincronizar_tamano_si_cambio()
            ancho = max(40, self._cols - 1)

            # 1. Línea separadora superior del pie (rows - 3)
            detalle_ai = str(stats_ai.get("detalle_alerta", "")).strip()
            ia_pausada = bool(stats_ai.get("consultas_pausadas"))
            ia_esperando = bool(stats_ai.get("esperando_evento_reanudacion"))
            mostrar_aviso_reciente = bool(self._buffer_comando) or ((time.time() - self._ts_aviso_manual) < 10.0)
            if detalle_ai and not mostrar_aviso_reciente:
                texto_barra = f"[Alerta IA] {detalle_ai} (Use /apikey <clave>)"
                col_div = Colores.AMARILLO
            elif ia_pausada and not mostrar_aviso_reciente:
                n_off = int(stats_ai.get("noticias_durante_pausa", 0))
                e_off = int(stats_ai.get("eventos_tec_durante_pausa", 0))
                ini_off = str(stats_ai.get("inicio_pausa_str", "")).strip()
                texto_barra = (
                    f"[IA PAUSADA OFFLINE desde {ini_off or '--'}] "
                    f"Not:{n_off} Ind:{e_off} — Datos/Noticias/Cartera activos (/reanudar)"
                )
                col_div = Colores.AMARILLO
            elif ia_esperando and not mostrar_aviso_reciente:
                n_off = int(stats_ai.get("noticias_durante_pausa", 0))
                e_off = int(stats_ai.get("eventos_tec_durante_pausa", 0))
                texto_barra = (
                    f"[IA REANUDADA] Esperando evento ALCISTA/BAJISTA para consultar "
                    f"(Contexto offline: {n_off} not., {e_off} ind.)"
                )
                col_div = Colores.CIAN
            else:
                reg_str = str(stats_ai.get("regimen_horario", "")).strip()
                if reg_str and not mostrar_aviso_reciente:
                    texto_barra = f"{self._aviso_sistema} [{reg_str}]"
                else:
                    texto_barra = self._aviso_sistema
                col_div = Colores.GRIS

            aviso_corto = self._recortar_visible(texto_barra, max(12, ancho - 6))
            relleno_guiones = "─" * max(1, ancho - len(aviso_corto) - 5)
            linea_div = recortar_ansi_visible(
                f"{col_div}── {aviso_corto} {relleno_guiones}{Colores.RESET}",
                ancho,
            )

            # 2. Antepenúltima línea (rows - 2): Supervisión AI (morado) + tickers por consulta
            tot_c = int(stats_ai.get("total_consultas", 0))
            tot_ev = int(stats_ai.get("total_activos_evaluados", 0))
            alc = int(stats_ai.get("alcistas", 0))
            alc_p = float(stats_ai.get("alcistas_pct", 0.0))
            baj = int(stats_ai.get("bajistas", 0))
            baj_p = float(stats_ai.get("bajistas_pct", 0.0))
            neu = int(stats_ai.get("neutrales", 0))
            neu_p = float(stats_ai.get("neutrales_pct", 0.0))
            est_ag = str(stats_ai.get("estado_agente", "LISTO"))
            err_con = int(stats_ai.get("fallos_conexion", 0))
            err_fon = int(stats_ai.get("fallos_fondos", 0))

            ult_tks: List[str] = list(stats_ai.get("ultimos_tickers_consulta", []))
            cola_tks: List[str] = list(stats_ai.get("tickers_cola", []))
            if ult_tks:
                tks_render = ",".join(ult_tks)
            elif cola_tks:
                tks_render = "cola:" + ",".join(cola_tks)
            else:
                tks_render = "-"

            if "FONDOS" in est_ag or "API KEY" in est_ag:
                color_est_ai = Colores.ROJO
            elif "PAUSADA" in est_ag or ia_pausada:
                color_est_ai = Colores.AMARILLO
            elif "ESPERANDO" in est_ag or ia_esperando:
                color_est_ai = Colores.CIAN
            elif "CONEXIÓN" in est_ag:
                color_est_ai = Colores.NARANJA
            elif "CONSULTANDO" in est_ag:
                color_est_ai = Colores.CIAN
            else:
                color_est_ai = Colores.VERDE

            alerta_extra = f" | Err:{err_con + err_fon}" if (err_con > 0 or err_fon > 0) else ""
            ley_ai = leyenda_supervision_ai(compacta=(self._cols < 74))

            if self._cols >= 134:
                raw_sup_ai = (
                    f"{ley_ai}{Colores.BG_SUPERVISOR_AI} "
                    f"Cons:{Colores.BLANCO}{tot_c}({tot_ev}ev){Colores.RESET}{Colores.BG_SUPERVISOR_AI} | "
                    f"Alza:{Colores.VERDE}{alc}({alc_p:.0f}%){Colores.RESET}{Colores.BG_SUPERVISOR_AI} | "
                    f"Baja:{Colores.ROJO}{baj}({baj_p:.0f}%){Colores.RESET}{Colores.BG_SUPERVISOR_AI} | "
                    f"Neu:{Colores.AMARILLO}{neu}({neu_p:.0f}%){Colores.RESET}{Colores.BG_SUPERVISOR_AI} | "
                    f"{color_est_ai}{est_ag}{Colores.RESET}{Colores.BG_SUPERVISOR_AI}{alerta_extra} | "
                    f"Emp:[{Colores.CIAN}{tks_render}{Colores.RESET}{Colores.BG_SUPERVISOR_AI}]{Colores.RESET}"
                )
            else:
                est_comp = "ESPERANDO ALZA/BAJA" if est_ag == "ESPERANDO EVENTO ALZA/BAJA" else est_ag
                raw_sup_ai = (
                    f"{ley_ai}{Colores.BG_SUPERVISOR_AI} "
                    f"C:{Colores.BLANCO}{tot_c}{Colores.RESET}{Colores.BG_SUPERVISOR_AI} "
                    f"▲{Colores.VERDE}{alc}{Colores.RESET}{Colores.BG_SUPERVISOR_AI} "
                    f"▼{Colores.ROJO}{baj}{Colores.RESET}{Colores.BG_SUPERVISOR_AI} "
                    f"={Colores.AMARILLO}{neu}{Colores.RESET}{Colores.BG_SUPERVISOR_AI} | "
                    f"{color_est_ai}{self._recortar_visible(est_comp, 20)}{Colores.RESET}{Colores.BG_SUPERVISOR_AI}{alerta_extra} | "
                    f"[{Colores.CIAN}{tks_render}{Colores.RESET}{Colores.BG_SUPERVISOR_AI}]{Colores.RESET}"
                )
            linea_sup_ai = recortar_ansi_visible(raw_sup_ai, ancho)

            # 3. Penúltima línea (rows - 1): Supervisión Noticias (azul)
            est_not = str(sup_noticias.get("estado", "INICIALIZANDO"))
            nivel_not = str(sup_noticias.get("nivel", "OK"))
            hora_ult = str(sup_noticias.get("hora_ultima_noticia", "--:--:-- --"))
            seg_ult = int(sup_noticias.get("segundos_desde_ultima", 0))
            tk_ult = str(sup_noticias.get("ultimo_ticker", "-"))
            plat_ult = self._recortar_visible(str(sup_noticias.get("ultima_plataforma", "")), 14)
            tot_not = int(sup_noticias.get("total_noticias", 0))
            f_ok = int(sup_noticias.get("fuentes_ok", 0))
            f_tot = int(sup_noticias.get("fuentes_total", 0))

            if nivel_not == "OK":
                col_not = Colores.VERDE_FUERTE
            elif nivel_not == "ADVERTENCIA":
                col_not = Colores.AMARILLO
            else:
                col_not = Colores.ROJO

            ley_not = leyenda_supervision_noticias(compacta=(self._cols < 74))
            raw_sup_not = (
                f"{ley_not}{Colores.BG_SUPERVISION_NEWS} "
                f"{col_not}{est_not}{Colores.RESET}{Colores.BG_SUPERVISION_NEWS} | "
                f"{Colores.BLANCO}{hora_ult}{Colores.RESET}{Colores.BG_SUPERVISION_NEWS} ({seg_ult}s) | "
                f"{ticker_azul(tk_ult)}{Colores.BG_SUPERVISION_NEWS}, {plat_ult} | "
                f"Not:{tot_not} OK:{f_ok}/{f_tot}{Colores.RESET}"
            )
            linea_sup_not = recortar_ansi_visible(raw_sup_not, ancho)

            # 4. Última línea (rows): Entrada interactiva
            linea_prompt, col_cursor = self._formatear_linea_input_y_offset(ancho)

            firma = f"{self._cols}x{self._rows}|{linea_div}|{linea_sup_ai}|{linea_sup_not}|{linea_prompt}|{col_cursor}"
            if not forzar and not cambio_tam and not self._dirty_input and firma == self._ultima_firma_pie:
                return

            self._ultima_firma_pie = firma
            self._dirty_input = False

            r_div = self._rows - 3
            r_ai = self._rows - 2
            r_not = self._rows - 1
            r_cmd = self._rows

            salida = (
                "\x1b[?7l"
                f"\x1b[{r_div};1H{linea_div}\x1b[K"
                f"\x1b[{r_ai};1H{linea_sup_ai}\x1b[K"
                f"\x1b[{r_not};1H{linea_sup_not}\x1b[K"
                f"\x1b[{r_cmd};1H{linea_prompt}\x1b[K"
                "\x1b[?7h"
                f"\x1b[?25h\x1b[{r_cmd};{col_cursor}H"
            )
            sys.stdout.write(salida)
            sys.stdout.flush()

    def _posicionar_cursor_en_input(self) -> None:
        ancho = max(40, self._cols - 1)
        _, col_cursor = self._formatear_linea_input_y_offset(ancho)
        sys.stdout.write(f"\x1b[?25h\x1b[{self._rows};{col_cursor}H")


# ==============================================================================
# 2. GESTOR REDISEÑADO DE COMANDOS INTERACTIVOS (MSVC-RT / PIPES ASÍNCRONO)
# ==============================================================================

class GestorComandosInteractivos:
    """
    Reemplaza por completo la lectura bloqueante o invasiva de Win32:
    - JAMÁS desactiva `ENABLE_VIRTUAL_TERMINAL_INPUT` ni `ENABLE_LINE_INPUT`.
    - Opera en un hilo dedicado silencioso con `msvcrt.kbhit()`/`getwch()` (Windows) o `stdin` (pipe),
      permitiendo pausar limpiamente cuando `/quit` requiere `input()` estándar.
    - Soporta historial de comandos (`↑`/`↓`), cursor en línea (`←`/`→`/`Inicio`/`Fin`/`Supr`),
      autocompletado con `Tab`, pegado con `Ctrl+V`, limpieza rápida con `Esc` y detección de `Ctrl+C`.
    """

    COMANDOS_SUGERIDOS = (
        "/v",
        "/n",
        "/d",
        "/pausar",
        "/reanudar",
        "/modelo ",
        "/lote",
        "/apikey ",
        "/help",
        "/quit",
    )

    def __init__(self, consola: ConsolaCuatroZonas) -> None:
        self.consola = consola
        self._es_consola_win32 = os.name == "nt" and sys.stdin.isatty() and configurar_consola_tui_interactiva()
        self._cola_comandos: "queue.Queue[str]" = queue.Queue()
        self._historial: List[str] = []
        self._idx_hist: int = -1
        self._detener = threading.Event()
        self._pausado = threading.Event()
        self._pulso_ctrl_c = threading.Event()
        self._evento_actividad = threading.Event()
        self._hilo: Optional[threading.Thread] = None

    def iniciar(self) -> None:
        if self._hilo and self._hilo.is_alive():
            return
        self._detener.clear()
        self._pausado.clear()
        target = self._bucle_msvcrt if self._es_consola_win32 else self._bucle_pipe
        self._hilo = threading.Thread(target=target, name="InteractiveCmdReader", daemon=True)
        self._hilo.start()

    def pausar(self) -> None:
        """Cede la entrada estándar limpiamente para prompts síncronos como `/quit`."""
        self._pausado.set()
        time.sleep(0.05)

    def detener(self) -> None:
        self._detener.set()
        self._pausado.set()
        self._evento_actividad.set()

    def esperar_actividad(self, timeout: float = 0.25) -> None:
        self._evento_actividad.wait(timeout=timeout)
        self._evento_actividad.clear()

    def ctrl_c_pulsado(self) -> bool:
        return self._pulso_ctrl_c.is_set()

    def extraer_comandos(self) -> List[str]:
        items: List[str] = []
        while True:
            try:
                items.append(self._cola_comandos.get_nowait())
            except queue.Empty:
                break
        return items

    def _autocompletar_tab(self) -> None:
        actual = self.consola.obtener_buffer().lstrip().lower()
        if not actual:
            return
        coincidencias = [c for c in self.COMANDOS_SUGERIDOS if c.startswith(actual)]
        if len(coincidencias) == 1:
            self.consola.establecer_buffer(coincidencias[0])
        elif len(coincidencias) > 1:
            pref = os.path.commonprefix(coincidencias)
            if len(pref) > len(actual):
                self.consola.establecer_buffer(pref)
            self.consola.establecer_aviso("Opciones: " + "  ".join(c.strip() for c in coincidencias))

    def _navegar_historial(self, direccion: int) -> None:
        if not self._historial:
            return
        if self._idx_hist == -1:
            if direccion < 0:
                self._idx_hist = len(self._historial) - 1
            else:
                return
        else:
            self._idx_hist += direccion
            if self._idx_hist < 0:
                self._idx_hist = 0
            elif self._idx_hist >= len(self._historial):
                self._idx_hist = -1
                self.consola.limpiar_buffer_input()
                return
        self.consola.establecer_buffer(self._historial[self._idx_hist])

    def _confirmar_actual(self) -> None:
        cmd = self.consola.confirmar_comando_actual()
        if cmd and (not self._historial or self._historial[-1] != cmd):
            self._historial.append(cmd)
            if len(self._historial) > 30:
                del self._historial[:-30]
        self._idx_hist = -1
        self._cola_comandos.put(cmd)

    def _bucle_msvcrt(self) -> None:
        """
        Lee pulsaciones sin alterar modos VT de ConPTY usando `msvcrt.kbhit()`/`getwch()`.
        """
        while not self._detener.is_set():
            if self._pausado.is_set():
                time.sleep(0.05)
                continue
            try:
                if not msvcrt.kbhit():
                    time.sleep(0.015)
                    continue

                ch = msvcrt.getwch()
                cambio = False

                # 1. Ctrl+C
                if ch == "\x03":
                    self._pulso_ctrl_c.set()
                    self._evento_actividad.set()
                    return

                # 2. Teclas extendidas de Windows (\x00 o \xe0): flechas, Supr, Inicio, Fin
                if ch in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        ext = msvcrt.getwch()
                        if ext == "H":      # Flecha arriba
                            self._navegar_historial(-1)
                            cambio = True
                        elif ext == "P":    # Flecha abajo
                            self._navegar_historial(+1)
                            cambio = True
                        elif ext == "K":    # Flecha izquierda
                            self.consola.mover_cursor(-1)
                            cambio = True
                        elif ext == "M":    # Flecha derecha
                            self.consola.mover_cursor(+1)
                            cambio = True
                        elif ext == "G":    # Inicio
                            self.consola.mover_cursor_extremo(True)
                            cambio = True
                        elif ext == "O":    # Fin
                            self.consola.mover_cursor_extremo(False)
                            cambio = True
                        elif ext == "S":    # Supr
                            self.consola.borrar_caracter_adelante()
                            cambio = True
                    if cambio:
                        self.consola.renderizar_solo_input()
                        self._evento_actividad.set()
                    continue

                # 3. Secuencia ESC o posible secuencia VT (\x1b[...)
                if ch == "\x1b":
                    time.sleep(0.008)
                    if msvcrt.kbhit():
                        sig = msvcrt.getwch()
                        if sig == "[":
                            seq = ""
                            while msvcrt.kbhit() and len(seq) < 8:
                                c_s = msvcrt.getwch()
                                seq += c_s
                                if "@" <= c_s <= "~":
                                    break
                            if seq.endswith("A"):
                                self._navegar_historial(-1)
                                cambio = True
                            elif seq.endswith("B"):
                                self._navegar_historial(+1)
                                cambio = True
                            elif seq.endswith("D"):
                                self.consola.mover_cursor(-1)
                                cambio = True
                            elif seq.endswith("C"):
                                self.consola.mover_cursor(+1)
                                cambio = True
                            elif seq in ("H", "1~", "7~"):
                                self.consola.mover_cursor_extremo(True)
                                cambio = True
                            elif seq in ("F", "4~", "8~"):
                                self.consola.mover_cursor_extremo(False)
                                cambio = True
                            elif seq.endswith("3~"):
                                self.consola.borrar_caracter_adelante()
                                cambio = True
                        if cambio:
                            self.consola.renderizar_solo_input()
                            self._evento_actividad.set()
                        continue
                    # ESC físico solo -> limpiar línea actual
                    self.consola.limpiar_buffer_input()
                    self.consola.renderizar_solo_input()
                    self._evento_actividad.set()
                    continue

                # 4. Enter
                if ch in ("\r", "\n"):
                    self._confirmar_actual()
                    self.consola.renderizar_solo_input()
                    self._evento_actividad.set()
                    continue

                # 5. Tab -> autocompletar comando
                if ch == "\t":
                    self._autocompletar_tab()
                    self.consola.renderizar_solo_input()
                    self._evento_actividad.set()
                    continue

                # 6. Backspace
                if ch in ("\x08", "\x7f"):
                    self.consola.borrar_ultimo_caracter()
                    self.consola.renderizar_solo_input()
                    self._evento_actividad.set()
                    continue

                # 7. Ctrl+V (pegar portapapeles)
                if ch == "\x16":
                    pegado = leer_portapapeles_windows()
                    if pegado:
                        self.consola.agregar_texto_input(pegado)
                        self.consola.renderizar_solo_input()
                        self._evento_actividad.set()
                    continue

                # 8. Ctrl+U (limpiar línea rápida)
                if ch == "\x15":
                    self.consola.limpiar_buffer_input()
                    self.consola.renderizar_solo_input()
                    self._evento_actividad.set()
                    continue

                # 9. Caracteres Unicode imprimibles (incluyendo /, Shift+7, AltGr, API keys)
                if ch and ord(ch) >= 32 and ch.isprintable():
                    self.consola.agregar_texto_input(ch)
                    self.consola.renderizar_solo_input()
                    self._evento_actividad.set()
            except Exception:
                time.sleep(0.05)

    def _bucle_pipe(self) -> None:
        try:
            for linea in sys.stdin:
                if self._detener.is_set():
                    break
                self._cola_comandos.put(linea.rstrip("\r\n"))
                self._evento_actividad.set()
        except Exception:
            pass


# ==============================================================================
# 3. FORMATEADOR EXACTO DEL BLOQUE DE ALTO POTENCIAL EN LA TERMINAL PRINCIPAL
# ==============================================================================

def formatear_bloque_alto_potencial(ev: EventoImpresionPrincipal) -> str:
    """
    Construye el bloque exacto requerido para la Zona 1 de la terminal principal cuando
    `analisis_ai` responde que la acción tiene alto potencial alcista o bajista:
    -> Fecha y hora (con segs AM/PM) | [ticker en morado] | predicción (alcista en verde / bajista en rojo)
    -> Prompt que se le envía al modelo
    -> Respuesta del modelo
    -> (Noticias analizadas) Fecha y hora (AM/PM) | [ticker en azul] | plataforma (verde pos / rojo neg) | link
    -> Datos financieros de la empresa (precio, P/E, soporte y resistencia, etc.)
    -> Indicadores
    """
    pred_upper = ev.prediccion_movimiento.strip().upper()
    if pred_upper == "ALCISTA":
        pred_coloreada = f"{Colores.VERDE_FUERTE}ALCISTA{Colores.RESET}"
    else:
        pred_coloreada = f"{Colores.ROJO}BAJISTA{Colores.RESET}"

    df = ev.analisis_tecnico.datos_financieros
    pe_s = f"{df.pe_ratio:.2f}" if df.pe_ratio is not None else "N/D"
    fpe_s = f"{df.pe_forward:.2f}" if df.pe_forward is not None else "N/D"
    mcap_s = f"${df.cap_mercado_b:.2f}B" if df.cap_mercado_b is not None else "N/D"
    r52 = (
        f"${df.min_52_semanas:.2f} - ${df.max_52_semanas:.2f}"
        if (df.min_52_semanas is not None and df.max_52_semanas is not None)
        else "N/D"
    )

    lineas: List[str] = []
    lineas.append(f"{Colores.MORADO}{'═' * 94}{Colores.RESET}")
    lineas.append(f"-> {ev.fecha_hora_ampm} | {ticker_morado(ev.ticker)} | {pred_coloreada}")

    lineas.append(f"-> {Colores.CIAN}{Colores.BOLD}Prompt enviado al modelo:{Colores.RESET}")
    for ln_p in ev.prompt_enviado.splitlines():
        lineas.append(f"   {Colores.GRIS}{ln_p}{Colores.RESET}")

    lineas.append(f"-> {Colores.CIAN}{Colores.BOLD}Respuesta del modelo:{Colores.RESET}")
    for ln_r in ev.respuesta_modelo_texto.splitlines():
        lineas.append(f"   {Colores.BLANCO}{ln_r}{Colores.RESET}")

    lineas.append(f"-> {Colores.CIAN}{Colores.BOLD}Noticias analizadas:{Colores.RESET}")
    if ev.noticias_analizadas:
        for n in ev.noticias_analizadas:
            plat_col = colorear_plataforma_noticia(n.plataforma, n.sentimiento)
            lineas.append(f"   {n.fecha_hora_ampm} | {ticker_azul(n.ticker)} | {plat_col} | {n.link}")
    else:
        lineas.append(f"   {Colores.GRIS}(Análisis impulsado por ruptura cuantitativa / estructura técnica){Colores.RESET}")

    lineas.append(
        f"-> {Colores.CIAN}{Colores.BOLD}Datos financieros:{Colores.RESET} "
        f"Precio=${df.precio_actual:.2f} ({df.cambio_diario_pct:+.2f}%) | "
        f"P/E={pe_s} (Fwd={fpe_s}) | Cap.Mercado={mcap_s} | "
        f"Soporte=${df.soporte_1:.2f}/${df.soporte_2:.2f} | "
        f"Resistencia=${df.resistencia_1:.2f}/${df.resistencia_2:.2f} | "
        f"VWAP=${df.vwap_sesion:.2f} | Rango 52s={r52}"
    )

    lineas.append(
        f"-> {Colores.CIAN}{Colores.BOLD}Indicadores (Evaluación General: "
        f"{colorear_evaluacion(ev.analisis_tecnico.evaluacion_general)}):{Colores.RESET}"
    )
    for ind in ev.analisis_tecnico.indicadores:
        nom_c = colorear_nombre_indicador(ind.nombre, ind.alarma)
        ev_c = colorear_evaluacion(ind.evaluacion)
        lineas.append(f"   • {ind.categoria:<22} | {nom_c}: {ind.texto_valor} -> {ev_c}")

    lineas.append(f"{Colores.MORADO}{'═' * 94}{Colores.RESET}")
    return "\n".join(lineas)


# ==============================================================================
# 4. ASISTENTE INTERACTIVO DE INICIALIZACIÓN (10 MODELOS + ESFUERZOS)
# ==============================================================================

def _seleccionar_modelo_y_esfuerzo() -> Tuple[str, str]:
    """Permite elegir uno de los 10 modelos de IA y su nivel de esfuerzo soportado."""
    opciones_modelos = list(MODELOS_IA_DISPONIBLES.keys())
    print(f"\n{Colores.BOLD}1) Seleccione el modelo de IA encargado del análisis:{Colores.RESET}")
    for idx, clave in enumerate(opciones_modelos, start=1):
        meta = MODELOS_IA_DISPONIBLES[clave]
        prov_tag = "Gemini" if meta["proveedor"] == "gemini" else "Claude"
        print(
            f"   [{idx:>2}] {Colores.MORADO}{meta['nombre_mostrar']:<10}{Colores.RESET} "
            f"({prov_tag:<6}) — {meta['descripcion']}"
        )

    modelo_elegido = "flash3.8"
    while True:
        resp_m = input(f"{Colores.CIAN}   Opción (1-10 o nombre, Enter=Flash 3.8): {Colores.RESET}").strip()
        resuelto = resolver_clave_modelo(resp_m)
        if resuelto is not None:
            modelo_elegido = resuelto
            break
        print(
            f"   {Colores.ROJO}Selección no válida. Elija del 1 al 10 o escriba el nombre "
            f"(flash3.6, flash3.7, flash3.8, argon, hydrogen, carbon, pro3.1, sonnet, opus, fable).{Colores.RESET}"
        )

    meta_sel = MODELOS_IA_DISPONIBLES[modelo_elegido]
    niveles: List[str] = list(meta_sel["niveles_esfuerzo"])
    default_esf = str(meta_sel["esfuerzo_default"])

    print(
        f"\n   {Colores.BOLD}1b) Seleccione el nivel de esfuerzo para "
        f"{Colores.MORADO}{meta_sel['nombre_mostrar']}{Colores.RESET}:{Colores.RESET}"
    )
    for idx_e, niv in enumerate(niveles, start=1):
        marca = f" {Colores.VERDE}(recomendado){Colores.RESET}" if niv == default_esf else ""
        print(f"      [{idx_e}] {Colores.CIAN}{niv}{Colores.RESET}{marca}")

    esfuerzo_elegido = default_esf
    while True:
        resp_e = input(
            f"{Colores.CIAN}      Esfuerzo (1-{len(niveles)} o nombre, Enter={default_esf}): {Colores.RESET}"
        ).strip().lower()
        if not resp_e:
            esfuerzo_elegido = default_esf
            break
        if resp_e.isdigit() and 1 <= int(resp_e) <= len(niveles):
            esfuerzo_elegido = niveles[int(resp_e) - 1]
            break
        if resp_e in niveles or (modelo_elegido == "pro3.1" and resp_e == "medium"):
            esfuerzo_elegido = normalizar_esfuerzo_modelo(modelo_elegido, resp_e)
            break
        print(f"      {Colores.ROJO}Nivel inválido. Opciones permitidas: {', '.join(niveles)}.{Colores.RESET}")

    return modelo_elegido, esfuerzo_elegido


def ejecutar_asistente_inicializacion(info_limpieza: Dict) -> Dict[str, Any]:
    """Solicita las opciones iniciales al usuario (modelo + esfuerzo, clave, modo, lista, USD) y muestra resumen."""
    print(f"{Colores.CIAN}{Colores.BOLD}{'═' * 86}{Colores.RESET}")
    print(f"{Colores.CIAN}{Colores.BOLD}   VAGY-M1 — SISTEMA ORQUESTADOR MULTI-TERMINAL (MÉTODO DE CONSULTA MÚLTIPLE){Colores.RESET}")
    print(f"{Colores.CIAN}{Colores.BOLD}{'═' * 86}{Colores.RESET}")

    movidos = info_limpieza.get("archivos_movidos_a_hystmp", [])
    if movidos:
        print(
            f"{Colores.AMARILLO}[Historial Limpio]{Colores.RESET} Se archivaron {len(movidos)} elemento(s) "
            f"previos en `{Colores.CIAN}hystmp/{Colores.RESET}`."
        )
    else:
        print(f"{Colores.VERDE}[Historial Limpio]{Colores.RESET} Espacio de trabajo verificado y `AGENTS.md` activo.")

    # 1 y 2. Modelo + Esfuerzo + API Key verificada en vivo
    modelo_elegido = "flash3.8"
    esfuerzo_elegido = "medium"
    api_key = ""
    modo_transporte = "gemini_interactions"
    conexion_verificada = False
    mensaje_conexion = ""
    pedir_modelo = True

    while True:
        if pedir_modelo:
            modelo_elegido, esfuerzo_elegido = _seleccionar_modelo_y_esfuerzo()

        meta_sel = MODELOS_IA_DISPONIBLES[modelo_elegido]
        proveedor = str(meta_sel["proveedor"])
        env_sugerida = (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if proveedor == "gemini"
            else os.environ.get("ANTHROPIC_API_KEY")
        )
        aviso_env = " (Enter para usar variable de entorno detectada)" if env_sugerida else ""
        guia_key = (
            f"   {Colores.GRIS}Tip: Obtenga una API Key directa en https://aistudio.google.com/apikey{Colores.RESET}\n"
            if proveedor == "gemini"
            else ""
        )
        print(
            f"\n{Colores.BOLD}2) Ingrese la API Key para '{meta_sel['nombre_mostrar']}' "
            f"(effort={esfuerzo_elegido}, {proveedor.upper()}){aviso_env}:{Colores.RESET}\n{guia_key}",
            end="",
        )
        raw_input_key = input(f"{Colores.CIAN}   API Key: {Colores.RESET}")
        if not raw_input_key.strip() and env_sugerida:
            api_key = limpiar_api_key(env_sugerida)
        else:
            api_key = limpiar_api_key(raw_input_key)

        if not api_key:
            conexion_verificada = False
            mensaje_conexion = "No se ingresó API Key. Puede agregarla luego en vivo con: /apikey TU_CLAVE"
            print(f"   {Colores.AMARILLO}⚠ {mensaje_conexion}{Colores.RESET}")
            break

        print(
            f"   {Colores.CIAN}Verificando conexión en vivo con {meta_sel['model_id']} "
            f"(effort={esfuerzo_elegido})...{Colores.RESET}"
        )
        ok_conn, modo_transporte, msg_conn = probar_conexion_modelo_ia(
            modelo_elegido, api_key, nivel_esfuerzo=esfuerzo_elegido
        )
        conexion_verificada = ok_conn
        mensaje_conexion = msg_conn

        if ok_conn:
            print(f"   {Colores.VERDE_FUERTE}✓ {msg_conn}{Colores.RESET}")
            break

        print(f"   {Colores.ROJO}✗ Fallo al conectar con el modelo:{Colores.RESET} {Colores.AMARILLO}{msg_conn}{Colores.RESET}")
        if not sys.stdin.isatty():
            break

        print(
            f"   {Colores.BLANCO}¿Qué desea hacer? "
            f"[1] Ingresar otra API Key  |  [2] Cambiar modelo/esfuerzo  |  [3] Continuar sin IA activa{Colores.RESET}"
        )
        reintento = input(f"{Colores.CIAN}   Opción (Enter=1): {Colores.RESET}").strip()
        if reintento in ("", "1"):
            pedir_modelo = False
            continue
        if reintento == "2":
            pedir_modelo = True
            continue
        break

    # 3. Tipo de operación
    print(f"\n{Colores.BOLD}3) Seleccione el tipo de operación:{Colores.RESET}")
    print("   [1] Spot (sport)    — Solo posiciones de compra / LONG (sin cortos al descubierto)")
    print("   [2] Short y Long    — Operaciones bidireccionales al alza (LONG) y a la baja (SHORT)")
    modo_operacion = "spot"
    etiqueta_modo = "Spot (sport)"
    while True:
        resp_op = input(f"{Colores.CIAN}   Opción (1=spot/sport, 2=short y long, Enter=1): {Colores.RESET}").strip().lower()
        if resp_op in ("", "1", "spot", "sport"):
            modo_operacion = "spot"
            etiqueta_modo = "Spot (sport)"
            break
        if resp_op in ("2", "short y long", "short", "long y short"):
            modo_operacion = "short y long"
            etiqueta_modo = "Short y Long"
            break
        print(f"   {Colores.ROJO}Ingrese 1 ('spot'/'sport') o 2 ('short y long').{Colores.RESET}")

    # 4. Lista de acciones
    print(f"\n{Colores.BOLD}4) Seleccione la lista de acciones a monitorear:{Colores.RESET}")
    print(f"   [1] Lista 1 ({len(LISTA_ACCIONES_1)} acciones: PANW, LITE, CLS, STRL, CIEN ... NVDA, AMD, GOOGL)")
    print(f"   [2] Lista 2 ({len(LISTA_ACCIONES_2)} acciones: GOOGL, ASML, LLY, AVGO, V ... MRVL, NBIS, MP, FLEX, QCOM)")
    while True:
        resp_l = input(f"{Colores.CIAN}   Opción (1 o 2, Enter=1): {Colores.RESET}").strip().lower()
        if resp_l in ("", "1", "lista 1"):
            tickers_sel = list(LISTA_ACCIONES_1)
            lista_nombre = "Lista 1"
            break
        if resp_l in ("2", "lista 2"):
            tickers_sel = list(LISTA_ACCIONES_2)
            lista_nombre = "Lista 2"
            break
        print(f"   {Colores.ROJO}Seleccione 1 para Lista 1 o 2 para Lista 2.{Colores.RESET}")

    # 5. Monto inicial virtual
    print(f"\n{Colores.BOLD}5) Ingrese el monto de capital inicial virtual en USD (ej. 1000):{Colores.RESET}")
    capital_usd = 1000.0
    while True:
        resp_cap = input(f"{Colores.CIAN}   Capital virtual USD (Enter=1000): {Colores.RESET}").strip().replace(",", "").replace("$", "")
        if not resp_cap:
            capital_usd = 1000.0
            break
        try:
            val = float(resp_cap)
            if val > 0:
                capital_usd = val
                break
        except ValueError:
            pass
        print(f"   {Colores.ROJO}Ingrese un monto numérico mayor a 0 (ej. 1000).{Colores.RESET}")

    if len(api_key) > 8:
        key_mask = f"{api_key[:4]}...{api_key[-4:]}"
    elif api_key:
        key_mask = "**** (configurada)"
    else:
        key_mask = "NO CONFIGURADA (/apikey <clave>)"

    estado_key_res = (
        f"{Colores.VERDE}VERIFICADA OK{Colores.RESET}"
        if conexion_verificada
        else f"{Colores.AMARILLO}PENDIENTE / RECHAZADA (usar /apikey <clave>){Colores.RESET}"
    )

    reg_inicial = obtener_regimen_horario_mercado()
    meta_final = MODELOS_IA_DISPONIBLES[modelo_elegido]

    # 6. Resumen de configuración seleccionada
    print(f"\n{Colores.VERDE_FUERTE}{'═' * 86}{Colores.RESET}")
    print(f"{Colores.VERDE_FUERTE}   RESUMEN DE CONFIGURACIÓN SELECCIONADA{Colores.RESET}")
    print(f"{Colores.VERDE_FUERTE}{'═' * 86}{Colores.RESET}")
    print(
        f"  • Modelo de IA        : {Colores.MORADO}{Colores.BOLD}{meta_final['nombre_mostrar']}{Colores.RESET} "
        f"({meta_final['model_id']}) | Esfuerzo: {Colores.CIAN}{esfuerzo_elegido}{Colores.RESET}"
    )
    print(f"  • Método de Consulta  : {Colores.CIAN}Consulta Múltiple por Lotes Sectoriales + Confluencia Horaria{Colores.RESET} [{reg_inicial['etiqueta']}]")
    print(f"  • API Key             : {key_mask} [{estado_key_res}]")
    print(f"  • Tipo de Operación   : {Colores.CIAN}{etiqueta_modo}{Colores.RESET}")
    print(f"  • Lista de Acciones   : {Colores.AZUL}{lista_nombre}{Colores.RESET} ({len(tickers_sel)} tickers activos)")
    print(f"  • Capital Virtual     : {Colores.VERDE}${capital_usd:,.2f} USD{Colores.RESET} (Operación principal en {Colores.BOLD}BASE PORCENTUAL %{Colores.RESET})")
    print(f"{Colores.VERDE_FUERTE}{'═' * 86}{Colores.RESET}\n")

    return {
        "modelo_clave": modelo_elegido,
        "modelo_mostrar": meta_final["nombre_mostrar"],
        "nivel_esfuerzo": esfuerzo_elegido,
        "model_id": meta_final["model_id"],
        "api_key": api_key,
        "modo_transporte": modo_transporte,
        "conexion_verificada": conexion_verificada,
        "mensaje_conexion": mensaje_conexion,
        "modo_operacion": modo_operacion,
        "etiqueta_modo": etiqueta_modo,
        "lista_nombre": lista_nombre,
        "tickers": tickers_sel,
        "capital_usd": capital_usd,
    }


# ==============================================================================
# 5. GESTIÓN DEL COMANDO /quit Y APAGADO LIMPIO SIN BLOQUEOS
# ==============================================================================

def _construir_y_exportar_reportes_finales(
    config_sesion: Dict[str, Any],
    nombre_cierre: str,
    infomarket: GestorInfoMarket,
    cartera: CarteraVirtual,
    analizador_ai: AnalizadorAI,
    historial: GestorHistorial,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    snap_final = cartera.construir_snapshot_estado()
    stats_ai = analizador_ai.obtener_estadisticas_supervisor()

    resumen_final: Dict[str, Any] = {
        "fecha_inicio": cartera.fecha_inicio_str,
        "fecha_cierre": formatear_fecha_hora_ampm(),
        "modelo_ai": f"{config_sesion['modelo_clave']} ({config_sesion.get('nivel_esfuerzo', 'medium')})",
        "model_id": config_sesion["model_id"],
        "nivel_esfuerzo": config_sesion.get("nivel_esfuerzo", "medium"),
        "modo_operacion": config_sesion["etiqueta_modo"],
        "lista_nombre": config_sesion["lista_nombre"],
        "total_tickers": len(config_sesion["tickers"]),
        "modo_cierre": nombre_cierre,
        "capital_inicial_usd": snap_final["capital_inicial_usd"],
        "patrimonio_final_usd": snap_final["patrimonio_actual_usd"],
        "rendimiento_total_pct": snap_final["rendimiento_total_pct"],
        "pyg_total_usd": snap_final["pyg_total_usd"],
        "operaciones_cerradas": snap_final["operaciones_cerradas"],
        "rotaciones_ejecutadas": snap_final["rotaciones_ejecutadas"],
        "win_rate_pct": snap_final["win_rate_pct"],
        "ai_total_consultas": stats_ai["total_consultas"],
        "ai_activos_evaluados": stats_ai["total_activos_evaluados"],
        "ai_consultas_ahorradas_lote": stats_ai["ahorradas_lote"],
        "ai_consultas_filtradas": stats_ai["filtradas_gatekeeper"],
        "ai_alcistas": stats_ai["alcistas"],
        "ai_bajistas": stats_ai["bajistas"],
        "ai_neutrales": stats_ai["neutrales"],
        "ai_pausas_offline": stats_ai.get("total_pausas", 0),
        "ai_errores": int(stats_ai["fallos_conexion"]) + int(stats_ai["fallos_fondos"]),
        "total_noticias": infomarket.total_noticias_capturadas(),
    }

    rutas_exportadas = historial.exportar_cierre_sesion(
        resumen_datos=resumen_final,
        historial_posiciones=cartera.historial_cerradas,
    )
    return resumen_final, rutas_exportadas


def ejecutar_apagado_inmediato_ctrl_c(
    consola: ConsolaCuatroZonas,
    lector: GestorComandosInteractivos,
    config_sesion: Dict[str, Any],
    infomarket: GestorInfoMarket,
    indicadores: GestorIndicadores,
    cartera: CarteraVirtual,
    analizador_ai: AnalizadorAI,
    historial: GestorHistorial,
    terminales: GestorTerminalesSecundarias,
) -> None:
    """
    Cierre inmediato y totalmente limpio ante Ctrl+C:
    1. Detiene lector de comandos y restaura el layout VT100 + consola Win32 primero.
    2. Marca sesión inactiva y cierra terminales secundarias dependientes.
    3. Exporta resumen e historial y finaliza sin dejar la terminal ni sus pestañas congeladas.
    """
    consola.bloquear_para_cierre()
    lector.detener()
    try:
        analizador_ai.callback_alto_potencial = None
        indicadores.callback_evento = None
        infomarket.callback_noticia_relevante = None
    except Exception:
        pass
    consola.restaurar_layout_normal(limpiar_pantalla=False)

    try:
        infomarket.detener()
        indicadores.detener()
        analizador_ai.detener()
        historial.actualizar_estado_sesion({"activa": False, "cerrando": True, "modo_cierre": "Interrupción Ctrl+C"})
    except Exception:
        pass

    try:
        terminales.cerrar_todas()
    except Exception:
        pass

    try:
        cartera.cierre_inmediato_todas("Cierre inmediato por interrupción (Ctrl+C)")
        _, rutas = _construir_y_exportar_reportes_finales(
            config_sesion=config_sesion,
            nombre_cierre="Cierre inmediato por Ctrl+C",
            infomarket=infomarket,
            cartera=cartera,
            analizador_ai=analizador_ai,
            historial=historial,
        )
        print(
            f"\n{Colores.AMARILLO}Sesión finalizada inmediatamente por Ctrl+C.{Colores.RESET}\n"
            f"  • Resumen guardado en   : {Colores.VERDE}{rutas['resumen_md']}{Colores.RESET}\n"
            f"  • Posiciones guardadas  : {Colores.VERDE}{rutas['historial_csv']}{Colores.RESET}"
        )
    except Exception:
        print(f"\n{Colores.AMARILLO}Sesión finalizada inmediatamente por Ctrl+C.{Colores.RESET}")

    restaurar_consola_windows(resetear_vt=True)
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.05)
    os._exit(0)


def ejecutar_flujo_cierre_quit(
    consola: ConsolaCuatroZonas,
    lector: GestorComandosInteractivos,
    config_sesion: Dict[str, Any],
    infomarket: GestorInfoMarket,
    indicadores: GestorIndicadores,
    cartera: CarteraVirtual,
    analizador_ai: AnalizadorAI,
    historial: GestorHistorial,
    terminales: GestorTerminalesSecundarias,
) -> None:
    """
    Presenta las 3 opciones de `/quit`, liquida según el modo, muestra y guarda Resumen + Historial,
    espera 5 segundos y cierra limpiamente orquestador y terminales secundarias.
    """
    consola.bloquear_para_cierre()
    lector.pausar()
    try:
        analizador_ai.callback_alto_potencial = None
        indicadores.callback_evento = None
        infomarket.callback_noticia_relevante = None
        infomarket.detener()
        analizador_ai.detener()
    except Exception:
        pass

    # Limpia viewport tras restaurar DECSTBM (\x1b[r) para evitar cualquier sobreimpresión
    consola.restaurar_layout_normal(limpiar_pantalla=True)
    restaurar_consola_windows(resetear_vt=True)

    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except Exception:
        pass

    try:
        print(f"\n{Colores.AMARILLO}{Colores.BOLD}{'═' * 86}{Colores.RESET}")
        print(f"{Colores.AMARILLO}{Colores.BOLD}   COMANDO /quit — SELECCIONE MODALIDAD DE CIERRE DE SESIÓN Y POSICIONES{Colores.RESET}")
        print(f"{Colores.AMARILLO}{Colores.BOLD}{'═' * 86}{Colores.RESET}")
        print(
            f"  {Colores.BOLD}[1] Cierre progresivo con MyM (máximos y mínimos):{Colores.RESET}\n"
            "      Ganancias esperan evento contrario para maximizar beneficio; pérdidas esperan\n"
            "      recuperación o se liquidan si la IA determina mal panorama de mercado."
        )
        print(
            f"  {Colores.BOLD}[2] Cierre progresivo:{Colores.RESET}\n"
            "      Posiciones en ganancia se cierran de inmediato; posiciones en pérdida esperan\n"
            "      minimizar pérdidas o se cierran si la IA determina que es mejor no esperar."
        )
        print(
            f"  {Colores.BOLD}[3] Cierre inmediato:{Colores.RESET}\n"
            "      Liquida todas las posiciones abiertas inmediatamente a precio actual de mercado."
        )

        opcion = 3
        nombre_cierre = "Opción 3: Cierre inmediato"
        while True:
            sel = input(f"\n{Colores.CIAN}Seleccione opción de cierre (1, 2 o 3, Enter=3): {Colores.RESET}").strip()
            if sel in ("", "3"):
                opcion = 3
                nombre_cierre = "Opción 3: Cierre inmediato"
                break
            if sel == "1":
                opcion = 1
                nombre_cierre = "Opción 1: Cierre progresivo con MyM (máximos y mínimos)"
                break
            if sel == "2":
                opcion = 2
                nombre_cierre = "Opción 2: Cierre progresivo"
                break
            print(f"{Colores.ROJO}Opción inválida. Ingrese 1, 2 o 3.{Colores.RESET}")

        historial.actualizar_estado_sesion({"cerrando": True, "modo_cierre": nombre_cierre})
        infomarket.detener()
        analizador_ai.detener()

        if opcion == 3:
            logs_cierre = cartera.cierre_inmediato_todas("Cierre inmediato (/quit Opción 3)")
            for lg in logs_cierre:
                print(f"  {Colores.VERDE}• {lg}{Colores.RESET}")
        else:
            print(
                f"\n{Colores.CIAN}Ejecutando {nombre_cierre}... "
                f"(Presione [F] o [Ctrl+C] en cualquier momento para liquidar de inmediato){Colores.RESET}"
            )
            mapa_m = indicadores.obtener_mapa_precios_y_pe()
            en_perdida = [tk for tk, p in cartera.posiciones.items() if p.pyg_porcentual <= 0]
            pan_tk = analizador_ai.evaluar_panorama_cierre_perdidas(en_perdida)
            for lg in cartera.evaluar_paso_cierre_progresivo(opcion, mapa_m, pan_tk):
                print(f"  {Colores.VERDE}• {lg}{Colores.RESET}")

            ciclo_espera = 0
            while len(cartera.posiciones) > 0:
                if os.name == "nt" and msvcrt.kbhit():
                    tecla = msvcrt.getwch().lower()
                    if tecla == "f":
                        print(f"  {Colores.AMARILLO}Liquidación manual completada por usuario [F].{Colores.RESET}")
                        for lg in cartera.cierre_inmediato_todas(f"{nombre_cierre} (finalización manual)"):
                            print(f"  {Colores.VERDE}• {lg}{Colores.RESET}")
                        break

                time.sleep(1.0)
                ciclo_espera += 1
                if ciclo_espera % 8 == 0:
                    print(
                        f"  {Colores.GRIS}[Monitoreando {len(cartera.posiciones)} posición(es) abierta(s) en {nombre_cierre}...]{Colores.RESET}"
                    )
                    indicadores.actualizar_todos_los_tickers()
                    mapa_m = indicadores.obtener_mapa_precios_y_pe()
                    en_perdida = [tk for tk, p in cartera.posiciones.items() if p.pyg_porcentual <= 0]
                    pan_tk = analizador_ai.evaluar_panorama_cierre_perdidas(en_perdida)
                    for lg in cartera.evaluar_paso_cierre_progresivo(opcion, mapa_m, pan_tk):
                        print(f"  {Colores.VERDE}• {lg}{Colores.RESET}")

        indicadores.detener()

        resumen_final, rutas_exportadas = _construir_y_exportar_reportes_finales(
            config_sesion=config_sesion,
            nombre_cierre=nombre_cierre,
            infomarket=infomarket,
            cartera=cartera,
            analizador_ai=analizador_ai,
            historial=historial,
        )

        rend = float(resumen_final["rendimiento_total_pct"])
        col_r = Colores.VERDE_FUERTE if rend >= 0 else Colores.ROJO

        print(f"\n{Colores.CIAN}{Colores.BOLD}{'═' * 86}{Colores.RESET}")
        print(f"{Colores.CIAN}{Colores.BOLD}   RESUMEN FINAL DE LA SESIÓN — VAGY-M1{Colores.RESET}")
        print(f"{Colores.CIAN}{Colores.BOLD}{'═' * 86}{Colores.RESET}")
        print(f"  • Modalidad de Cierre : {nombre_cierre}")
        print(
            f"  • Modelo IA / Effort  : {str(config_sesion['modelo_clave']).upper()} "
            f"(effort={config_sesion.get('nivel_esfuerzo')}) | {config_sesion['etiqueta_modo']}"
        )
        print(f"  • Capital Inicial     : ${resumen_final['capital_inicial_usd']:,.2f} USD")
        print(f"  • Patrimonio Final    : ${resumen_final['patrimonio_final_usd']:,.2f} USD")
        print(f"  • Resultado Neto (PyG): {col_r}{rend:+.2f}% (${resumen_final['pyg_total_usd']:+,.2f} USD){Colores.RESET}")
        print(f"  • Posiciones Cerradas : {resumen_final['operaciones_cerradas']} (Win Rate: {resumen_final['win_rate_pct']:.1f}%)")
        print(
            f"  • Consultas a IA      : {resumen_final['ai_total_consultas']} lotes ({resumen_final['ai_activos_evaluados']} empresas) | "
            f"Alza: {resumen_final['ai_alcistas']} | Baja: {resumen_final['ai_bajistas']} | Neutrales: {resumen_final['ai_neutrales']} | "
            f"Ahorradas lote/filtro: {int(resumen_final['ai_consultas_ahorradas_lote']) + int(resumen_final['ai_consultas_filtradas'])}"
        )
        print(f"  • Noticias Procesadas : {resumen_final['total_noticias']}")
        print(f"  • Archivo Resumen     : {Colores.VERDE}{rutas_exportadas['resumen_md']}{Colores.RESET}")
        print(f"  • Archivo Posiciones  : {Colores.VERDE}{rutas_exportadas['historial_csv']}{Colores.RESET}")
        print(f"{Colores.CIAN}{'═' * 86}{Colores.RESET}")

        ancho_fin = max(40, shutil.get_terminal_size((100, 28)).columns - 2)
        for seg in range(5, 0, -1):
            msg_cnt = f"Cerrando orquestador y terminales secundarias en {seg} segundo(s)..."
            msg_cnt = msg_cnt[:ancho_fin]
            sys.stdout.write(f"\r\x1b[K{Colores.AMARILLO}{msg_cnt}{Colores.RESET}")
            sys.stdout.flush()
            time.sleep(1.0)
        sys.stdout.write("\r\x1b[K\n")
        sys.stdout.flush()

    except (KeyboardInterrupt, EOFError):
        ejecutar_apagado_inmediato_ctrl_c(
            consola=consola,
            lector=lector,
            config_sesion=config_sesion,
            infomarket=infomarket,
            indicadores=indicadores,
            cartera=cartera,
            analizador_ai=analizador_ai,
            historial=historial,
            terminales=terminales,
        )
        return

    lector.detener()
    historial.actualizar_estado_sesion({"activa": False, "cerrando": True})
    terminales.cerrar_todas()
    restaurar_consola_windows(resetear_vt=True)
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.05)
    os._exit(0)


# ==============================================================================
# 6. PUNTO DE ENTRADA PRINCIPAL DEL ORQUESTADOR
# ==============================================================================

def main() -> None:
    activar_ansi_windows()

    historial = GestorHistorial()
    info_limpieza = historial.preparar_espacio_trabajo()

    try:
        cfg = ejecutar_asistente_inicializacion(info_limpieza)
    except (KeyboardInterrupt, EOFError):
        restaurar_consola_windows(resetear_vt=True)
        print(f"\n{Colores.AMARILLO}Inicialización cancelada por el usuario.{Colores.RESET}", flush=True)
        return

    historial.actualizar_estado_sesion({
        "modelo_ai": f"{cfg['modelo_clave']} ({cfg['nivel_esfuerzo']})",
        "modo_operacion": cfg["modo_operacion"],
        "lista_acciones": cfg["lista_nombre"],
    })

    consola = ConsolaCuatroZonas()
    lector = GestorComandosInteractivos(consola)
    terminales = GestorTerminalesSecundarias(historial.sesion_dir)
    evento_interrupcion = threading.Event()

    def _limpieza_salida(*_args: Any) -> None:
        lector.detener()
        consola.restaurar_layout_normal()
        try:
            historial.actualizar_estado_sesion({"activa": False})
        except Exception:
            pass
        terminales.cerrar_todas()
        restaurar_consola_windows(resetear_vt=True)

    atexit.register(_limpieza_salida)

    def _marcar_senal_interrupcion(_signum: int, _frame: Any) -> None:
        evento_interrupcion.set()
        lector._evento_actividad.set()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig_val = getattr(signal, sig_name, None)
        if sig_val is not None:
            try:
                signal.signal(sig_val, _marcar_senal_interrupcion)
            except Exception:
                pass

    cartera = CarteraVirtual(
        capital_inicial_usd=float(cfg["capital_usd"]),
        modo_operacion=str(cfg["modo_operacion"]),
        modelo_ai_nombre=f"{str(cfg['modelo_clave']).upper()} ({cfg['nivel_esfuerzo']})",
        lista_nombre=str(cfg["lista_nombre"]),
        historial=historial,
    )

    infomarket = GestorInfoMarket(
        tickers=list(cfg["tickers"]),
        historial=historial,
    )

    indicadores = GestorIndicadores(
        tickers=list(cfg["tickers"]),
        modo_operacion=str(cfg["modo_operacion"]),
        historial=historial,
    )

    def _on_alto_potencial_ia(evento: EventoImpresionPrincipal) -> None:
        bloque = formatear_bloque_alto_potencial(evento)
        consola.imprimir_bloque_zona_1(bloque)

    analizador_ai = AnalizadorAI(
        modelo_clave=str(cfg["modelo_clave"]),
        api_key=str(cfg["api_key"]),
        modo_operacion=str(cfg["modo_operacion"]),
        infomarket=infomarket,
        indicadores=indicadores,
        cartera=cartera,
        historial=historial,
        callback_alto_potencial=_on_alto_potencial_ia,
        modo_transporte=str(cfg.get("modo_transporte", "gemini_interactions")),
        nivel_esfuerzo=str(cfg.get("nivel_esfuerzo", "medium")),
    )
    if not cfg.get("conexion_verificada") and cfg.get("mensaje_conexion"):
        analizador_ai._stats.estado_agente = (
            "SIN API KEY" if not cfg.get("api_key") else "SIN RESPUESTA: API KEY NO VÁLIDA (401)"
        )
        analizador_ai._stats.detalle_alerta = str(cfg.get("mensaje_conexion", ""))[:110]

    infomarket.callback_noticia_relevante = analizador_ai.encolar_por_noticia
    indicadores.callback_evento = analizador_ai.encolar_por_indicadores

    # Abrir las 3 terminales secundarias en sus cuadrantes sin robar foco
    print(f"{Colores.CIAN}Abriendo terminales secundarias (/v, /n, /d) en cuadrantes de pantalla...{Colores.RESET}")
    resultados_term = terminales.abrir_todas_iniciales()
    fallos_term = [msg_t for _, ok_t, msg_t in resultados_term if not ok_t]

    # Iniciar layout de 4 zonas y lector interactivo dedicado
    consola.iniciar_layout()
    lector.iniciar()
    aviso_init = (
        f"Aviso terminales: {fallos_term[0]}"
        if fallos_term
        else "Descargando noticias 48h y cotizaciones... Comandos activos: /v /n /d /help /quit."
    )
    consola.establecer_aviso(
        aviso_init,
        es_feedback_usuario=True,
    )
    consola.renderizar_pie_tres_lineas(
        stats_ai=analizador_ai.obtener_estadisticas_supervisor(),
        sup_noticias=infomarket.obtener_estado_supervision(),
        forzar=True,
    )

    def _on_ciclo_indicadores(mapa_precios: Dict[str, Dict[str, Any]]) -> None:
        cartera.actualizar_precios_mercado(mapa_precios)
        analizador_ai.registrar_ciclo_datos_mercado(mapa_precios)

    def _arranque_datos_background() -> None:
        n_hist = infomarket.descargar_contexto_inicial(dias_previos=2)
        if evento_interrupcion.is_set() or consola._cerrando_tui:
            return
        consola.establecer_aviso(
            f"Contexto listo ({n_hist} noticias 48h). Calculando indicadores de {len(cfg['tickers'])} acciones...",
            es_feedback_usuario=False,
        )
        indicadores.actualizar_todos_los_tickers()
        if evento_interrupcion.is_set() or consola._cerrando_tui:
            return
        _on_ciclo_indicadores(indicadores.obtener_mapa_precios_y_pe())
        infomarket.iniciar_monitoreo_continuo()
        indicadores.iniciar_monitoreo_continuo(callback_ciclo_precios=_on_ciclo_indicadores)
        analizador_ai.iniciar_worker()
        consola.establecer_aviso(
            "Sistema activo | /v /n /d | /pausar /reanudar IA | /modelo /lote /apikey /quit (Tab=completar)",
            es_feedback_usuario=False,
        )

    threading.Thread(target=_arranque_datos_background, name="BootstrapThread", daemon=True).start()

    def _procesar_apikey_en_segundo_plano(nueva_k: str) -> None:
        ok_k, msg_k = analizador_ai.actualizar_api_key(nueva_k)
        if ok_k:
            cfg["api_key"] = limpiar_api_key(nueva_k)
            consola.establecer_aviso(f"✓ API Key actualizada: {msg_k}")
        else:
            consola.establecer_aviso(f"✗ {msg_k}")
        lector._evento_actividad.set()

    def _procesar_modelo_en_segundo_plano(mod_in: str, esf_in: Optional[str]) -> None:
        ok_m, msg_m = analizador_ai.actualizar_modelo_o_esfuerzo(mod_in, esf_in)
        if ok_m:
            cfg["modelo_clave"] = analizador_ai.modelo_clave
            cfg["model_id"] = analizador_ai.model_id
            cfg["nivel_esfuerzo"] = analizador_ai.nivel_esfuerzo
            cartera.modelo_ai_nombre = f"{analizador_ai.modelo_clave.upper()} ({analizador_ai.nivel_esfuerzo})"
            cartera.publicar_snapshot()
            consola.establecer_aviso(f"✓ Modelo actualizado: {msg_m}")
        else:
            consola.establecer_aviso(f"✗ {msg_m}")
        lector._evento_actividad.set()

    ultimo_refresco_ui = 0.0
    ultimo_latido_disco = 0.0
    ejecutando = True

    try:
        while ejecutando:
            if evento_interrupcion.is_set() or lector.ctrl_c_pulsado():
                ejecutar_apagado_inmediato_ctrl_c(
                    consola=consola,
                    lector=lector,
                    config_sesion=cfg,
                    infomarket=infomarket,
                    indicadores=indicadores,
                    cartera=cartera,
                    analizador_ai=analizador_ai,
                    historial=historial,
                    terminales=terminales,
                )
                return

            comandos_listos = lector.extraer_comandos()
            hubo_comando = False

            for comando_recibido in comandos_listos:
                hubo_comando = True
                cmd_raw = comando_recibido.strip()
                cmd_norm = cmd_raw.lower()

                if cmd_norm == "":
                    consola.establecer_aviso(
                        "Comandos: /v /n /d | /pausar /reanudar (IA offline) | /modelo <mod> [esf] | /lote | /apikey <clave> | /quit"
                    )
                elif cmd_norm == "/quit":
                    ejecutar_flujo_cierre_quit(
                        consola=consola,
                        lector=lector,
                        config_sesion=cfg,
                        infomarket=infomarket,
                        indicadores=indicadores,
                        cartera=cartera,
                        analizador_ai=analizador_ai,
                        historial=historial,
                        terminales=terminales,
                    )
                    ejecutando = False
                    break
                elif cmd_norm in ("/v", "/car-v", "/carv", "/n", "/noticias", "/mkinfo", "/d", "/datos"):
                    _, msg_t = terminales.abrir_terminal(cmd_norm)
                    consola.establecer_aviso(msg_t)
                elif cmd_norm in ("/pausar", "/pausa", "/pause", "/offline"):
                    _, msg_p = analizador_ai.pausar_consultas()
                    consola.establecer_aviso(msg_p)
                elif cmd_norm in ("/reanudar", "/resume", "/online"):
                    _, msg_r = analizador_ai.reanudar_consultas()
                    consola.establecer_aviso(msg_r)
                elif cmd_norm == "/help":
                    consola.establecer_aviso(
                        "Comandos: /v Cartera /n Noticias /d Datos | /pausar /reanudar IA | /modelo | /lote | /apikey | /quit | Tab=Completar"
                    )
                elif cmd_norm == "/lote":
                    st = analizador_ai.obtener_estadisticas_supervisor()
                    cola_s = ", ".join(st.get("tickers_cola", [])) or "vacía"
                    ult_s = ", ".join(st.get("ultimos_tickers_consulta", [])) or "ninguno aún"
                    if st.get("consultas_pausadas"):
                        est_off = (
                            f" | OFFLINE(Not:{st.get('noticias_durante_pausa', 0)},"
                            f"Ind:{st.get('eventos_tec_durante_pausa', 0)})"
                        )
                    elif st.get("esperando_evento_reanudacion"):
                        est_off = (
                            f" | ESPERANDO ALZA/BAJA(Ctx:{st.get('noticias_durante_pausa', 0)} not,"
                            f"{st.get('eventos_tec_durante_pausa', 0)} ind)"
                        )
                    else:
                        est_off = ""
                    consola.establecer_aviso(
                        f"Lotes [{st.get('regimen_horario')}] | Último: [{ult_s}] | Cola: [{cola_s}] | Ahorradas: {st.get('ahorradas_lote')}/{st.get('filtradas_gatekeeper')}{est_off}"
                    )
                elif cmd_norm == "/apikey" or cmd_norm.startswith("/apikey "):
                    nueva_k = cmd_raw[7:].strip()
                    if not nueva_k:
                        consola.establecer_aviso("Uso: /apikey TU_CLAVE  (Ctrl+V permitido)")
                    else:
                        consola.establecer_aviso("Verificando nueva API Key en segundo plano...")
                        threading.Thread(
                            target=_procesar_apikey_en_segundo_plano,
                            args=(nueva_k,),
                            daemon=True,
                        ).start()
                elif cmd_norm == "/modelo" or cmd_norm.startswith("/modelo "):
                    partes = cmd_raw.split()
                    if len(partes) == 1:
                        consola.establecer_aviso(
                            f"Actual: {analizador_ai.modelo_clave} ({analizador_ai.model_id}, effort={analizador_ai.nivel_esfuerzo}). Uso: /modelo <nombre> [low|medium|high|xhigh|max]"
                        )
                    else:
                        mod_arg = partes[1]
                        esf_arg = partes[2] if len(partes) >= 3 else None
                        consola.establecer_aviso(f"Cambiando a modelo '{mod_arg}'...")
                        threading.Thread(
                            target=_procesar_modelo_en_segundo_plano,
                            args=(mod_arg, esf_arg),
                            daemon=True,
                        ).start()
                else:
                    consola.establecer_aviso(
                        f"Comando '{cmd_raw}' no reconocido. Use /help, /v, /n, /d, /pausar, /reanudar, /modelo, /lote, /apikey o /quit"
                    )

            if not ejecutando:
                break

            ahora = time.time()
            if ahora - ultimo_latido_disco >= 3.0:
                historial.actualizar_estado_sesion({"activa": True})
                ultimo_latido_disco = ahora

            if hubo_comando or (ahora - ultimo_refresco_ui >= 0.75):
                consola.renderizar_pie_tres_lineas(
                    stats_ai=analizador_ai.obtener_estadisticas_supervisor(),
                    sup_noticias=infomarket.obtener_estado_supervision(),
                    forzar=hubo_comando,
                )
                ultimo_refresco_ui = ahora

            lector.esperar_actividad(timeout=0.25)
    finally:
        lector.detener()
        consola.restaurar_layout_normal()
        restaurar_consola_windows(resetear_vt=True)


if __name__ == "__main__":
    main()
