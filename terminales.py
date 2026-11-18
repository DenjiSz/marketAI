# -*- coding: utf-8 -*-
"""
terminales.py
=============
Gestor de ventanas multi-terminal en Windows (distribución en 4 cuadrantes de pantalla)
y visor dedicado para las tres terminales secundarias dependientes:

1. `/v` (`/car-v`) (Cuadrante Superior Derecho) : Cartera virtual y estadísticas porcentuales en tiempo real
                                             actualizadas en la misma área visual sin parpadeo.
2. `/n` (`/mkinfo`) (Cuadrante Inferior Izquierdo): Flujo secuencial de eventos de las 6 secciones de `analisis.py`
                                             (Noticias, Filings 8-K, Reportes Resultados, Congreso, Fed y Macro):
                                             Fecha y hora (AM/PM con segs) | [ticker azul] | Sección | plataforma | link
3. `/d` (`/datos`) (Cuadrante Inferior Derecho)  : Datos financieros, reporte de resultados, patrones swing y los
                                             17 indicadores de `scanner.py` SOLO de empresas con alto potencial.

Diseño anti-bloqueo / anti-robo de foco:
- Las terminales secundarias nacen en su cuadrante usando `STARTUPINFO` (`SW_SHOWNOACTIVATE`) y
  `CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP` sin robar foco al usuario ni disparar temporizadores
  de `SetForegroundWindow`.
- Jamás se redimensiona ni manipula externamente la ventana host multi-pestaña de Windows Terminal
  (`CASCADIA_HOSTING_WINDOW_CLASS`), evitando congelamientos en el cambio de pestañas PowerShell.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.dont_write_bytecode = True

from config import (
    ETIQUETAS_SECCIONES_ANALISIS,
    Colores,
    activar_ansi_windows,
    colorear_evaluacion,
    colorear_nombre_indicador,
    desactivar_quickedit_consola,
    formatear_fecha_hora_ampm,
    restaurar_consola_windows,
    ticker_azul,
    ticker_cartera_pyg,
)


# ==============================================================================
# 1. UTILIDADES WIN32 PARA CUADRANTES DE PANTALLA Y SUPERVISIÓN DE PROCESO PADRE
# ==============================================================================

def es_windows_terminal_host() -> bool:
    """Detecta si el proceso actual se ejecuta dentro de Windows Terminal (ConPTY)."""
    return bool(os.environ.get("WT_SESSION") or os.environ.get("WT_PROFILE_ID"))


def obtener_area_trabajo_pantalla() -> Tuple[int, int, int, int]:
    """
    Obtiene el rectángulo útil del escritorio principal `(x, y, ancho, alto)`
    descontando la barra de tareas de Windows mediante `SPI_GETWORKAREA`.
    """
    if os.name != "nt":
        return (0, 0, 1920, 1040)
    try:
        rect = wintypes.RECT()
        ok = ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
        if ok:
            ancho = max(800, rect.right - rect.left)
            alto = max(600, rect.bottom - rect.top)
            return (rect.left, rect.top, ancho, alto)
        ancho = ctypes.windll.user32.GetSystemMetrics(0)
        alto = ctypes.windll.user32.GetSystemMetrics(1) - 48
        return (0, 0, max(800, ancho), max(600, alto))
    except Exception:
        return (0, 0, 1920, 1040)


def calcular_rectangulo_cuadrante(indice_cuadrante: int) -> Tuple[int, int, int, int]:
    """
    Divide la pantalla en 4 partes iguales:
    - Cuadrante 1 (0): Superior Izquierdo -> Terminal Principal (`main.py`)
    - Cuadrante 2 (1): Superior Derecho   -> Cartera Virtual (`/v` / `/car-v`)
    - Cuadrante 3 (2): Inferior Izquierdo -> Análisis 6 Secciones (`/n` / `/mkinfo`)
    - Cuadrante 4 (3): Inferior Derecho   -> Scanner Datos, Reportes e Indicadores (`/d` / `/datos`)
    """
    ox, oy, ancho_total, alto_total = obtener_area_trabajo_pantalla()
    mitad_w = ancho_total // 2
    mitad_h = alto_total // 2

    mapa = {
        0: (ox, oy, mitad_w, mitad_h),
        1: (ox + mitad_w, oy, ancho_total - mitad_w, mitad_h),
        2: (ox, oy + mitad_h, mitad_w, alto_total - mitad_h),
        3: (ox + mitad_w, oy + mitad_h, ancho_total - mitad_w, alto_total - mitad_h),
    }
    return mapa.get(indice_cuadrante, mapa[0])


def posicionar_ventana_consola_actual(
    indice_cuadrante: int,
    titulo: Optional[str] = None,
    es_principal: bool = False,
) -> None:
    """
    Ajusta título y posiciona únicamente ventanas `ConsoleWindowClass` reales y visibles
    (jamás redimensiona `CASCADIA_HOSTING_WINDOW_CLASS` de Windows Terminal para no bloquear
    el selector de pestañas PowerShell ni desincronizar dimensiones ConPTY).
    """
    if titulo:
        try:
            sys.stdout.write(f"\x1b]0;{titulo}\x07")
            sys.stdout.flush()
        except Exception:
            pass

    if os.name != "nt":
        return

    try:
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        if titulo and not es_windows_terminal_host():
            kernel32.SetConsoleTitleW(titulo)

        # Si la principal corre dentro de Windows Terminal, no alterar ni restaurar su HWND host
        if es_principal and es_windows_terminal_host():
            return

        hwnd = kernel32.GetConsoleWindow()
        if not hwnd or not user32.IsWindowVisible(hwnd):
            return

        buf_cls = ctypes.create_unicode_buffer(128)
        user32.GetClassNameW(hwnd, buf_cls, 128)
        if buf_cls.value != "ConsoleWindowClass":
            return

        x, y, w, h = calcular_rectangulo_cuadrante(indice_cuadrante)
        if es_principal:
            if user32.IsZoomed(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE solo para conhost clásico maximizado
            user32.MoveWindow(hwnd, x, y, w, h, True)
            # Esperar brevemente a que Win32 sincronice filas/columnas antes de iniciar el layout TUI
            time.sleep(0.18)
        else:
            # SW_SHOWNOACTIVATE = 4; SWP_NOACTIVATE (0x0010) | SWP_NOZORDER (0x0004)
            user32.ShowWindow(hwnd, 4)
            user32.SetWindowPos(hwnd, 0, x, y, w, h, 0x0010 | 0x0004)
    except Exception:
        pass


def pid_sigue_activo_windows(pid: int) -> bool:
    """Comprueba instantáneamente a nivel de kernel Win32 si un proceso `pid` sigue en ejecución."""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
        if not handle:
            return False
        try:
            res = kernel32.WaitForSingleObject(handle, 0)
            return res == 0x00000102  # WAIT_TIMEOUT -> sigue vivo
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def terminar_pid_windows(pid: int) -> None:
    """Finaliza limpiamente un proceso secundario por su PID en Windows."""
    if pid <= 0 or os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0001, False, int(pid))  # PROCESS_TERMINATE
        if handle:
            kernel32.TerminateProcess(handle, 0)
            kernel32.CloseHandle(handle)
    except Exception:
        pass


# ==============================================================================
# 2. GESTOR ORQUESTADOR DE LAS 3 TERMINALES SECUNDARIAS (USADO POR main.py)
# ==============================================================================

ROLES_TERMINALES = {
    "car-v": {
        "cuadrante": 1,
        "comando": "/v",
        "titulo": "VAGY-M1 | CARTERA VIRTUAL EN TIEMPO REAL (/v)",
    },
    "mkinfo": {
        "cuadrante": 2,
        "comando": "/n",
        "titulo": "VAGY-M1 | ANÁLISIS 6 SECCIONES EN TIEMPO REAL (/n)",
    },
    "datos": {
        "cuadrante": 3,
        "comando": "/d",
        "titulo": "VAGY-M1 | SCANNER DATOS, REPORTES Y 17 INDICADORES (/d)",
    },
}

MAPA_COMANDO_A_ROL: Dict[str, str] = {
    "v": "car-v",
    "car-v": "car-v",
    "carv": "car-v",
    "n": "mkinfo",
    "noticias": "mkinfo",
    "mkinfo": "mkinfo",
    "d": "datos",
    "datos": "datos",
}

_STARTF_USESHOWWINDOW = 0x00000001
_STARTF_USESIZE = 0x00000002
_STARTF_USEPOSITION = 0x00000004
_SW_SHOWNOACTIVATE = 4


def normalizar_rol_terminal(comando_o_rol: str) -> str:
    """Convierte comandos (`/v`, `/n`, `/d` o aliases) al identificador interno del visor."""
    limpio = (comando_o_rol or "").strip().lstrip("/").lower()
    return MAPA_COMANDO_A_ROL.get(limpio, limpio)


class GestorTerminalesSecundarias:
    """Administra la apertura inicial, reapertura bajo comando y cierre dependiente de las 3 terminales."""

    def __init__(self, sesion_dir: Path) -> None:
        self.sesion_dir = sesion_dir
        self.pids_dir = sesion_dir / "pids"
        self.pids_dir.mkdir(parents=True, exist_ok=True)
        self._procesos: Dict[str, subprocess.Popen] = {}
        self._script_actual = Path(__file__).resolve()

    def _leer_pid_hijo(self, rol: str) -> int:
        archivo_pid = self.pids_dir / f"{rol}.pid"
        if not archivo_pid.exists():
            return 0
        try:
            return int(archivo_pid.read_text(encoding="utf-8").strip())
        except Exception:
            return 0

    def esta_abierta(self, rol: str) -> bool:
        rol_canon = normalizar_rol_terminal(rol)
        proc = self._procesos.get(rol_canon)
        if proc is not None and proc.poll() is None:
            return True
        pid_h = self._leer_pid_hijo(rol_canon)
        return pid_sigue_activo_windows(pid_h)

    def abrir_terminal(self, rol: str) -> Tuple[bool, str]:
        """
        Abre una terminal secundaria (`/v`, `/n`, `/d` o alias previo) en su cuadrante asignado
        sin robar foco del teclado. Si ya estaba abierta, confirma su estado activo sin duplicar.
        """
        rol_canon = normalizar_rol_terminal(rol)
        if rol_canon not in ROLES_TERMINALES:
            return False, f"Rol de terminal desconocido: {rol}"

        cmd_corto = str(ROLES_TERMINALES[rol_canon]["comando"])
        if self.esta_abierta(rol_canon):
            return True, f"La terminal {cmd_corto} ya se encuentra abierta y activa."

        args_script = [
            sys.executable,
            "-B",
            "-u",
            str(self._script_actual),
            "--role",
            rol_canon,
            "--parent-pid",
            str(os.getpid()),
            "--session-dir",
            str(self.sesion_dir),
        ]

        try:
            workdir = str(self._script_actual.parent)
            env_hijo = os.environ.copy()
            env_hijo["PYTHONDONTWRITEBYTECODE"] = "1"
            if os.name == "nt":
                x, y, w, h = calcular_rectangulo_cuadrante(ROLES_TERMINALES[rol_canon]["cuadrante"])
                si = subprocess.STARTUPINFO()
                si.dwFlags = _STARTF_USESHOWWINDOW | _STARTF_USESIZE | _STARTF_USEPOSITION
                si.wShowWindow = _SW_SHOWNOACTIVATE
                si.dwX = x
                si.dwY = y
                si.dwXSize = w
                si.dwYSize = h

                flags_aislados = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
                ruta_conhost = shutil.which("conhost.exe") or r"C:\Windows\System32\conhost.exe"
                cmd = ([ruta_conhost] + args_script) if os.path.exists(ruta_conhost) else args_script
                env_hijo.pop("WT_SESSION", None)
                env_hijo.pop("WT_PROFILE_ID", None)
                proc = subprocess.Popen(
                    cmd,
                    cwd=workdir,
                    startupinfo=si,
                    creationflags=flags_aislados,
                    env=env_hijo,
                    close_fds=True,
                )
            else:
                proc = subprocess.Popen(args_script, cwd=workdir, env=env_hijo, close_fds=True)

            time.sleep(0.06)
            rc = proc.poll()
            if rc is not None and rc != 0:
                return False, f"La terminal {cmd_corto} se cerró inesperadamente (código {rc})."

            self._procesos[rol_canon] = proc
            return True, f"Terminal secundaria {cmd_corto} abierta en su cuadrante."
        except Exception as exc:
            return False, f"No se pudo abrir {cmd_corto}: {exc}"

    def abrir_todas_iniciales(self) -> list[Tuple[str, bool, str]]:
        """
        Posiciona título/consola principal (sin tocar Windows Terminal host) y abre
        las 3 terminales secundarias (`/v`, `/n`, `/d`) manteniendo el foco intacto en `main.py`.
        """
        posicionar_ventana_consola_actual(
            0,
            titulo="VAGY-M1 | ORQUESTADOR PRINCIPAL",
            es_principal=True,
        )
        resultados: list[Tuple[str, bool, str]] = []
        for rol in ("car-v", "mkinfo", "datos"):
            ok, msg = self.abrir_terminal(rol)
            resultados.append((rol, ok, msg))
            time.sleep(0.06)
        return resultados

    def cerrar_todas(self) -> None:
        """
        Cierra ordenadamente las terminales secundarias dependientes:
        1) Primero deja ~0.18s para que detecten `session_state.json -> activa: False` y salgan limpias.
        2) Termina cualquier PID o proceso host restante.
        """
        time.sleep(0.18)
        for rol in ("car-v", "mkinfo", "datos"):
            pid_hijo = self._leer_pid_hijo(rol)
            if pid_hijo > 0 and pid_sigue_activo_windows(pid_hijo):
                terminar_pid_windows(pid_hijo)
            proc = self._procesos.get(rol)
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=0.20)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


# ==============================================================================
# 3. RENDERIZADORES EN TIEMPO REAL DE CADA TERMINAL SECUNDARIA
# ==============================================================================

def _padre_y_sesion_siguen_activos(parent_pid: int, archivo_estado: Path) -> bool:
    if not pid_sigue_activo_windows(parent_pid):
        return False
    if archivo_estado.exists():
        try:
            data = json.loads(archivo_estado.read_text(encoding="utf-8"))
            if not data.get("activa", True):
                return False
        except Exception:
            pass
    return True


def ejecutar_vista_cartera(parent_pid: int, sesion_dir: Path) -> None:
    """
    Vista `/v` (`/car-v`): Cartera virtual actualizada en la MISMA área de visualización (sin scroll infinito).
    Tickers en verde si hay ganancias y en rojo si hay pérdidas (corchetes sin resaltar).
    """
    archivo_estado = sesion_dir / "session_state.json"
    archivo_cartera = sesion_dir / "cartera_state.json"

    sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
    sys.stdout.flush()

    try:
        while _padre_y_sesion_siguen_activos(parent_pid, archivo_estado):
            ancho_term = max(68, shutil.get_terminal_size((95, 28)).columns - 1)
            sep = "═" * min(ancho_term, 108)
            sep_fino = "─" * min(ancho_term, 108)

            snap: Dict = {}
            if archivo_cartera.exists():
                try:
                    snap = json.loads(archivo_cartera.read_text(encoding="utf-8"))
                except Exception:
                    snap = {}

            estado_ses: Dict = {}
            if archivo_estado.exists():
                try:
                    estado_ses = json.loads(archivo_estado.read_text(encoding="utf-8"))
                except Exception:
                    estado_ses = {}

            lineas = []
            lineas.append(f"{Colores.CIAN}{Colores.BOLD}{sep}{Colores.RESET}")
            lineas.append(
                f"{Colores.CIAN}{Colores.BOLD}  CARTERA VIRTUAL EN TIEMPO REAL (/v) — BASE PORCENTUAL{Colores.RESET}"
            )
            lineas.append(f"{Colores.CIAN}{sep}{Colores.RESET}")

            if not snap:
                lineas.append(f"  {Colores.AMARILLO}Sincronizando estado inicial de la cartera virtual...{Colores.RESET}")
            else:
                rend_pct = float(snap.get("rendimiento_total_pct", 0.0))
                pyg_tot = float(snap.get("pyg_total_usd", 0.0))
                color_rend = Colores.VERDE_FUERTE if rend_pct > 0 else (Colores.ROJO if rend_pct < 0 else Colores.AMARILLO)

                cierre_txt = snap.get("modo_cierre_activo")
                banner_cierre = (
                    f" | {Colores.AMARILLO}{Colores.BOLD}[EN CIERRE: {cierre_txt}]{Colores.RESET}"
                    if cierre_txt else ""
                )
                banner_offline = ""
                if estado_ses.get("ia_pausada"):
                    banner_offline = f" | {Colores.AMARILLO}{Colores.BOLD}[IA OFFLINE]{Colores.RESET}"
                elif estado_ses.get("ia_esperando_evento"):
                    banner_offline = f" | {Colores.CIAN}{Colores.BOLD}[ESPERANDO ALZA/BAJA]{Colores.RESET}"

                lineas.append(
                    f"  {Colores.GRIS}Actualizado:{Colores.RESET} {snap.get('timestamp', '')} | "
                    f"{Colores.GRIS}Modelo:{Colores.RESET} {Colores.MORADO}{snap.get('modelo_ai', '')}{Colores.RESET} | "
                    f"{Colores.GRIS}Modo:{Colores.RESET} {Colores.CIAN}{snap.get('modo_operacion', '')}{Colores.RESET}{banner_cierre}{banner_offline}"
                )
                lineas.append(
                    f"  {Colores.GRIS}Capital Inicial:{Colores.RESET} ${snap.get('capital_inicial_usd', 0.0):,.2f} | "
                    f"{Colores.GRIS}Patrimonio Actual:{Colores.RESET} {Colores.BOLD}${snap.get('patrimonio_actual_usd', 0.0):,.2f}{Colores.RESET} | "
                    f"{Colores.GRIS}PyG Total:{Colores.RESET} {color_rend}{rend_pct:+.2f}% (${pyg_tot:+,.2f}){Colores.RESET}"
                )
                lineas.append(
                    f"  {Colores.GRIS}Efectivo Libre:{Colores.RESET} {snap.get('efectivo_pct', 100.0):.2f}% (${snap.get('efectivo_usd', 0.0):,.2f}) | "
                    f"{Colores.GRIS}Exposición:{Colores.RESET} {snap.get('exposicion_pct', 0.0):.2f}% | "
                    f"{Colores.GRIS}Cerradas:{Colores.RESET} {snap.get('operaciones_cerradas', 0)} "
                    f"(WinRate: {snap.get('win_rate_pct', 0.0):.1f}% | Rotaciones: {snap.get('rotaciones_ejecutadas', 0)})"
                )
                lineas.append(f"{Colores.GRIS}{sep_fino}{Colores.RESET}")
                lineas.append(
                    f"  {Colores.BOLD}{'TICKER':<10} {'TIPO':<6} {'% CART':>7} {'ENTRADA':>10} "
                    f"{'ACTUAL':>10} {'P/E':>7} {'PyG %':>9} {'PyG USD':>10} {'MÁX/MÍN %':>13}{Colores.RESET}"
                )
                lineas.append(f"{Colores.GRIS}{sep_fino}{Colores.RESET}")

                posiciones = snap.get("posiciones", [])
                if not posiciones:
                    lineas.append(f"  {Colores.GRIS}(Sin posiciones abiertas actualmente — 100% en liquidez esperando señal IA){Colores.RESET}")
                else:
                    for p in posiciones[:16]:
                        tk = str(p.get("ticker", ""))
                        pyg_p = float(p.get("pyg_pct", 0.0))
                        pyg_u = float(p.get("pyg_usd", 0.0))
                        tk_col = ticker_cartera_pyg(tk, pyg_p)
                        relleno_tk = " " * max(1, 10 - (len(tk) + 2))

                        c_pyg = Colores.VERDE_FUERTE if pyg_p > 0 else (Colores.ROJO if pyg_p < 0 else Colores.AMARILLO)
                        pe_v = f"{float(p['pe']):.1f}" if p.get("pe") is not None else "N/D"
                        mym_str = f"{float(p.get('max_pyg_pct', 0.0)):+.1f}/{float(p.get('min_pyg_pct', 0.0)):+.1f}%"

                        lineas.append(
                            f"  {tk_col}{relleno_tk} {str(p.get('tipo', '')):<6} "
                            f"{float(p.get('peso_pct', 0.0)):>6.2f}% "
                            f"${float(p.get('precio_entrada', 0.0)):>9.2f} "
                            f"${float(p.get('precio_actual', 0.0)):>9.2f} "
                            f"{pe_v:>7} "
                            f"{c_pyg}{pyg_p:>+8.2f}% {pyg_u:>+9.2f}${Colores.RESET} "
                            f"{Colores.GRIS}{mym_str:>13}{Colores.RESET}"
                        )

            salida = "\x1b[H" + "\n".join(f"{ln}\x1b[K" for ln in lineas) + "\x1b[J"
            sys.stdout.write(salida)
            sys.stdout.flush()
            time.sleep(0.55)
    finally:
        sys.stdout.write("\x1b[?25h" + Colores.RESET)
        sys.stdout.flush()


def _fmt_monto_compacto(valor: object) -> str:
    """Formatea montos de ingresos en $B / $M sin dependencias externas pesadas."""
    try:
        if valor is None:
            return "N/D"
        v = float(valor)
    except Exception:
        return "N/D"
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def _etiqueta_seccion_color(seccion: str, subcategoria: str = "") -> str:
    sec_u = (seccion or "NOTICIAS").strip().upper()
    etq = ETIQUETAS_SECCIONES_ANALISIS.get(sec_u, sec_u)
    sub = (subcategoria or "").strip()
    if sec_u == "DIVULGACIONES_CONGRESO" and sub:
        etq = f"Congreso/{sub}"
    elif sec_u in ("FILINGS_8K", "REPORTES_RESULTADOS", "MACROECONOMIA") and sub:
        etq = f"{etq}/{sub[:26]}"

    mapa_col = {
        "NOTICIAS": Colores.GRIS,
        "FILINGS_8K": Colores.AMARILLO,
        "REPORTES_RESULTADOS": Colores.CIAN,
        "DIVULGACIONES_CONGRESO": Colores.MORADO,
        "FED": Colores.VERDE,
        "MACROECONOMIA": Colores.NARANJA,
    }
    col = mapa_col.get(sec_u, Colores.GRIS)
    return f"{col}{etq}{Colores.RESET}"


def ejecutar_vista_mkinfo(parent_pid: int, sesion_dir: Path) -> None:
    """Vista `/n` (`/mkinfo`): Flujo secuencial de eventos de las 6 secciones de `analisis.py`."""
    archivo_estado = sesion_dir / "session_state.json"
    archivo_noticias = sesion_dir / "mkinfo_noticias.jsonl"

    print(f"{Colores.CIAN}{Colores.BOLD}═══ ANÁLISIS MULTI-SECCIÓN — FLUJO SECUENCIAL (/n) ═══{Colores.RESET}")
    print(f"{Colores.GRIS}Secciones: Noticias | Filings 8-K | Resultados (Scanner) | Congreso | Fed | Macroeconomía{Colores.RESET}")
    print(f"{Colores.GRIS}Formato: Fecha y hora (AM/PM) | [ticker] | Sección | plataforma | link{Colores.RESET}\n")

    offset = 0
    while _padre_y_sesion_siguen_activos(parent_pid, archivo_estado):
        if archivo_noticias.exists():
            try:
                with archivo_noticias.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    nuevas_lineas = f.readlines()
                    offset = f.tell()

                for linea in nuevas_lineas:
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        n = json.loads(linea)
                        fh = n.get("fecha_hora_ampm", "")
                        tk = n.get("ticker", "MKT")
                        sec = str(n.get("seccion", "NOTICIAS"))
                        subcat = str(n.get("subcategoria", ""))
                        plat = n.get("plataforma", "Fuente")
                        link = n.get("link", "")
                        etq_sec = _etiqueta_seccion_color(sec, subcat)
                        print(f"{fh} | {ticker_azul(tk)} | {etq_sec} | {plat} | {link}", flush=True)
                    except Exception:
                        continue
            except Exception:
                pass
        time.sleep(0.35)


def ejecutar_vista_datos(parent_pid: int, sesion_dir: Path) -> None:
    """Vista `/d` (`/datos`): Datos financieros, reporte resultados, patrones y 17 indicadores de `scanner.py`."""
    archivo_estado = sesion_dir / "session_state.json"
    archivo_datos = sesion_dir / "datos_eventos.jsonl"

    print(f"{Colores.CIAN}{Colores.BOLD}═══ SCANNER CUANTITATIVO — DATOS, REPORTES Y 17 INDICADORES (/d) ═══{Colores.RESET}")
    print(f"{Colores.GRIS}Solo se listan empresas que superan el filtro cuantitativo de alto potencial y certeza.{Colores.RESET}\n")

    offset = 0
    while _padre_y_sesion_siguen_activos(parent_pid, archivo_estado):
        if archivo_datos.exists():
            try:
                with archivo_datos.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    nuevas = f.readlines()
                    offset = f.tell()

                for linea in nuevas:
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        ev = json.loads(linea)
                        tk = ev.get("ticker", "")
                        fh = ev.get("fecha_hora_ampm", formatear_fecha_hora_ampm())
                        df = ev.get("datos_financieros", {})
                        eval_gen = ev.get("evaluacion_general", "NEUTRAL")
                        certeza = float(ev.get("certeza_pct", 0.0))
                        potencial = float(ev.get("potencial_score", 0.0))
                        patrones = list(ev.get("patrones_detectados") or [])

                        pe_s = f"{df['pe_ratio']:.2f}" if df.get("pe_ratio") is not None else "N/D"
                        fpe_s = f"{df['pe_forward']:.2f}" if df.get("pe_forward") is not None else "N/D"
                        mcap_s = f"${df['cap_mercado_b']:.2f}B" if df.get("cap_mercado_b") is not None else "N/D"

                        print(f"{Colores.CIAN}{'─' * 92}{Colores.RESET}")
                        print(
                            f"{fh} | {ticker_azul(tk)} | Potencial: {Colores.BOLD}{potencial:.1f}/100{Colores.RESET} "
                            f"(Certeza: {certeza:.1f}%) | Evaluación General: {colorear_evaluacion(eval_gen)}"
                        )
                        print(
                            f"  {Colores.BOLD}Datos Financieros:{Colores.RESET} "
                            f"Precio=${df.get('precio_actual', 0.0):.2f} ({df.get('cambio_diario_pct', 0.0):+.2f}%) | "
                            f"P/E={pe_s} (Fwd={fpe_s}) | Cap={mcap_s} | VWAP=${df.get('vwap_sesion', 0.0):.2f} | "
                            f"S1/S2=${df.get('soporte_1', 0.0):.2f}/${df.get('soporte_2', 0.0):.2f} | "
                            f"R1/R2=${df.get('resistencia_1', 0.0):.2f}/${df.get('resistencia_2', 0.0):.2f}"
                        )
                        rep = df.get("reporte_resultados")
                        if isinstance(rep, dict):
                            d_h = rep.get("dias_hasta_reporte")
                            d_txt = f" ({d_h}d)" if d_h is not None else ""
                            ing_esp = _fmt_monto_compacto(rep.get("ingresos_esperados"))
                            ing_rep = _fmt_monto_compacto(rep.get("ingresos_reportados"))
                            yoy = rep.get("ingresos_crecimiento_yoy_pct")
                            yoy_txt = f" ({float(yoy):+.1f}% YoY)" if yoy is not None else ""
                            eps_rep = rep.get("eps_reportado")
                            eps_esp = rep.get("eps_esperado")
                            srp = rep.get("eps_sorpresa_pct")
                            eps_txt = (
                                f"{float(eps_rep):.2f}/{float(eps_esp):.2f} ({float(srp):+.1f}%)"
                                if (eps_rep is not None and eps_esp is not None and srp is not None)
                                else (f"{float(eps_rep):.2f}" if eps_rep is not None else "N/D")
                            )
                            print(
                                f"  {Colores.BOLD}Reporte Resultados:{Colores.RESET} "
                                f"Próx={rep.get('fecha_proximo_reporte') or 'N/D'}{d_txt} | "
                                f"Ing Esp={ing_esp} | Ing Rep={ing_rep}{yoy_txt} | "
                                f"EPS={eps_txt} | Est={Colores.CIAN}{rep.get('evaluacion_reporte', 'PENDIENTE')}{Colores.RESET}"
                            )
                        if patrones:
                            print(
                                f"  {Colores.BOLD}Patrones Swing:{Colores.RESET} "
                                f"{Colores.MORADO}{', '.join(patrones)}{Colores.RESET}"
                            )
                        print(f"  {Colores.BOLD}Indicadores Cuantitativos ({len(ev.get('indicadores', []))}):{Colores.RESET}")
                        for ind in ev.get("indicadores", []):
                            cat = ind.get("categoria", "")
                            nom = ind.get("nombre", "")
                            val_txt = ind.get("texto_valor", "")
                            ev_ind = ind.get("evaluacion", "NEUTRAL")
                            alarma = ind.get("alarma")
                            nom_col = colorear_nombre_indicador(nom, alarma)
                            ev_col = colorear_evaluacion(ev_ind)
                            print(f"    • {Colores.GRIS}{cat:<22}{Colores.RESET} | {nom_col} : {val_txt} -> {ev_col}")
                        print(
                            f"  {Colores.BOLD}Conclusión Técnica [{tk}]:{Colores.RESET} {colorear_evaluacion(eval_gen)}\n",
                            flush=True,
                        )
                    except Exception:
                        continue
            except Exception:
                pass
        time.sleep(0.4)


# ==============================================================================
# 4. PUNTO DE ENTRADA CLI PARA CADA PROCESO DE TERMINAL SECUNDARIA
# ==============================================================================

def main_secundaria() -> None:
    activar_ansi_windows()
    desactivar_quickedit_consola()
    parser = argparse.ArgumentParser(description="Visor de terminal secundaria VAGY-M1")
    parser.add_argument("--role", required=True, choices=["car-v", "mkinfo", "datos"])
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--session-dir", required=True, type=str)
    args = parser.parse_args()

    rol = args.role
    sesion_dir = Path(args.session_dir)
    pids_dir = sesion_dir / "pids"
    pids_dir.mkdir(parents=True, exist_ok=True)
    archivo_pid = pids_dir / f"{rol}.pid"
    archivo_pid.write_text(str(os.getpid()), encoding="utf-8")

    meta = ROLES_TERMINALES[rol]
    posicionar_ventana_consola_actual(meta["cuadrante"], titulo=meta["titulo"], es_principal=False)

    try:
        if rol == "car-v":
            ejecutar_vista_cartera(args.parent_pid, sesion_dir)
        elif rol == "mkinfo":
            ejecutar_vista_mkinfo(args.parent_pid, sesion_dir)
        elif rol == "datos":
            ejecutar_vista_datos(args.parent_pid, sesion_dir)
    except KeyboardInterrupt:
        pass
    finally:
        restaurar_consola_windows(resetear_vt=False)
        try:
            if archivo_pid.exists():
                archivo_pid.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main_secundaria()
