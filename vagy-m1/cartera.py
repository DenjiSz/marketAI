# -*- coding: utf-8 -*-
"""
cartera.py
==========
Gestor de Cartera Virtual de operación estrictamente porcentual (% del patrimonio).

Características principales:
- Opera sobre base porcentual independientemente del monto virtual inicial en USD.
- Soporta modo `spot` (solo posiciones LONG) y modo `short y long` (posiciones LONG y SHORT).
- Permite rotación parcial o total de posiciones abiertas cuando `analisis_ai.py` detecta
  una oportunidad con mayor potencial (exigiendo alta certeza si la posición a rotar está en pérdidas).
- Registra máximos y mínimos (`MyM`) en vivo por posición e implementa las 3 modalidades de `/quit`:
  1. Cierre progresivo con MyM (máximos y mínimos).
  2. Cierre progresivo (cierra ganancias al instante; gestiona pérdidas o cierra ante mal panorama).
  3. Cierre inmediato (liquida toda la cartera al instante).
- Publica continuamente su estado en `.sesion_activa/cartera_state.json` para la terminal `/car-v`.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from config import (
    MAX_ASIGNACION_POR_POSICION_PCT,
    UMBRAL_CERTEZA_AI_ROTACION_PERDIDA,
    formatear_fecha_hora_ampm,
)
from historial import GestorHistorial


@dataclass
class PosicionAbierta:
    """Representa una posición activa valorada porcentualmente en la cartera virtual."""

    id_posicion: str
    ticker: str
    tipo: str                     # "LONG" | "SHORT"
    fecha_apertura: str
    precio_entrada: float
    precio_actual: float
    unidades: float               # Acciones virtuales equivalentes
    asignacion_inicial_pct: float # % de la cartera asignado al abrir
    pe_accion: Optional[float]
    soporte: Optional[float]
    resistencia: Optional[float]
    max_pyg_pct: float = 0.0      # Máxima ganancia % alcanzada (para MyM)
    min_pyg_pct: float = 0.0      # Máxima caída % registrada (para MyM)
    potencial_estimado: float = 70.0
    tesis_apertura: str = ""

    @property
    def nocional_entrada_usd(self) -> float:
        return self.unidades * self.precio_entrada

    @property
    def pyg_porcentual(self) -> float:
        """Rendimiento porcentual propio de la posición según sea LONG o SHORT."""
        if self.precio_entrada <= 0:
            return 0.0
        if self.tipo == "LONG":
            return ((self.precio_actual - self.precio_entrada) / self.precio_entrada) * 100.0
        return ((self.precio_entrada - self.precio_actual) / self.precio_entrada) * 100.0

    @property
    def pyg_usd(self) -> float:
        if self.tipo == "LONG":
            return self.unidades * (self.precio_actual - self.precio_entrada)
        return self.unidades * (self.precio_entrada - self.precio_actual)

    @property
    def valor_actual_posicion_usd(self) -> float:
        """Capital colateral + ganancia/pérdida latente de la posición."""
        return max(0.0, self.nocional_entrada_usd + self.pyg_usd)


class CarteraVirtual:
    """Motor contable y de ejecución porcentual en tiempo real para VAGY-M1."""

    def __init__(
        self,
        capital_inicial_usd: float,
        modo_operacion: str,
        modelo_ai_nombre: str,
        lista_nombre: str,
        historial: GestorHistorial,
    ) -> None:
        self.capital_inicial_usd = max(1.0, float(capital_inicial_usd))
        modo_norm = modo_operacion.strip().lower()
        self.modo_operacion = "short y long" if "short" in modo_norm else "spot"
        self.modelo_ai_nombre = modelo_ai_nombre
        self.lista_nombre = lista_nombre
        self.historial = historial

        self._lock = threading.RLock()
        self.efectivo_usd: float = self.capital_inicial_usd
        self.pyg_realizado_usd: float = 0.0
        self.posiciones: Dict[str, PosicionAbierta] = {}
        self.historial_cerradas: List[Dict[str, object]] = []
        self.rotaciones_ejecutadas: int = 0
        self.modo_cierre_activo: Optional[str] = None
        self.panorama_ia_global: str = "NEUTRAL"
        self.fecha_inicio_str: str = formatear_fecha_hora_ampm()

        self.publicar_snapshot()

    # ------------------------------------------------------------------
    # 1. MÉTRICAS PATRIMONIALES EN BASE PORCENTUAL
    # ------------------------------------------------------------------

    def patrimonio_total_usd(self) -> float:
        with self._lock:
            valor_pos = sum(p.valor_actual_posicion_usd for p in self.posiciones.values())
            return max(0.01, self.efectivo_usd + valor_pos)

    def efectivo_disponible_pct(self) -> float:
        with self._lock:
            pat = self.patrimonio_total_usd()
            return round((self.efectivo_usd / pat) * 100.0, 2)

    # ------------------------------------------------------------------
    # 2. APERTURA, CIERRE PARCIAL/TOTAL Y ROTACIÓN PORCENTUAL
    # ------------------------------------------------------------------

    def abrir_o_ajustar_posicion(
        self,
        ticker: str,
        direccion: str,              # "LONG" o "SHORT"
        porcentaje_cartera: float,   # % del patrimonio total a asignar (ej. 15.0%)
        precio: float,
        pe: Optional[float] = None,
        soporte: Optional[float] = None,
        resistencia: Optional[float] = None,
        potencial_score: float = 75.0,
        motivo: str = "Señal IA",
    ) -> Tuple[bool, str]:
        """
        Ejecuta una operación calculada en porcentaje del patrimonio total.
        Respeta las reglas de `spot` vs `short y long` y liquida posiciones contrarias si existen.
        """
        tk = ticker.strip().upper()
        dir_upper = direccion.strip().upper()
        if precio <= 0:
            return False, f"Precio inválido para {tk}."

        with self._lock:
            if self.modo_cierre_activo is not None:
                return False, "Cartera en proceso de cierre (/quit); no se abren nuevas posiciones."

            # Si estamos en SPOT y llega señal SHORT, cerrar LONG abierto si existe; no abrir corto desnudo
            if self.modo_operacion == "spot" and dir_upper == "SHORT":
                if tk in self.posiciones and self.posiciones[tk].tipo == "LONG":
                    ok_c, msg_c = self.liquidar_posicion(tk, 100.0, precio, f"Venta Spot por señal bajista: {motivo}")
                    return ok_c, msg_c
                return False, f"Modo SPOT activo: se omite apertura SHORT en {tk}."

            # Si existe posición de signo contrario en el mismo ticker, cerrarla primero (flip)
            if tk in self.posiciones and self.posiciones[tk].tipo != dir_upper:
                self.liquidar_posicion(tk, 100.0, precio, f"Cambio de dirección a {dir_upper}: {motivo}")

            patrimonio = self.patrimonio_total_usd()
            pct_objetivo = min(max(2.0, float(porcentaje_cartera)), MAX_ASIGNACION_POR_POSICION_PCT)

            # Si ya tenemos posición en la misma dirección, calcular solo el tramo incremental hasta el tope
            peso_actual_pct = 0.0
            if tk in self.posiciones:
                peso_actual_pct = (self.posiciones[tk].valor_actual_posicion_usd / patrimonio) * 100.0
                if peso_actual_pct >= MAX_ASIGNACION_POR_POSICION_PCT - 1.0:
                    return False, f"{tk} ya alcanzó asignación máxima ({peso_actual_pct:.1f}%)."
                pct_objetivo = min(pct_objetivo, MAX_ASIGNACION_POR_POSICION_PCT - peso_actual_pct)

            pct_efectivo_disp = (self.efectivo_usd / patrimonio) * 100.0
            pct_ejecutable = min(pct_objetivo, pct_efectivo_disp)
            if pct_ejecutable < 1.5:
                return False, f"Efectivo porcentual insuficiente ({pct_efectivo_disp:.1f}%) para abrir {tk}."

            monto_usd = patrimonio * (pct_ejecutable / 100.0)
            unidades_nuevas = monto_usd / precio
            self.efectivo_usd = max(0.0, self.efectivo_usd - monto_usd)

            if tk in self.posiciones:
                pos = self.posiciones[tk]
                total_u = pos.unidades + unidades_nuevas
                pos.precio_entrada = ((pos.unidades * pos.precio_entrada) + (unidades_nuevas * precio)) / total_u
                pos.unidades = total_u
                pos.precio_actual = precio
                pos.asignacion_inicial_pct = round(
                    ((pos.nocional_entrada_usd) / patrimonio) * 100.0, 2
                )
                pos.potencial_estimado = max(pos.potencial_estimado, potencial_score)
                if pe is not None:
                    pos.pe_accion = pe
                accion_txt = f"Aumentada posición {dir_upper} en {tk} (+{pct_ejecutable:.1f}% cartera @ ${precio:.2f})"
            else:
                self.posiciones[tk] = PosicionAbierta(
                    id_posicion=uuid.uuid4().hex[:8].upper(),
                    ticker=tk,
                    tipo=dir_upper,
                    fecha_apertura=formatear_fecha_hora_ampm(),
                    precio_entrada=precio,
                    precio_actual=precio,
                    unidades=unidades_nuevas,
                    asignacion_inicial_pct=round(pct_ejecutable, 2),
                    pe_accion=pe,
                    soporte=soporte,
                    resistencia=resistencia,
                    max_pyg_pct=0.0,
                    min_pyg_pct=0.0,
                    potencial_estimado=potencial_score,
                    tesis_apertura=motivo[:160],
                )
                accion_txt = f"Abierta posición {dir_upper} en {tk} ({pct_ejecutable:.1f}% cartera @ ${precio:.2f})"

            self.publicar_snapshot()
            return True, accion_txt

    def liquidar_posicion(
        self,
        ticker: str,
        porcentaje_de_la_posicion: float = 100.0,
        precio_ejecucion: Optional[float] = None,
        motivo: str = "Cierre operativo",
    ) -> Tuple[bool, str]:
        """
        Liquida parcial (ej. 50%) o totalmente (100%) una posición abierta,
        acreditando el efectivo recuperado y registrando el historial detallado de PyG.
        """
        tk = ticker.strip().upper()
        with self._lock:
            pos = self.posiciones.get(tk)
            if pos is None:
                return False, f"No existe posición abierta en {tk}."

            fraccion = min(1.0, max(0.05, float(porcentaje_de_la_posicion) / 100.0))
            precio_cierre = float(precio_ejecucion) if (precio_ejecucion and precio_ejecucion > 0) else pos.precio_actual
            pos.precio_actual = precio_cierre

            patrimonio_antes = self.patrimonio_total_usd()
            unidades_cerrar = pos.unidades * fraccion
            nocional_entrada_tramo = unidades_cerrar * pos.precio_entrada

            if pos.tipo == "LONG":
                pyg_tramo_usd = unidades_cerrar * (precio_cierre - pos.precio_entrada)
                pyg_tramo_pct = ((precio_cierre - pos.precio_entrada) / pos.precio_entrada) * 100.0
            else:
                pyg_tramo_usd = unidades_cerrar * (pos.precio_entrada - precio_cierre)
                pyg_tramo_pct = ((pos.precio_entrada - precio_cierre) / pos.precio_entrada) * 100.0

            efectivo_recuperado = max(0.0, nocional_entrada_tramo + pyg_tramo_usd)
            tamano_pct_cartera = (nocional_entrada_tramo / max(patrimonio_antes, 0.01)) * 100.0

            self.efectivo_usd += efectivo_recuperado
            self.pyg_realizado_usd += pyg_tramo_usd

            self.historial_cerradas.append({
                "id_posicion": pos.id_posicion,
                "fecha_apertura": pos.fecha_apertura,
                "ticker": pos.ticker,
                "tipo": pos.tipo,
                "tamano_pct": round(tamano_pct_cartera, 2),
                "tamano_usd": round(nocional_entrada_tramo, 2),
                "precio_entrada": round(pos.precio_entrada, 4),
                "fecha_cierre": formatear_fecha_hora_ampm(),
                "precio_cierre": round(precio_cierre, 4),
                "pyg_pct": round(pyg_tramo_pct, 2),
                "pyg_usd": round(pyg_tramo_usd, 2),
                "motivo_cierre": motivo,
            })

            if fraccion >= 0.999 or (pos.unidades - unidades_cerrar) * precio_cierre < 0.50:
                del self.posiciones[tk]
                resumen = f"Cerrada 100% {pos.tipo} {tk} @ ${precio_cierre:.2f} (PyG: {pyg_tramo_pct:+.2f}%)"
            else:
                pos.unidades -= unidades_cerrar
                pos.asignacion_inicial_pct = round(pos.asignacion_inicial_pct * (1.0 - fraccion), 2)
                resumen = (
                    f"Liquidación parcial ({fraccion*100:.0f}%) {pos.tipo} {tk} @ ${precio_cierre:.2f} "
                    f"(PyG tramo: {pyg_tramo_pct:+.2f}%)"
                )

            self.publicar_snapshot()
            return True, resumen

    def ejecutar_rotacion_ia(
        self,
        liquidaciones: List[Dict[str, object]],
        certeza_nueva_oportunidad: float,
        ticker_destino: str,
    ) -> List[str]:
        """
        Liquida parcial o totalmente posiciones con menor potencial para financiar `ticker_destino`.
        Regla de protección: si una posición candidata está en pérdidas (`pyg_porcentual < 0`),
        solo permite liquidarla si `certeza_nueva_oportunidad >= UMBRAL_CERTEZA_AI_ROTACION_PERDIDA`.
        """
        mensajes: List[str] = []
        with self._lock:
            for item in liquidaciones:
                tk_liq = str(item.get("ticker", "")).strip().upper()
                pct_liq = float(item.get("porcentaje_liquidar", 0.0) or 0.0)
                justif = str(item.get("justificacion", "Rotación hacia mayor potencial"))
                if not tk_liq or tk_liq == ticker_destino or pct_liq <= 0:
                    continue

                pos = self.posiciones.get(tk_liq)
                if pos is None:
                    continue

                if pos.pyg_porcentual < -0.05 and certeza_nueva_oportunidad < UMBRAL_CERTEZA_AI_ROTACION_PERDIDA:
                    mensajes.append(
                        f"Rotación omitida en {tk_liq} (en pérdida {pos.pyg_porcentual:+.2f}% y certeza IA "
                        f"{certeza_nueva_oportunidad:.0f}% < {UMBRAL_CERTEZA_AI_ROTACION_PERDIDA:.0f}%)"
                    )
                    continue

                ok, msg = self.liquidar_posicion(
                    tk_liq,
                    porcentaje_de_la_posicion=pct_liq,
                    motivo=f"Rotación IA -> {ticker_destino} ({certeza_nueva_oportunidad:.0f}% certeza): {justif}",
                )
                if ok:
                    self.rotaciones_ejecutadas += 1
                    mensajes.append(msg)

        return mensajes

    # ------------------------------------------------------------------
    # 3. ACTUALIZACIÓN EN TIEMPO REAL DE PRECIOS Y MÁXIMOS/MÍNIMOS (MyM)
    # ------------------------------------------------------------------

    def actualizar_precios_mercado(self, mapa_mercado: Dict[str, Dict[str, Optional[float]]]) -> None:
        """Actualiza precios, P/E, S/R y picos históricos de ganancia/pérdida (`MyM`) de cada posición."""
        with self._lock:
            for tk, pos in self.posiciones.items():
                info = mapa_mercado.get(tk)
                if not info:
                    continue
                nuevo_precio = info.get("precio")
                if nuevo_precio and nuevo_precio > 0:
                    pos.precio_actual = float(nuevo_precio)
                if info.get("pe") is not None:
                    pos.pe_accion = float(info["pe"])  # type: ignore[arg-type]
                if info.get("soporte") is not None:
                    pos.soporte = float(info["soporte"])  # type: ignore[arg-type]
                if info.get("resistencia") is not None:
                    pos.resistencia = float(info["resistencia"])  # type: ignore[arg-type]

                pyg_act = pos.pyg_porcentual
                if pyg_act > pos.max_pyg_pct:
                    pos.max_pyg_pct = pyg_act
                if pyg_act < pos.min_pyg_pct:
                    pos.min_pyg_pct = pyg_act

            self.publicar_snapshot()

    # ------------------------------------------------------------------
    # 4. ALGORITMOS DE CIERRE (/quit OPCIONES 1, 2 y 3)
    # ------------------------------------------------------------------

    def cierre_inmediato_todas(self, motivo: str = "Cierre inmediato (/quit Opción 3)") -> List[str]:
        """Opción 3: Cierra inmediatamente el 100% de las posiciones abiertas."""
        logs: List[str] = []
        with self._lock:
            self.modo_cierre_activo = "Cierre inmediato"
            for tk in list(self.posiciones.keys()):
                ok, msg = self.liquidar_posicion(tk, 100.0, motivo=motivo)
                if ok:
                    logs.append(msg)
            self.publicar_snapshot()
        return logs

    def evaluar_paso_cierre_progresivo(
        self,
        modo: int,  # 1 = MyM (máximos y mínimos), 2 = Progresivo
        mapa_mercado: Dict[str, Dict[str, object]],
        panorama_ia_por_ticker: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Ejecuta un ciclo de evaluación para el cierre progresivo:
        - Modo 1 (`MyM`):
          * En ganancia (`pyg > 0`): espera un evento contrario a la posición (LONG -> señal bajista,
            SHORT -> señal alcista, o retroceso desde máximo MyM con agotamiento) para maximizar ganancias.
          * En pérdida (`pyg <= 0`): espera recuperación (`pyg >= 0`) para cerrar sin pérdidas, O liquida
            si el modelo de IA dictamina panorama `MALO`.
        - Modo 2 (`Progresivo`):
          * En ganancia (`pyg > 0`): liquida inmediatamente para asegurar beneficios.
          * En pérdida (`pyg <= 0`): espera evento favorable/rebote que minimice la pérdida (recuperación
            significativa desde su mínimo o cruce a 0%) O liquida si la IA determina panorama `MALO`.
        """
        logs: List[str] = []
        panorama_tk = panorama_ia_por_ticker or {}

        with self._lock:
            self.modo_cierre_activo = (
                "Cierre progresivo con MyM" if modo == 1 else "Cierre progresivo"
            )
            self.actualizar_precios_mercado(mapa_mercado)  # type: ignore[arg-type]

            for tk in list(self.posiciones.keys()):
                pos = self.posiciones.get(tk)
                if pos is None:
                    continue

                info = mapa_mercado.get(tk, {})
                dir_mercado = str(info.get("direccion", "NEUTRAL")).upper()
                eval_mercado = str(info.get("evaluacion", "NEUTRAL")).upper()
                pan_ia = panorama_tk.get(tk, self.panorama_ia_global).upper()
                pyg = pos.pyg_porcentual

                # Evento contrario a la posición actual
                evento_contrario = (
                    (pos.tipo == "LONG" and (dir_mercado == "BAJISTA" or "VENTA" in eval_mercado))
                    or (pos.tipo == "SHORT" and (dir_mercado == "ALCISTA" or "COMPRA" in eval_mercado))
                )
                # Evento a favor que ayuda a minimizar pérdidas
                evento_recuperacion = (
                    (pos.tipo == "LONG" and (dir_mercado == "ALCISTA" or "COMPRA" in eval_mercado))
                    or (pos.tipo == "SHORT" and (dir_mercado == "BAJISTA" or "VENTA" in eval_mercado))
                )

                if modo == 1:
                    # --- OPCIÓN 1: CIERRE PROGRESIVO CON MyM ---
                    if pyg > 0.0:
                        retroceso_mym = (pos.max_pyg_pct >= 0.8) and ((pos.max_pyg_pct - pyg) >= 0.35)
                        if evento_contrario or retroceso_mym:
                            _, msg = self.liquidar_posicion(
                                tk, 100.0,
                                motivo=f"Cierre MyM en ganancia máxima ({pyg:+.2f}%, pico {pos.max_pyg_pct:+.2f}%)",
                            )
                            logs.append(msg)
                    else:
                        if pyg >= -0.02:
                            _, msg = self.liquidar_posicion(
                                tk, 100.0, motivo="Cierre MyM tras recuperación a punto de equilibrio"
                            )
                            logs.append(msg)
                        elif pan_ia == "MALO":
                            _, msg = self.liquidar_posicion(
                                tk, 100.0, motivo="Cierre MyM preventivo: IA detecta panorama de mercado MALO"
                            )
                            logs.append(msg)

                elif modo == 2:
                    # --- OPCIÓN 2: CIERRE PROGRESIVO ---
                    if pyg > 0.0:
                        _, msg = self.liquidar_posicion(
                            tk, 100.0, motivo=f"Cierre progresivo inmediato en ganancia ({pyg:+.2f}%)"
                        )
                        logs.append(msg)
                    else:
                        mejora_desde_minimo = (pos.min_pyg_pct < -0.4) and ((pyg - pos.min_pyg_pct) >= 0.30)
                        if pyg >= -0.05 or (evento_recuperacion and mejora_desde_minimo):
                            _, msg = self.liquidar_posicion(
                                tk, 100.0, motivo=f"Cierre progresivo minimizando pérdida ({pyg:+.2f}%)"
                            )
                            logs.append(msg)
                        elif pan_ia == "MALO":
                            _, msg = self.liquidar_posicion(
                                tk, 100.0, motivo="Cierre progresivo: IA determina no esperar por panorama MALO"
                            )
                            logs.append(msg)

            self.publicar_snapshot()
        return logs

    # ------------------------------------------------------------------
    # 5. SERIALIZACIÓN DE CONTEXTO PARA IA Y PARA TERMINAL /car-v
    # ------------------------------------------------------------------

    def obtener_resumen_contexto_ia(self) -> Dict[str, object]:
        """Genera un diccionario compacto porcentual para enviar dentro del prompt a `analisis_ai.py`."""
        with self._lock:
            pat = self.patrimonio_total_usd()
            efectivo_pct = round((self.efectivo_usd / pat) * 100.0, 2)
            abiertas = []
            for tk, p in self.posiciones.items():
                peso_pct = round((p.valor_actual_posicion_usd / pat) * 100.0, 2)
                abiertas.append({
                    "ticker": tk,
                    "tipo": p.tipo,
                    "peso_cartera_pct": peso_pct,
                    "precio_entrada": round(p.precio_entrada, 2),
                    "precio_actual": round(p.precio_actual, 2),
                    "pyg_pct": round(p.pyg_porcentual, 2),
                    "potencial_previo": p.potencial_estimado,
                })
            return {
                "modo_operacion": self.modo_operacion,
                "operacion_base": "PORCENTUAL",
                "efectivo_disponible_pct": efectivo_pct,
                "exposicion_actual_pct": round(100.0 - efectivo_pct, 2),
                "rendimiento_global_cartera_pct": round(((pat - self.capital_inicial_usd) / self.capital_inicial_usd) * 100.0, 2),
                "posiciones_abiertas": abiertas,
            }

    def construir_snapshot_estado(self) -> Dict[str, object]:
        """Construye el estado completo mostrado en vivo en la terminal secundaria `/car-v`."""
        with self._lock:
            pat = self.patrimonio_total_usd()
            pyg_latente_usd = sum(p.pyg_usd for p in self.posiciones.values())
            pyg_total_usd = pat - self.capital_inicial_usd
            rend_total_pct = (pyg_total_usd / self.capital_inicial_usd) * 100.0
            efectivo_pct = (self.efectivo_usd / pat) * 100.0
            exposicion_pct = max(0.0, 100.0 - efectivo_pct)

            ganadoras = sum(1 for c in self.historial_cerradas if float(c.get("pyg_pct", 0.0)) > 0.0)
            total_cerr = len(self.historial_cerradas)
            win_rate = (ganadoras / total_cerr * 100.0) if total_cerr > 0 else 0.0

            lista_pos = []
            for tk, p in sorted(self.posiciones.items(), key=lambda kv: kv[1].valor_actual_posicion_usd, reverse=True):
                peso_actual_pct = (p.valor_actual_posicion_usd / pat) * 100.0
                lista_pos.append({
                    "ticker": tk,
                    "tipo": p.tipo,
                    "peso_pct": round(peso_actual_pct, 2),
                    "asignacion_inicial_pct": p.asignacion_inicial_pct,
                    "precio_entrada": round(p.precio_entrada, 2),
                    "precio_actual": round(p.precio_actual, 2),
                    "pe": p.pe_accion,
                    "soporte": p.soporte,
                    "resistencia": p.resistencia,
                    "pyg_pct": round(p.pyg_porcentual, 2),
                    "pyg_usd": round(p.pyg_usd, 2),
                    "max_pyg_pct": round(p.max_pyg_pct, 2),
                    "min_pyg_pct": round(p.min_pyg_pct, 2),
                    "fecha_apertura": p.fecha_apertura,
                })

            return {
                "timestamp": formatear_fecha_hora_ampm(),
                "fecha_inicio": self.fecha_inicio_str,
                "modelo_ai": self.modelo_ai_nombre,
                "modo_operacion": self.modo_operacion.upper(),
                "lista_acciones": self.lista_nombre,
                "modo_cierre_activo": self.modo_cierre_activo,
                "capital_inicial_usd": round(self.capital_inicial_usd, 2),
                "patrimonio_actual_usd": round(pat, 2),
                "efectivo_usd": round(self.efectivo_usd, 2),
                "efectivo_pct": round(efectivo_pct, 2),
                "exposicion_pct": round(exposicion_pct, 2),
                "pyg_latente_usd": round(pyg_latente_usd, 2),
                "pyg_realizado_usd": round(self.pyg_realizado_usd, 2),
                "pyg_total_usd": round(pyg_total_usd, 2),
                "rendimiento_total_pct": round(rend_total_pct, 2),
                "operaciones_cerradas": total_cerr,
                "rotaciones_ejecutadas": self.rotaciones_ejecutadas,
                "win_rate_pct": round(win_rate, 1),
                "posiciones": lista_pos,
            }

    def publicar_snapshot(self) -> None:
        """Persiste de forma atómica el estado en vivo para que `/car-v` se refresque sin parpadeos."""
        snap = self.construir_snapshot_estado()
        self.historial.publicar_estado_cartera(snap)
