# AGENTS.md — Contexto y Reglas de Arquitectura del Proyecto VAGY-M1

Este documento define la arquitectura, convenciones técnicas, contratos entre módulos y reglas de operación del sistema **VAGY-M1** (Sistema Multi-Terminal de Análisis Financiero Cuantitativo, Noticias en Tiempo Real y Gestión de Cartera Virtual asistido por IA). Está diseñado como guía de contexto directo para agentes **Gemini (Flash / Argon / Hydrogen / Carbon / Pro)** y **Claude (Sonnet / Opus / Fable)**.

---

## 1. Propósito General del Sistema

VAGY-M1 monitorea listas amplias de acciones bursátiles combinando tres flujos en paralelo con **Método de Consulta Múltiple por Lotes Sectoriales**:
1. **Noticias e información macroeconómica y sectorial (`infomarket.py`)**: Descarga contexto histórico inicial de hasta 2 días previos en el arranque y mantiene vigilancia continua en tiempo real sobre acciones específicas, Reserva Federal (Fed) / tasas de interés, inflación (CPI/PCE), conflictos geopolíticos/aranceles, reportes trimestrales (`earnings`) y noticias sectoriales interrelacionadas (Semiconductores + IA, Memoria RAM/HBM, Tecnología/Cloud, Financiero, Salud, Consumo y Defensa/Energía).
2. **Datos financieros e indicadores cuantitativos (`indicadores.py` / `datos_indicadores.py`)**: Descarga vectorizada por lotes mediante `yfinance` y cálculo en `numpy`/`pandas` de fundamentales (Precio, P/E, Soporte/Resistencia, VWAP, Cap. de Mercado) y 13 indicadores técnicos, de tendencia, flujo institucional y estructura de volumen real (`RVOL`).
3. **Motor de decisión multi-empresa con IA y reducción de consultas neutrales (`analisis_ai.py`)**: Agrupa empresas relacionadas sectorialmente (hasta `MAX_TICKERS_LOTE_AI = 6` por consulta) tras evaluar confluencia (`indicadores + noticias`), propagación sectorial (*ripple*) y sensibilidad horaria del mercado US (`America/New_York`: más permisivo en pre-market apertura `08:30-09:30 ET`, apertura `09:30-10:45 ET` y pre-cierre `14:45-16:00 ET`; estricto en sesión media y fuera de horario). Envía en **una sola consulta** una ventana de contexto específico por cada empresa evaluada + una ventana de contexto general de mercado/sectores al modelo elegido con su nivel de esfuerzo configurado (`low`, `medium`, `high`, `xhigh`, `max`).

Toda la gestión de posiciones en `cartera.py` opera de forma **estrictamente porcentual (% del capital total)**, independientemente del monto virtual inicial en USD ingresado al iniciar.

---

## 2. Catálogo de Modelos y Niveles de Esfuerzo Soportados

- **Flash 3.6** (`gemini-3.6-flash`): `low`, `medium`, `high`
- **Flash 3.7** (`gemini-3.7-flash`): `low`, `medium`, `high`
- **Flash 3.8** (`gemini-3.8-flash`, alias `flash`): `low`, `medium`, `high`
- **Argon** (`gemini-4-argon`): `low`, `medium`, `high`
- **Hydrogen** (`gemini-4-hydrogen`): `low`, `medium`, `high`, `xhigh`
- **Carbon** (`gemini-4-carbon`): `low`, `medium`, `high`
- **Pro 3.1** (`gemini-3.1-pro-preview`, alias `pro`): `low`, `high`
- **Sonnet** (`claude-sonnet-5-5`): `low`, `medium`, `high`, `xhigh`, `max`
- **Opus** (`claude-opus-5-5`): `low`, `medium`, `high`, `xhigh`, `max`
- **Fable** (`claude-fable-5-2`): `low`, `medium`, `high`, `xhigh`, `max`

---

## 3. Mapa de Archivos y Responsabilidades

### Archivos Base Requeridos
- `main.py`: Orquestador principal. Ejecuta la limpieza inicial de historial, asistente interactivo de configuración (10 modelos + niveles de esfuerzo + prueba de conexión en vivo), posicionamiento seguro de ventanas secundarias sin robo de foco, interfaz TUI de 4 zonas con `GestorComandosInteractivos` no bloqueante (edición en línea, historial `↑/↓`, cursor `←/→`, autocompletado `Tab`, pegado `Ctrl+V`), leyendas resaltadas **`Supervisión AI`** (morado, mostrando todos los tickers del lote) y **`Supervisión Noticias`** (azul), comandos interactivos (`/quit`, `/v`, `/n`, `/d` con aliases previos, `/pausar`, `/reanudar` para modo offline, `/modelo`, `/lote`, `/apikey`, `/help`) y 3 modos de cierre con limpieza de viewport para evitar sobreimpresión.
- `infomarket.py`: Motor concurrente de noticias RSS + Yahoo Finance API con deduplicación, clasificación de categoría macro/sectorial (`clasificar_categoria_y_sectores`), sentimiento (`POSITIVA`, `NEGATIVA`, `NEUTRAL`), captura temporal (`epoch_captura`, `obtener_noticias_en_rango`), contexto por lotes (`obtener_contexto_lote`) y métricas de salud/latencia en tiempo real.
- `indicadores.py`: Motor cuantitativo de datos financieros e indicadores técnicos/estructurales con evaluación (`COMPRA FUERTE`, `COMPRA`, `NEUTRAL`, `VENTA`, `VENTA FUERTE`), detección de alarmas individuales, filtro de certeza y `obtener_snapshots_todos()` para armado de lotes sectoriales.
- `analisis_ai.py`: Integración con `google-genai` (`client.interactions.create` con `thinking_level`) y API REST `v1/messages` de Anthropic (`output_config.effort`), Smart Gatekeeper de confluencia horaria, propagación sectorial *ripple*, modo pausa/offline (`/pausar` y `/reanudar` con espera obligatoria de evento alcista/bajista post-reanudación e inyección en sección D del marco de contexto de datos, indicadores y noticias generado durante la interrupción), consulta múltiple estructurada (`DecisionLoteModeloIA`) y rotación inteligente de posiciones.

### Módulos de Soporte Modular
- `datos_indicadores.py`: Puente de compatibilidad directa que reexporta la API completa de `indicadores.py`.
- `cartera.py`: Gestor de cartera virtual porcentual (`SPOT` vs `SHORT Y LONG`), valoración *mark-to-market*, seguimiento de máximos/mínimos (`MyM`) por posición y algoritmos de liquidación progresiva (`/quit` opciones 1, 2 y 3).
- `terminales.py`: Ventanas secundarias dependientes (`/v` Cartera, `/n` Noticias, `/d` Datos) lanzadas en cuadrantes mediante `STARTUPINFO` (constantes explícitas `_STARTF_USESIZE=0x2`, `_STARTF_USEPOSITION=0x4`, `_SW_SHOWNOACTIVATE=4`) sin redimensionar Windows Terminal (`CASCADIA_HOSTING_WINDOW_CLASS`) ni robar foco.
- `historial.py`: Higiene del espacio de trabajo al iniciar (mueve residuos de sesiones previas a `hystmp/` sin tocar `.py` ni `.md`), bus de estado IPC en `.sesion_activa/` y exportación final de resumen e historial separado de posiciones.
- `config.py`: Constantes globales, listas de acciones 1 y 2, mapeo de sectores interrelacionados (`MAPA_SECTORES_TICKERS`), régimen horario US ET (`obtener_regimen_horario_mercado`), catálogo de modelos/esfuerzos, códigos ANSI TrueColor y gestión segura de consola VT/Win32 sin desactivar `ENABLE_VIRTUAL_TERMINAL_INPUT` (`0x0200`).

---

## 4. Reglas Críticas de Implementación para Agentes

1. **SDK de Gemini (`google-genai`)**:
   - Usar siempre `from google import genai` y `client.interactions.create(...)` con `generation_config={"thinking_level": nivel_esfuerzo}`.
   - **Prohibido** usar el SDK legado `google-generativeai` o modelos deprecados (`gemini-1.5-*`, `gemini-2.0-*`, `gemini-2.5-*`).
   - No enviar `temperature`, `top_p`, `top_k` ni `thinking_budget` en Gemini Interactions API.
   - Para salidas JSON estructuradas en `interactions.create`, utilizar `response_format={"type": "text", "mime_type": "application/json", "schema": obtener_esquema_decision_lote_plano()}` (sin `$defs`/`$ref`).

2. **Reglas Visuales y de Terminal Windows**:
   - Fechas y horas siempre con segundos y formato AM/PM (`%Y-%m-%d %I:%M:%S %p` o `%I:%M:%S %p`).
   - Al mostrar `[ticker]`, **nunca colorear los corchetes `[` `]`**:
     - Morado únicamente en el texto del ticker para el encabezado de predicción IA en `main.py`.
     - Azul únicamente en el texto del ticker para líneas de noticias (`main.py` y `/n`).
     - Verde (ganancia) o Rojo (pérdida) únicamente en el texto del ticker dentro de `/v`.
   - En la línea antepenúltima usar la leyenda **`Supervisión AI`** con fondo morado (`Colores.BG_LEYENDA_AI`) mostrando todos los tickers evaluados por consulta; en la penúltima usar **`Supervisión Noticias`** con fondo azul (`Colores.BG_LEYENDA_NOTICIAS`).
   - En la terminal principal **no imprimir bloques en la Zona 1** si un activo resulta neutral/sin alto potencial; únicamente actualizar los contadores del Supervisor.
   - **Prohibido** invocar `MoveWindow` o `ShowWindow` sobre `CASCADIA_HOSTING_WINDOW_CLASS` (Windows Terminal) o desactivar `ENABLE_VIRTUAL_TERMINAL_INPUT` (`0x0200`); al iniciar `/quit` limpiar el viewport después de `\x1b[r` (`\x1b[r\x1b[2J\x1b[H`) y no re-emitir `\x1b[r` tras imprimir el menú o resumen final.

3. **Operación Porcentual y Rotación de Cartera**:
   - El tamaño de cada operación se calcula y registra primero como porcentaje del patrimonio (`%`), derivando el monto equivalente en USD a partir del capital virtual actualizado.
   - En modo `spot` (`sport`), solo se abren posiciones `LONG` y se cierran posiciones `LONG` existentes; en modo `short y long` se permiten tanto `LONG` como `SHORT`.
   - La rotación de posiciones con pérdidas para financiar una nueva oportunidad solo se autoriza cuando la certeza del modelo supera el umbral estricto (`UMBRAL_CERTEZA_AI_ROTACION_PERDIDA`).

4. **Manejo de Historial y Limpieza (`hystmp/`)**:
   - Al arrancar `main.py`, `historial.py` traslada archivos generados por sesiones anteriores hacia `hystmp/sesion_previa_<timestamp>/`.
   - **Jamás mover ni eliminar** archivos `.py`, archivos `.md` (`AGENTS.md`, `README.md`), configuraciones esenciales ni el propio directorio `hystmp/`.
   - Al ejecutar `/quit` (en cualquiera de sus 3 modalidades), guardar siempre dos reportes diferenciados antes de la espera de 5 segundos:
     1. Archivo de **Resumen de Sesión** (`resumen_sesion_<timestamp>.md` y `.json`).
     2. Archivo separado de **Historial de Posiciones** (`historial_posiciones_<timestamp>.csv`).
