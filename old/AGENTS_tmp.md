# AGENTS.md — Contexto y Reglas de Arquitectura del Proyecto VAGY-M1

Este documento define la arquitectura, convenciones técnicas, contratos entre módulos y reglas de operación del sistema **VAGY-M1** (Sistema Multi-Terminal de Análisis Financiero Cuantitativo, Noticias en Tiempo Real y Gestión de Cartera Virtual asistido por IA). Está diseñado como guía de contexto directo para agentes **Gemini (Carbon / Argon / Flash)** y **Claude (Opus / Sonnet / Fable)**.

---

## 1. Propósito General del Sistema

VAGY-M1 monitorea listas amplias de acciones bursátiles combinando tres flujos en paralelo:
1. **Noticias e información macroeconómica (`infomarket.py`)**: Descarga contexto histórico inicial de hasta 2 días previos en el arranque y mantiene vigilancia continua en tiempo real sobre acciones específicas, Reserva Federal (Fed), inflación (CPI/PCE), reportes trimestrales (earnings) y eventos geopolíticos/globales.
2. **Datos financieros e indicadores cuantitativos (`indicadores.py` / `datos_indicadores.py`)**: Descarga vectorizada por lotes mediante `yfinance` y cálculo en `numpy`/`pandas` de fundamentales (Precio, P/E, Soporte/Resistencia, Cap. de Mercado) y 13 indicadores técnicos, de tendencia, flujo institucional y estructura de volumen real.
3. **Motor de decisión con IA y mitigación de cuota (`analisis_ai.py`)**: Filtra eventos mediante un *Smart Gatekeeper* (evitando consultas redundantes o excesivas) y envía paquetes completos de contexto (noticias actuales + 2 días previos + macro + fundamentales + trayectoria de indicadores + estado porcentual de la cartera) al modelo elegido (`flash`, `argon`, `carbon`, `hydrogen`, `sonnet`, `opus`).

Toda la gestión de posiciones en `cartera.py` opera de forma **estrictamente porcentual (% del capital total)**, independientemente del monto virtual inicial en USD ingresado al iniciar.

---

## 2. Mapa de Archivos y Responsabilidades

### Archivos Base Requeridos
- `main.py`: Orquestador principal. Ejecuta la limpieza inicial de historial, asistente interactivo de configuración (con prueba de conexión en vivo de la API Key), posicionamiento de ventanas en 4 cuadrantes, interfaz TUI de 4 zonas en la terminal principal (1: Salida IA de alto potencial, 2: Supervisor de Agente IA, 3: Supervisión de flujo de noticias, 4: Línea interactiva de comandos `/quit`, `/mkinfo`, `/datos`, `/car-v`, `/apikey <clave>`) y coordinación de los 3 modos de cierre.
- `infomarket.py`: Motor concurrente de noticias RSS + Yahoo Finance API con deduplicación, clasificación léxica de sentimiento (`POSITIVA`, `NEGATIVA`, `NEUTRAL`), puntuación de impacto catalizador y métricas de salud/latencia en tiempo real.
- `indicadores.py`: Motor cuantitativo de datos financieros e indicadores técnicos/estructurales con evaluación (`COMPRA FUERTE`, `COMPRA`, `NEUTRAL`, `VENTA`, `VENTA FUERTE`), detección de alarmas individuales y filtro de alto potencial/certeza.
- `analisis_ai.py`: Integración con `google-genai` (Interactions API v2+/v3+ para Gemini) y API REST `v1/messages` de Anthropic (para Claude Sonnet/Opus), *Smart Gatekeeper* anti-spam, validación estructurada con Pydantic y lógica de rotación parcial/total de posiciones (incluyendo liquidación en pérdida solo bajo certeza muy alta).

### Módulos de Soporte Modular
- `datos_indicadores.py`: Puente de compatibilidad directa que reexporta la API completa de `indicadores.py`.
- `cartera.py`: Gestor de cartera virtual porcentual (`SPOT` vs `SHORT Y LONG`), valoración *mark-to-market*, seguimiento de máximos/mínimos (`MyM`) por posición y algoritmos de liquidación progresiva (`/quit` opciones 1, 2 y 3).
- `terminales.py`: Gestor multi-ventana en Windows (`conhost.exe` + Win32 `MoveWindow` en 4 cuadrantes de pantalla) y renderizador independiente de las 3 terminales secundarias (`car-v`, `mkinfo`, `datos`) con cierre automático vinculado al PID de `main.py`.
- `historial.py`: Higiene del espacio de trabajo al iniciar (mueve residuos de sesiones previas a `hystmp/` sin tocar jamás scripts `.py` ni documentos `.md`), bus de estado IPC en `.sesion_activa/` y exportación final de resumen e historial separado de posiciones.
- `config.py`: Constantes globales, listas de acciones 1 y 2, mapeo de símbolos (`BRKB -> BRK-B`), códigos de color ANSI TrueColor y formateadores exactos de texto.

---

## 3. Reglas Críticas de Implementación para Agentes

1. **SDK de Gemini (`google-genai` >= 2.30 / 3.x)**:
   - Usar siempre `from google import genai` y `client.interactions.create(...)`.
   - **Prohibido** usar el SDK legado `google-generativeai` o modelos deprecados (`gemini-1.5-*`, `gemini-2.0-*`, `gemini-2.5-*`).
   - No enviar `temperature`, `top_p` ni `top_k` a modelos Gemini 3.8 / Gemini 4 (`gemini-4-carbon`, `gemini-4-argon`, `gemini-3.8-flash`), ya que están deprecados en la API moderna.
   - Para salidas JSON estructuradas en `interactions.create`, utilizar `response_format={"type": "text", "mime_type": "application/json", "schema": EsquemaPydantic.model_json_schema()}`.

2. **Reglas Visuales y de Formato ANSI**:
   - Fechas y horas siempre con segundos y formato AM/PM (`%Y-%m-%d %I:%M:%S %p` o `%I:%M:%S %p`).
   - Al mostrar `[ticker]`, **nunca colorear los corchetes `[` `]`**:
     - Morado únicamente en el texto del ticker para el encabezado de predicción IA en `main.py`.
     - Azul únicamente en el texto del ticker para líneas de noticias (`main.py` y `/mkinfo`).
     - Verde (ganancia) o Rojo (pérdida) únicamente en el texto del ticker dentro de `/car-v`.
   - Colores de evaluación en `/datos`: `COMPRA FUERTE` (verde fuerte), `COMPRA` (verde), `NEUTRAL` (amarillo), `VENTA` (naranja TrueColor), `VENTA FUERTE` (rojo). Los nombres de indicadores que activen alarma van en verde (compra) o rojo (venta).
   - En la terminal principal **no imprimir bloques en la Zona 1** si la IA determina que no existe alto potencial (respuesta neutral/descartada); únicamente actualizar los contadores del Supervisor.

3. **Operación Porcentual y Rotación de Cartera**:
   - El tamaño de cada operación se calcula y registra primero como porcentaje del patrimonio (`%`), derivando el monto equivalente en USD a partir del capital virtual actualizado.
   - En modo `spot` (`sport`), solo se abren posiciones `LONG` y se cierran posiciones `LONG` existentes; en modo `short y long` se permiten tanto `LONG` como `SHORT`.
   - La rotación de posiciones con pérdidas para financiar una nueva oportunidad solo se autoriza cuando la certeza del modelo supera el umbral estricto (`UMBRAL_CERTEZA_AI_ROTACION_PERDIDA`).

4. **Manejo de Historial y Limpieza (`hystmp/`)**:
   - Al arrancar `main.py`, `historial.py` traslada archivos generados por sesiones anteriores (logs, estados temporales, CSV/JSON previos de salida) hacia `hystmp/sesion_anterior_<timestamp>/`.
   - **Jamás mover ni eliminar** archivos `.py`, archivos `.md` (`AGENTS.md`, `README.md`), configuraciones esenciales ni el propio directorio `hystmp/`.
   - Al ejecutar `/quit` (en cualquiera de sus 3 modalidades), guardar siempre dos reportes diferenciados antes de la espera de 5 segundos:
     1. Archivo de **Resumen de Sesión** (`resumen_sesion_<timestamp>.md` y `.json`).
     2. Archivo separado de **Historial de Posiciones** (`historial_posiciones_<timestamp>.csv`) con apertura, ticker, tipo, tamaño `%` y USD, cierre y PyG (`%` y USD).
