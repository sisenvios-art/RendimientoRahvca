# =============================================================================
# Dashboard: Rendimiento Hora Médico - H. II Huancavelica 2026
# =============================================================================
# Descripción : Evalúa el rendimiento hora médico por mes, servicio y médico
# Cálculo     : ATE / HRAS_PROG  (atenciones / horas programadas)
# Estándar    : 5 atenciones por hora
# Semáforo    : Verde >= 5.0 | Amarillo 4.9–4.99 | Rojo < 4.9
# Grupo       : MEDICO
# Subactiv.   : CONSULTA MEDICA | ATENCION ADULTO MAYOR FRAGIL
# Privacidad  : Nombres mostrados solo como iniciales (J.G.P.)
# Base datos  : Supabase (PostgreSQL) — credenciales via variables de entorno
# -----------------------------------------------------------------------------
# Variables de entorno requeridas (configurar en Streamlit Cloud > Settings > Secrets):
#   SUPABASE_URL   = https://xxxxxxxxxxxx.supabase.co
#   SUPABASE_KEY   = eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
#   SUPABASE_TABLE = nombre_de_tu_tabla
# =============================================================================

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import httpx
import resend
from supabase import create_client, Client

# ── CONFIGURACIÓN GENERAL ─────────────────────────────────────────────────────

# Orden cronológico de meses para gráficos y tablas
MESES_ORDER = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Setiembre", "Octubre", "Noviembre", "Diciembre"
]

# Filtros de negocio
GRUPO          = "MEDICO"
SUBACTIVIDADES = ["CONSULTA MEDICA", "ATENCION ADULTO MAYOR FRAGIL"]
ESTANDAR       = 5.0   # atenciones/hora — umbral óptimo
UMBRAL_RIESGO  = 4.9   # por debajo de este valor se considera "En riesgo"

# Colores del semáforo
COLOR_VERDE    = "#4CAF50"
COLOR_AMARILLO = "#FFC107"
COLOR_ROJO     = "#F44336"
COLOR_ND       = "#9E9E9E"  # sin dato / horas programadas = 0


# ── CONEXIÓN A SUPABASE ───────────────────────────────────────────────────────

def conectar_supabase() -> Client:
    """
    Crea y retorna el cliente de Supabase usando las credenciales
    almacenadas como secrets en Streamlit Cloud.
    Lanza un error claro si falta alguna variable de entorno.
    """
    try:
        url   = st.secrets["SUPABASE_URL"]
        key   = st.secrets["SUPABASE_KEY"]
    except KeyError as e:
        st.error(
            f"❌ Variable de entorno faltante: {e}\n\n"
            "Ve a Streamlit Cloud → tu app → **Settings → Secrets** y agrega:\n"
            "```\n"
            "SUPABASE_URL = 'https://xxxx.supabase.co'\n"
            "SUPABASE_KEY = 'tu-anon-key'\n"
            "SUPABASE_TABLE = 'nombre_tabla'\n"
            "```"
        )
        st.stop()

    return create_client(url, key)


# ── FUNCIONES DE PRIVACIDAD ───────────────────────────────────────────────────

def anonimizar_nombre(nombre_completo: str) -> str:
    """
    Convierte un nombre completo en iniciales para proteger datos personales.
    Ejemplo: "GARCIA PEREZ JOSE LUIS" -> "G.P.J.L."
    Retorna "S/N" si el nombre está vacío o es nulo.
    """
    if pd.isna(nombre_completo) or str(nombre_completo).strip() == "":
        return "S/N"
    partes = str(nombre_completo).strip().split()
    return ".".join(p[0].upper() for p in partes if p) + "."


# ── FUNCIONES DE SEMÁFORO ─────────────────────────────────────────────────────

def semaforo(val) -> str:
    """Clasifica un valor de rendimiento según los umbrales definidos."""
    if pd.isna(val):             return "Sin dato"
    if val >= ESTANDAR:          return "Óptimo"
    if val >= UMBRAL_RIESGO:     return "En riesgo"
    return "Bajo estándar"

def color_semaforo(val) -> str:
    """Retorna el color HEX correspondiente al nivel del semáforo."""
    return {
        "Óptimo":        COLOR_VERDE,
        "En riesgo":     COLOR_AMARILLO,
        "Bajo estándar": COLOR_ROJO,
        "Sin dato":      COLOR_ND,
    }[semaforo(val)]


# ── TENDENCIA POR MÉDICO ──────────────────────────────────────────────────────

def calcular_tendencia(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la tendencia de rendimiento por médico comparando
    el último mes disponible vs el mes anterior.
    Retorna un DataFrame con columna TENDENCIA: '↑', '↓' o '→'
    """
    # Obtener los dos últimos meses con datos
    meses_disp = grp["MES"].cat.categories.tolist()
    if len(meses_disp) < 2:
        # Si solo hay un mes no hay tendencia
        grp["TENDENCIA"] = "—"
        grp["TEND_DELTA"] = 0.0
        return grp

    mes_actual   = meses_disp[-1]
    mes_anterior = meses_disp[-2]

    # Rendimiento acumulado por médico en cada mes
    def rend_mes(mes):
        dm = grp[grp["MES"] == mes].groupby("PROFESIONAL", as_index=False).agg(
            ATE=("ATENCIONES", "sum"), HRS=("HORAS_PROG", "sum")
        )
        dm["REND"] = np.where(dm["HRS"] > 0, (dm["ATE"] / dm["HRS"]).round(2), np.nan)
        return dm.set_index("PROFESIONAL")["REND"]

    r_actual   = rend_mes(mes_actual)
    r_anterior = rend_mes(mes_anterior)

    # Unir y calcular delta
    tend = pd.DataFrame({"ACTUAL": r_actual, "ANTERIOR": r_anterior})
    tend["DELTA"] = tend["ACTUAL"] - tend["ANTERIOR"]
    tend["TENDENCIA"] = tend["DELTA"].apply(
        lambda d: "↑" if d > 0.05 else ("↓" if d < -0.05 else "→")
    )
    tend = tend.reset_index().rename(columns={"index": "PROFESIONAL", "DELTA": "TEND_DELTA"})

    # Unir tendencia al DataFrame principal
    grp = grp.merge(
        tend[["PROFESIONAL", "TENDENCIA", "TEND_DELTA"]],
        on="PROFESIONAL", how="left"
    )
    grp["TENDENCIA"]  = grp["TENDENCIA"].fillna("—")
    grp["TEND_DELTA"] = grp["TEND_DELTA"].fillna(0.0)
    return grp


# ── ALERTAS POR EMAIL ─────────────────────────────────────────────────────────

def enviar_alerta_email(df_bajo: pd.DataFrame, grp_completo: pd.DataFrame, email_destino: str) -> bool:
    """
    Envía un informe HTML por email con:
    - Resumen ejecutivo del período analizado
    - Sección por cada servicio con médicos bajo el umbral
    - Promedio anual a la fecha por cada médico afectado
    Usa Resend. Requiere RESEND_API_KEY en los secrets de Streamlit.
    """
    try:
        resend.api_key = st.secrets["RESEND_API_KEY"]

        meses_incluidos = sorted(df_bajo["MES"].unique().tolist())
        servicios_afectados = sorted(df_bajo["SERVICIO"].unique().tolist())
        total_casos = len(df_bajo)

        # Calcular promedio anual a la fecha por médico (sobre todos los meses disponibles)
        prom_anual = (
            grp_completo.groupby("PROFESIONAL", as_index=False)
            .agg(ATE_TOTAL=("ATENCIONES", "sum"), HRS_TOTAL=("HORAS_PROG", "sum"))
        )
        prom_anual["PROM_ANUAL"] = np.where(
            prom_anual["HRS_TOTAL"] > 0,
            (prom_anual["ATE_TOTAL"] / prom_anual["HRS_TOTAL"]).round(2),
            np.nan,
        )
        prom_dict = prom_anual.set_index("PROFESIONAL")["PROM_ANUAL"].to_dict()

        # Estilos base reutilizables
        st_card   = "background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08)"
        st_badge_r = "display:inline-block;background:#FF6B6B;color:white;font-weight:700;padding:3px 10px;border-radius:12px;font-size:13px"
        st_badge_a = "display:inline-block;background:#FFC107;color:#333;font-weight:700;padding:3px 10px;border-radius:12px;font-size:13px"
        st_badge_g = "display:inline-block;background:#4CAF50;color:white;font-weight:700;padding:3px 10px;border-radius:12px;font-size:13px"

        def badge_rend(val):
            if pd.isna(val): return f"<span style='{st_badge_a}'>S/D</span>"
            if val >= ESTANDAR: return f"<span style='{st_badge_g}'>{val:.2f}</span>"
            if val >= UMBRAL_RIESGO: return f"<span style='{st_badge_a}'>{val:.2f}</span>"
            return f"<span style='{st_badge_r}'>{val:.2f}</span>"

        def semaforo_texto(val):
            if pd.isna(val): return "Sin dato"
            if val >= ESTANDAR: return "✅ Óptimo"
            if val >= UMBRAL_RIESGO: return "⚠️ En riesgo"
            return "🔴 Bajo estándar"

        # ── Secciones por servicio ────────────────────────────────────────────
        secciones_html = ""
        for servicio in servicios_afectados:
            df_srv = df_bajo[df_bajo["SERVICIO"] == servicio].copy()
            meses_srv = sorted(df_srv["MES"].unique().tolist())

            # Tarjetas por mes dentro del servicio
            meses_cards = ""
            for mes in meses_srv:
                df_mes = df_srv[df_srv["MES"] == mes].copy()
                medicos_html = ""
                for _, r in df_mes.iterrows():
                    prom = prom_dict.get(r["PROFESIONAL"], np.nan)
                    tend = r.get("TENDENCIA", "—")
                    tend_color = "#4CAF50" if tend == "↑" else ("#F44336" if tend == "↓" else "#888")
                    medicos_html += f"""
                    <div style='display:flex;align-items:center;justify-content:space-between;
                                padding:10px 12px;border-bottom:1px solid #f0f0f0'>
                        <div>
                            <div style='font-weight:600;font-size:14px;color:#1F4E79'>
                                {r['PROFESIONAL']}
                                <span style='color:{tend_color};margin-left:6px;font-size:16px'>{tend}</span>
                            </div>
                            <div style='font-size:12px;color:#888;margin-top:2px'>
                                {r['SUBACTIVIDAD'].title()} &nbsp;·&nbsp;
                                Promedio anual: {badge_rend(prom)}
                            </div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:11px;color:#aaa'>Este mes</div>
                            {badge_rend(r['RENDIMIENTO'])}
                            <div style='font-size:11px;color:#aaa;margin-top:2px'>{semaforo_texto(r['RENDIMIENTO'])}</div>
                        </div>
                    </div>"""

                meses_cards += f"""
                <div style='margin-bottom:10px'>
                    <div style='background:#E3F2FD;color:#1F4E79;font-weight:600;font-size:13px;
                                padding:6px 12px;border-radius:4px 4px 0 0'>
                        📅 {mes} — {len(df_mes)} médico(s) con bajo rendimiento
                    </div>
                    <div style='border:1px solid #e0e0e0;border-top:none;border-radius:0 0 4px 4px;background:#fafafa'>
                        {medicos_html}
                    </div>
                </div>"""

            secciones_html += f"""
            <div style='{st_card}'>
                <h3 style='color:#1F4E79;margin:0 0 12px 0;font-size:16px;border-bottom:2px solid #1F4E79;padding-bottom:6px'>
                    🏥 {servicio}
                </h3>
                {meses_cards}
            </div>"""

        # ── Resumen ejecutivo ─────────────────────────────────────────────────
        resumen_html = f"""
        <div style='{st_card};background:#FFF8E1;border-left:4px solid #FFC107'>
            <h3 style='color:#E65100;margin:0 0 10px 0;font-size:15px'>📊 Resumen ejecutivo</h3>
            <table style='width:100%;font-size:13px;color:#333'>
                <tr>
                    <td style='padding:4px 0'>📅 Período analizado</td>
                    <td style='font-weight:600'>{", ".join(meses_incluidos)}</td>
                </tr>
                <tr>
                    <td style='padding:4px 0'>🏥 Servicios con alertas</td>
                    <td style='font-weight:600'>{len(servicios_afectados)} de los seleccionados</td>
                </tr>
                <tr>
                    <td style='padding:4px 0'>👨‍⚕️ Casos bajo umbral</td>
                    <td style='font-weight:600'>{total_casos} registro(s)</td>
                </tr>
                <tr>
                    <td style='padding:4px 0'>⚠️ Umbral de alerta</td>
                    <td style='font-weight:600'>&lt; {UMBRAL_RIESGO} atenciones/hora</td>
                </tr>
                <tr>
                    <td style='padding:4px 0'>🎯 Estándar óptimo</td>
                    <td style='font-weight:600'>≥ {ESTANDAR} atenciones/hora</td>
                </tr>
            </table>
        </div>"""

        # ── HTML completo ─────────────────────────────────────────────────────
        html = f"""
        <html>
        <body style='margin:0;padding:0;background:#F5F7FA;font-family:Arial,sans-serif'>
        <div style='max-width:640px;margin:0 auto;padding:24px 16px'>

            <!-- Encabezado -->
            <div style='background:#1F4E79;border-radius:10px 10px 0 0;padding:24px;text-align:center;margin-bottom:0'>
                <h1 style='color:white;margin:0;font-size:20px'>⚠️ Alerta de Rendimiento Hora Médico</h1>
                <p style='color:#B3D4F5;margin:6px 0 0 0;font-size:13px'>
                    H. II Huancavelica &nbsp;·&nbsp; Grupo Ocupacional: MÉDICO
                </p>
            </div>

            <!-- Banda de alerta -->
            <div style='background:#FF6B6B;padding:10px;text-align:center;margin-bottom:20px'>
                <span style='color:white;font-weight:700;font-size:14px'>
                    {total_casos} médico(s) con rendimiento bajo {UMBRAL_RIESGO} atenciones/hora
                </span>
            </div>

            {resumen_html}

            <!-- Secciones por servicio -->
            <h2 style='color:#1F4E79;font-size:15px;margin:20px 0 12px 0'>
                📋 Detalle por servicio y período
            </h2>
            {secciones_html}

            <!-- Pie -->
            <div style='text-align:center;color:#aaa;font-size:11px;margin-top:24px;padding-top:16px;
                        border-top:1px solid #e0e0e0'>
                Cálculo: ATE ÷ HRAS_PROG &nbsp;·&nbsp;
                Nombres mostrados como iniciales (protección de datos) &nbsp;·&nbsp;
                Generado automáticamente por el Dashboard de Rendimiento
            </div>

        </div>
        </body></html>"""

        resend.Emails.send({
            "from":    "alertas@resend.dev",
            "to":      [email_destino],
            "subject": f"⚠️ Alerta rendimiento médico — {total_casos} caso(s) en {len(meses_incluidos)} mes(es)",
            "html":    html,
        })
        return True

    except Exception as e:
        st.error(f"Error al enviar email: {e}")
        return False


# ── CARGA Y PROCESAMIENTO DE DATOS ────────────────────────────────────────────

@st.cache_data(ttl=600)  # Cachea 10 minutos para no sobrecargar Supabase
def cargar_datos() -> pd.DataFrame:
    """
    Consulta la tabla de Supabase via API REST directa (httpx),
    filtra en Python por grupo ocupacional y subactividades,
    calcula el rendimiento (ATE / HRAS_PROG) y anonimiza nombres.
    """
    url   = st.secrets["SUPABASE_URL"]
    key   = st.secrets["SUPABASE_KEY"]
    tabla = st.secrets.get("SUPABASE_TABLE", "hras_efectivas")

    # Headers requeridos por la API REST de Supabase (PostgREST)
    headers = {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
    }

    # Columnas a traer — sin MES (no existe), se deriva de PERIODO
    columnas = "PERIODO,SERVICIO,PROFESIONAL,SUBACTIVIDAD,ATE,HRAS_PROG,GRPO_OCUPACIONAL"

    # Paginación con offset/limit via API REST de PostgREST
    todos  = []
    offset = 0
    batch  = 1000

    while True:
        # PostgREST requiere el filtro como parámetro separado con el nombre exacto de columna
        respuesta = httpx.get(
            f"{url}/rest/v1/{tabla}",
            headers={
                **headers,
                "Range-Unit": "items",
                "Range": f"{offset}-{offset + batch - 1}",
            },
            params={
                "select": columnas,
                "GRPO_OCUPACIONAL": f"eq.{GRUPO}",
                "limit":  str(batch),
                "offset": str(offset),
            },
            timeout=30,
        )
        respuesta.raise_for_status()
        filas = respuesta.json()
        # Si la respuesta no es lista (error de PostgREST), salir
        if not isinstance(filas, list):
            break

        if not filas:
            break
        todos.extend(filas)
        if len(filas) < batch:
            break
        offset += batch

    if not todos:
        return pd.DataFrame()

    datos = pd.DataFrame(todos)

    # Filtrar subactividades en Python
    datos = datos[datos["SUBACTIVIDAD"].isin(SUBACTIVIDADES)].copy()

    if datos.empty:
        return pd.DataFrame()

    # Derivar columna MES desde PERIODO (formato dd/mm/yyyy o yyyy-mm-dd)
    MAPA_MESES = {
        1: "Enero",     2: "Febrero",    3: "Marzo",
        4: "Abril",     5: "Mayo",       6: "Junio",
        7: "Julio",     8: "Agosto",     9: "Setiembre",
        10: "Octubre",  11: "Noviembre", 12: "Diciembre"
    }
    fechas       = pd.to_datetime(datos["PERIODO"], dayfirst=True, errors="coerce")
    datos["MES"] = fechas.dt.month.map(MAPA_MESES)

    # Convertir columnas numéricas
    datos["ATE"]       = pd.to_numeric(datos["ATE"],       errors="coerce").fillna(0)
    datos["HRAS_PROG"] = pd.to_numeric(datos["HRAS_PROG"], errors="coerce").fillna(0)

    # Anonimizar nombres antes de agrupar
    datos["PROFESIONAL"] = datos["PROFESIONAL"].apply(anonimizar_nombre)

    # Agrupar por mes, servicio, médico y subactividad
    grp = datos.groupby(
        ["MES", "SERVICIO", "PROFESIONAL", "SUBACTIVIDAD"], as_index=False
    ).agg(
        ATENCIONES=("ATE",       "sum"),
        HORAS_PROG=("HRAS_PROG", "sum"),
    )

    # Orden cronológico de meses
    meses_presentes = [m for m in MESES_ORDER if m in grp["MES"].values]
    grp["MES"] = pd.Categorical(grp["MES"], categories=meses_presentes, ordered=True)
    grp.sort_values(["MES", "SERVICIO", "PROFESIONAL"], inplace=True)

    # Calcular rendimiento — NaN si HRAS_PROG = 0
    grp["RENDIMIENTO"] = np.where(
        grp["HORAS_PROG"] > 0,
        (grp["ATENCIONES"] / grp["HORAS_PROG"]).round(2),
        np.nan,
    )

    # Columnas auxiliares para gráficos
    grp["SEMAFORO"] = grp["RENDIMIENTO"].apply(semaforo)
    grp["COLOR"]    = grp["RENDIMIENTO"].apply(color_semaforo)

    return grp


def exportar_excel(df: pd.DataFrame) -> bytes:
    """
    Genera un Excel en memoria con semáforo visual en la columna RENDIMIENTO.
    Listo para descarga desde el dashboard sin guardar archivos en disco.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.drop(columns=["COLOR"], errors="ignore").to_excel(
            writer, index=False, sheet_name="Rendimiento"
        )
        wb  = writer.book
        ws  = writer.sheets["Rendimiento"]

        # Formatos de celda para cada nivel del semáforo
        fmt_v = wb.add_format({"bg_color": "#70AD47", "bold": True,
                                "num_format": "0.00", "border": 1})
        fmt_a = wb.add_format({"bg_color": "#FFD966", "bold": True,
                                "num_format": "0.00", "border": 1})
        fmt_r = wb.add_format({"bg_color": "#FF6B6B", "bold": True,
                                "num_format": "0.00", "border": 1,
                                "font_color": "white"})

        col_rend = df.columns.get_loc("RENDIMIENTO")
        for row_num, val in enumerate(df["RENDIMIENTO"], start=1):
            if pd.isna(val):
                continue
            fmt = fmt_v if val >= ESTANDAR else (fmt_a if val >= UMBRAL_RIESGO else fmt_r)
            ws.write(row_num, col_rend, val, fmt)

    return buf.getvalue()


# ── CONFIGURACIÓN DE PÁGINA ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Rendimiento Hora Médico",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Rendimiento Hora Médico")
st.caption(
    "H. II Huancavelica 2026 · Grupo Ocupacional: MÉDICO · "
    "Estándar: 5 atenc/hora · Cálculo: ATE ÷ HRAS_PROG · "
    "🔒 Nombres mostrados como iniciales · 🗄️ Datos desde Supabase"
)

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────

# Botón para forzar recarga desde Supabase (limpia el caché)
with st.sidebar:
    if st.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.rerun()

grp = cargar_datos()

if grp.empty:
    st.error(
        "No se encontraron datos en Supabase con los filtros aplicados. "
        "Verifica que la tabla tenga registros del grupo MEDICO y las subactividades configuradas."
    )
    st.stop()

# Calcular tendencia comparando último mes vs mes anterior
grp = calcular_tendencia(grp)


# ── FILTROS (PANEL LATERAL) ───────────────────────────────────────────────────

with st.sidebar:
    st.header("🔍 Filtros")

    # Solo muestra meses que tienen datos reales en la tabla
    meses_disp = [m for m in MESES_ORDER if m in grp["MES"].values]
    sel_mes = st.multiselect("Mes", meses_disp, default=meses_disp)

    servicios_disp = sorted(grp["SERVICIO"].unique())
    sel_srv = st.multiselect("Servicio", servicios_disp, default=servicios_disp)

    medicos_disp = sorted(grp["PROFESIONAL"].unique())
    sel_med = st.multiselect("Médico (iniciales)", medicos_disp, default=medicos_disp)

    subs_disp = sorted(grp["SUBACTIVIDAD"].unique())
    sel_sub = st.multiselect("Subactividad", subs_disp, default=subs_disp)

    sel_sem = st.multiselect(
        "Semáforo",
        ["Óptimo", "En riesgo", "Bajo estándar", "Sin dato"],
        default=["Óptimo", "En riesgo", "Bajo estándar", "Sin dato"],
    )

    st.markdown("---")
    st.markdown(
        "**Semáforo de rendimiento**\n"
        f"- 🟢 **Óptimo** ≥ {ESTANDAR}\n"
        f"- 🟡 **En riesgo** {UMBRAL_RIESGO} – {ESTANDAR - 0.01}\n"
        f"- 🔴 **Bajo estándar** < {UMBRAL_RIESGO}\n"
        "- ⚪ **Sin dato** HRAS_PROG = 0"
    )
    st.markdown("---")
    st.caption("🔒 Nombres mostrados como iniciales para proteger datos personales.")

    st.markdown("---")
    st.header("📧 Alertas por email")

    # Filtros específicos para el envío de alertas
    alerta_meses = st.multiselect(
        "Meses a incluir en alerta",
        options=meses_disp,
        default=meses_disp,
        key="alerta_mes",
    )
    alerta_servicios = st.multiselect(
        "Servicios a incluir en alerta",
        options=servicios_disp,
        default=servicios_disp,
        key="alerta_srv",
    )
    email_destino = st.text_input(
        "Email destino",
        placeholder="correo@ejemplo.com",
        help="Recibirá la lista de médicos bajo el umbral de rendimiento",
    )

    if st.button("🚨 Enviar alerta de bajo rendimiento"):
        if not email_destino:
            st.warning("Ingresa un email destino.")
        elif "RESEND_API_KEY" not in st.secrets:
            st.error("Agrega RESEND_API_KEY en los Secrets de Streamlit Cloud.")
        else:
            # Filtrar por meses y servicios seleccionados en la sección de alertas
            df_bajo = grp[
                grp["MES"].isin(alerta_meses) &
                grp["SERVICIO"].isin(alerta_servicios) &
                (grp["RENDIMIENTO"] < UMBRAL_RIESGO)
            ].copy()
            if df_bajo.empty:
                st.success("✅ Ningún médico está bajo el umbral en el período y servicios seleccionados.")
            else:
                with st.spinner("Enviando email..."):
                    ok = enviar_alerta_email(df_bajo, grp, email_destino)
                if ok:
                    st.success(f"✅ Alerta enviada a {email_destino} — {len(df_bajo)} caso(s) en {len(alerta_meses)} mes(es) y {len(alerta_servicios)} servicio(s).")
                else:
                    st.error("No se pudo enviar el email. Revisa la API key de Resend.")

# Aplicar filtros seleccionados
df = grp[
    grp["MES"].isin(sel_mes) &
    grp["SERVICIO"].isin(sel_srv) &
    grp["PROFESIONAL"].isin(sel_med) &
    grp["SUBACTIVIDAD"].isin(sel_sub) &
    grp["SEMAFORO"].isin(sel_sem)
].copy()

if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()


# ── KPIs PRINCIPALES ──────────────────────────────────────────────────────────

vals       = df["RENDIMIENTO"].dropna()
n_verde    = (vals >= ESTANDAR).sum()
n_amarillo = ((vals >= UMBRAL_RIESGO) & (vals < ESTANDAR)).sum()
n_rojo     = (vals < UMBRAL_RIESGO).sum()
pct_verde  = n_verde / len(vals) * 100 if len(vals) else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📋 Registros",     len(df))
k2.metric("📊 Promedio",      f"{vals.mean():.2f}" if len(vals) else "—")
k3.metric("🟢 Óptimo",        f"{n_verde}  ({pct_verde:.0f}%)")
k4.metric("🟡 En riesgo",     str(n_amarillo))
k5.metric("🔴 Bajo estándar", str(n_rojo))

st.markdown("---")


# ── TABS DE CONTENIDO ─────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Evolución mensual",
    "🏢 Por servicio",
    "👨‍⚕️ Por médico",
    "📋 Tabla detalle",
])


# ── TAB 1: Evolución mensual ───────────────────────────────────────────────────
with tab1:

    evo = (
        df.groupby("MES", as_index=False, observed=True)
          .agg(ATENCIONES=("ATENCIONES", "sum"), HORAS_PROG=("HORAS_PROG", "sum"))
    )
    evo["RENDIMIENTO"] = np.where(
        evo["HORAS_PROG"] > 0,
        (evo["ATENCIONES"] / evo["HORAS_PROG"]).round(2), np.nan
    )
    evo["COLOR"] = evo["RENDIMIENTO"].apply(color_semaforo)

    fig = go.Figure()
    fig.add_hline(y=ESTANDAR,      line_dash="dash", line_color=COLOR_VERDE,
                  annotation_text=f"Estándar ({ESTANDAR})")
    fig.add_hline(y=UMBRAL_RIESGO, line_dash="dot",  line_color=COLOR_AMARILLO,
                  annotation_text=f"Mínimo aceptable ({UMBRAL_RIESGO})")
    fig.add_trace(go.Scatter(
        x=evo["MES"].astype(str), y=evo["RENDIMIENTO"],
        mode="lines+markers+text",
        marker=dict(color=evo["COLOR"], size=12, line=dict(width=1.5, color="white")),
        line=dict(color="#607D8B", width=2),
        text=evo["RENDIMIENTO"].apply(lambda v: f"{v:.2f}" if not pd.isna(v) else ""),
        textposition="top center",
        name="Rendimiento promedio",
    ))
    fig.update_layout(
        title="Rendimiento promedio mensual (todos los servicios seleccionados)",
        xaxis_title="Mes", yaxis_title="Atenciones / Hora programada",
        plot_bgcolor="white", height=420,
        yaxis=dict(gridcolor="#ECEFF1"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparativa por subactividad
    evo_sub = (
        df.groupby(["MES", "SUBACTIVIDAD"], as_index=False, observed=True)
          .agg(ATENCIONES=("ATENCIONES", "sum"), HORAS_PROG=("HORAS_PROG", "sum"))
    )
    evo_sub["RENDIMIENTO"] = np.where(
        evo_sub["HORAS_PROG"] > 0,
        (evo_sub["ATENCIONES"] / evo_sub["HORAS_PROG"]).round(2), np.nan
    )
    fig2 = px.line(
        evo_sub, x="MES", y="RENDIMIENTO", color="SUBACTIVIDAD",
        markers=True, title="Evolución mensual por subactividad",
        labels={"RENDIMIENTO": "Atenciones/hora", "MES": "Mes"},
    )
    fig2.add_hline(y=ESTANDAR, line_dash="dash", line_color=COLOR_VERDE,
                   annotation_text="Estándar")
    fig2.update_layout(plot_bgcolor="white", height=380, yaxis=dict(gridcolor="#ECEFF1"))
    st.plotly_chart(fig2, use_container_width=True)


# ── TAB 2: Por servicio ────────────────────────────────────────────────────────
with tab2:

    srv_acum = (
        df.groupby("SERVICIO", as_index=False)
          .agg(ATENCIONES=("ATENCIONES", "sum"), HORAS_PROG=("HORAS_PROG", "sum"))
    )
    srv_acum["RENDIMIENTO"] = np.where(
        srv_acum["HORAS_PROG"] > 0,
        (srv_acum["ATENCIONES"] / srv_acum["HORAS_PROG"]).round(2), np.nan
    )
    srv_acum["COLOR"]    = srv_acum["RENDIMIENTO"].apply(color_semaforo)
    srv_acum["SEMAFORO"] = srv_acum["RENDIMIENTO"].apply(semaforo)
    srv_acum.sort_values("RENDIMIENTO", ascending=True, inplace=True)

    # Barras horizontales ordenadas de menor a mayor
    fig3 = go.Figure(go.Bar(
        x=srv_acum["RENDIMIENTO"], y=srv_acum["SERVICIO"],
        orientation="h", marker_color=srv_acum["COLOR"],
        text=srv_acum["RENDIMIENTO"].apply(lambda v: f"{v:.2f}" if not pd.isna(v) else ""),
        textposition="outside",
    ))
    fig3.add_vline(x=ESTANDAR, line_dash="dash", line_color=COLOR_VERDE,
                   annotation_text="Estándar")
    fig3.update_layout(
        title="Rendimiento acumulado por servicio (período seleccionado)",
        xaxis_title="Atenciones / hora programada",
        plot_bgcolor="white", height=max(400, len(srv_acum) * 30),
        xaxis=dict(gridcolor="#ECEFF1"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap: servicio x mes
    srv_mes = (
        df.groupby(["SERVICIO", "MES"], as_index=False, observed=True)
          .agg(ATENCIONES=("ATENCIONES", "sum"), HORAS_PROG=("HORAS_PROG", "sum"))
    )
    srv_mes["RENDIMIENTO"] = np.where(
        srv_mes["HORAS_PROG"] > 0,
        (srv_mes["ATENCIONES"] / srv_mes["HORAS_PROG"]).round(2), np.nan
    )
    pivot = srv_mes.pivot_table(index="SERVICIO", columns="MES", values="RENDIMIENTO")
    pivot = pivot.reindex(columns=[m for m in MESES_ORDER if m in pivot.columns])

    fig4 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  COLOR_ROJO],    [0.45, COLOR_ROJO],
            [0.45, COLOR_AMARILLO],[0.50, COLOR_AMARILLO],
            [0.50, COLOR_VERDE],   [1.0,  COLOR_VERDE],
        ],
        zmin=0, zmax=10,
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="Servicio: %{y}<br>Mes: %{x}<br>Rendimiento: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Rend."),
    ))
    fig4.update_layout(
        title="Mapa de calor: rendimiento por servicio y mes",
        height=max(400, len(pivot) * 28),
    )
    st.plotly_chart(fig4, use_container_width=True)


# ── TAB 3: Por médico ──────────────────────────────────────────────────────────
with tab3:

    st.info("🔒 Los nombres se muestran como iniciales para proteger los datos personales.")

    med_df = (
        df.groupby(["PROFESIONAL", "SERVICIO"], as_index=False)
          .agg(ATENCIONES=("ATENCIONES", "sum"), HORAS_PROG=("HORAS_PROG", "sum"))
    )
    med_df["RENDIMIENTO"] = np.where(
        med_df["HORAS_PROG"] > 0,
        (med_df["ATENCIONES"] / med_df["HORAS_PROG"]).round(2), np.nan
    )
    # Unir tendencia al DataFrame de médicos
    tend_med = grp[["PROFESIONAL","TENDENCIA","TEND_DELTA"]].drop_duplicates("PROFESIONAL")
    med_df = med_df.merge(tend_med, on="PROFESIONAL", how="left")
    med_df["TENDENCIA"]  = med_df["TENDENCIA"].fillna("—")
    med_df["TEND_DELTA"] = med_df["TEND_DELTA"].fillna(0.0)
    # Etiqueta combinada: rendimiento + tendencia
    med_df["LABEL"] = med_df.apply(
        lambda r: f"{r['RENDIMIENTO']:.2f} {r['TENDENCIA']}" if not pd.isna(r["RENDIMIENTO"]) else "S/D", axis=1
    )
    med_df["COLOR"]    = med_df["RENDIMIENTO"].apply(color_semaforo)
    med_df["SEMAFORO"] = med_df["RENDIMIENTO"].apply(semaforo)
    med_df.sort_values("RENDIMIENTO", ascending=True, inplace=True)

    fig5 = px.bar(
        med_df, x="RENDIMIENTO", y="PROFESIONAL",
        color="SEMAFORO", orientation="h",
        color_discrete_map={
            "Óptimo":        COLOR_VERDE,
            "En riesgo":     COLOR_AMARILLO,
            "Bajo estándar": COLOR_ROJO,
            "Sin dato":      COLOR_ND,
        },
        hover_data={"SERVICIO": True, "ATENCIONES": True, "HORAS_PROG": True,
                    "TENDENCIA": True, "TEND_DELTA": True},
        text="LABEL",
        title="Rendimiento acumulado por médico — con tendencia (↑ sube · ↓ baja · → estable)",
        labels={"RENDIMIENTO": "Atenciones/hora", "PROFESIONAL": "Médico"},
    )
    fig5.add_vline(x=ESTANDAR, line_dash="dash", line_color="#333",
                   annotation_text="Estándar")
    fig5.update_layout(
        plot_bgcolor="white", height=max(500, len(med_df) * 22),
        xaxis=dict(gridcolor="#ECEFF1"),
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Distribución por nivel de semáforo
    sem_counts = med_df["SEMAFORO"].value_counts().reset_index()
    sem_counts.columns = ["Semáforo", "Médicos"]
    fig6 = px.pie(
        sem_counts, names="Semáforo", values="Médicos",
        color="Semáforo",
        color_discrete_map={
            "Óptimo":        COLOR_VERDE,
            "En riesgo":     COLOR_AMARILLO,
            "Bajo estándar": COLOR_ROJO,
            "Sin dato":      COLOR_ND,
        },
        title="Distribución de médicos por nivel de rendimiento",
        hole=0.4,
    )
    st.plotly_chart(fig6, use_container_width=True)


# ── TAB 4: Tabla detalle ───────────────────────────────────────────────────────
with tab4:

    st.info("🔒 La columna 'Médico' muestra solo las iniciales del profesional.")

    tabla = df[[
        "MES", "SERVICIO", "PROFESIONAL", "SUBACTIVIDAD",
        "ATENCIONES", "HORAS_PROG", "RENDIMIENTO", "SEMAFORO", "TENDENCIA"
    ]].copy() if "TENDENCIA" in df.columns else df[[
        "MES", "SERVICIO", "PROFESIONAL", "SUBACTIVIDAD",
        "ATENCIONES", "HORAS_PROG", "RENDIMIENTO", "SEMAFORO"
    ]].copy()
    tabla["MES"] = tabla["MES"].astype(str)
    tabla = tabla.rename(columns={"PROFESIONAL": "MÉDICO (INICIALES)"})

    def colorear_fila(row):
        c     = color_semaforo(row["RENDIMIENTO"])
        texto = "white" if c == COLOR_ROJO else "black"
        return (
            [""] * (len(row) - 2) +
            [f"background-color:{c};color:{texto};font-weight:bold"] * 2
        )

    st.dataframe(
        tabla.style
             .apply(colorear_fila, axis=1)
             .format({"RENDIMIENTO": "{:.2f}"}),
        use_container_width=True,
        height=500,
    )

    col1, col2 = st.columns(2)
    with col1:
        csv = tabla.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", csv,
                           "rendimiento_medico.csv", "text/csv")
    with col2:
        xlsx_bytes = exportar_excel(tabla)
        st.download_button(
            "⬇️ Descargar Excel", xlsx_bytes,
            "rendimiento_medico.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
