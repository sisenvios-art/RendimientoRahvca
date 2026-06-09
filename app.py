# =============================================================================
# Dashboard: Rendimiento Hora Médico - H. II Huancavelica 2026
# =============================================================================
# Descripción : Evalúa el rendimiento hora médico por mes, servicio y médico
# Cálculo     : ATE / HRAS_PROG  (atenciones / horas programadas)
# Estándar    : 5 atenciones por hora
# Semáforo    : Verde >= 5.0 | Amarillo 4.5–4.99 | Rojo < 4.5
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
UMBRAL_RIESGO  = 4.5   # por debajo de este valor se considera "En riesgo"

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


# ── CARGA Y PROCESAMIENTO DE DATOS ────────────────────────────────────────────

@st.cache_data(ttl=600)  # Cachea 10 minutos para no sobrecargar Supabase
def cargar_datos() -> pd.DataFrame:
    """
    Consulta la tabla de Supabase, filtra por grupo ocupacional MEDICO
    y las subactividades definidas, calcula el rendimiento (ATE / HRAS_PROG)
    y anonimiza los nombres. Retorna un DataFrame agrupado y ordenado.

    Estrategia de filtrado: se traen solo las filas del grupo MEDICO
    paginando de 1000 en 1000, y el filtro de subactividad se aplica
    en Python para evitar problemas de case-sensitivity en Supabase.
    """
    supabase = conectar_supabase()
    tabla    = st.secrets.get("SUPABASE_TABLE", "hras_efectivas")

    # Columnas a traer — PERIODO en lugar de MES (no existe esa columna)
    columnas = '"PERIODO","SERVICIO","PROFESIONAL","SUBACTIVIDAD","ATE","HRAS_PROG","GRPO_OCUPACIONAL"'

    # Paginación: traer todas las filas del grupo MEDICO de 1000 en 1000
    # Se omite el filtro .eq() para evitar problemas de case-sensitivity
    # y se filtra todo en Python
    todos  = []
    offset = 0
    batch  = 1000

    while True:
        respuesta = (
            supabase.table(tabla)
            .select(columnas)
            .range(offset, offset + batch - 1)
            .execute()
        )
        filas = respuesta.data
        if not filas:
            break
        todos.extend(filas)
        if len(filas) < batch:
            break  # última página
        offset += batch

    if not todos:
        return pd.DataFrame()

    datos = pd.DataFrame(todos)

    # Depuración temporal — mostrar valores únicos recibidos
    st.sidebar.markdown("**DEBUG (eliminar luego)**")
    st.sidebar.write("Filas totales:", len(datos))
    st.sidebar.write("Columnas:", list(datos.columns))
    if "GRPO_OCUPACIONAL" in datos.columns:
        st.sidebar.write("Grupos únicos:", datos["GRPO_OCUPACIONAL"].unique().tolist())
    if "SUBACTIVIDAD" in datos.columns:
        st.sidebar.write("Subactividades únicas:", datos["SUBACTIVIDAD"].unique().tolist()[:10])

    # Filtrar grupo y subactividades en Python
    datos = datos[datos["GRPO_OCUPACIONAL"] == GRUPO].copy()
    datos = datos[datos["SUBACTIVIDAD"].isin(SUBACTIVIDADES)].copy()

    if datos.empty:
        return pd.DataFrame()

    # Derivar columna MES desde PERIODO (formato dd/mm/yyyy o yyyy-mm-dd)
    MAPA_MESES = {
        1: "Enero",     2: "Febrero",   3: "Marzo",
        4: "Abril",     5: "Mayo",      6: "Junio",
        7: "Julio",     8: "Agosto",    9: "Setiembre",
        10: "Octubre",  11: "Noviembre", 12: "Diciembre"
    }
    fechas       = pd.to_datetime(datos["PERIODO"], dayfirst=True, errors="coerce")
    datos["MES"] = fechas.dt.month.map(MAPA_MESES)

    # Convertir columnas numéricas; reemplaza valores no numéricos con 0
    datos["ATE"]       = pd.to_numeric(datos["ATE"],       errors="coerce").fillna(0)
    datos["HRAS_PROG"] = pd.to_numeric(datos["HRAS_PROG"], errors="coerce").fillna(0)

    # Anonimizar nombres ANTES de agrupar para no exponer datos personales
    datos["PROFESIONAL"] = datos["PROFESIONAL"].apply(anonimizar_nombre)

    # Agrupar por mes, servicio, médico y subactividad
    grp = datos.groupby(
        ["MES", "SERVICIO", "PROFESIONAL", "SUBACTIVIDAD"], as_index=False
    ).agg(
        ATENCIONES=("ATE",       "sum"),
        HORAS_PROG=("HRAS_PROG", "sum"),
    )

    # Aplicar orden cronológico solo a los meses con datos reales
    meses_presentes = [m for m in MESES_ORDER if m in grp["MES"].values]
    grp["MES"] = pd.Categorical(grp["MES"], categories=meses_presentes, ordered=True)
    grp.sort_values(["MES", "SERVICIO", "PROFESIONAL"], inplace=True)

    # Calcular rendimiento; NaN si HRAS_PROG = 0 para evitar división por cero
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
        hover_data={"SERVICIO": True, "ATENCIONES": True, "HORAS_PROG": True},
        title="Rendimiento acumulado por médico (iniciales)",
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
