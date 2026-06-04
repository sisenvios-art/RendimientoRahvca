"""
Dashboard Rendimiento Hora Médico - H. II Huancavelica 2026
============================================================
Cálculo : ATE / HRAS_PROG
Estándar: 5 atenciones/hora
Semáforo: Verde >= 5 | Amarillo 4.5-4.99 | Rojo < 4.5

Despliegue en Streamlit Community Cloud (gratis):
  1. Sube este archivo y requirements.txt a un repo de GitHub
  2. Ve a https://share.streamlit.io → "New app" → apunta al repo
  3. Listo — la URL pública queda disponible al instante

Uso local:
  pip install -r requirements.txt
  streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────

MESES_ORDER    = ["Enero", "Febrero", "Marzo", "Abril", "Mayo"]
GRUPO          = "MEDICO"
SUBACTIVIDADES = ["CONSULTA MEDICA", "ATENCION ADULTO MAYOR FRAGIL"]
ESTANDAR       = 5.0

COLOR_VERDE    = "#4CAF50"
COLOR_AMARILLO = "#FFC107"
COLOR_ROJO     = "#F44336"
COLOR_ND       = "#9E9E9E"

# ── HELPERS ────────────────────────────────────────────────────────────────────

def semaforo(val):
    if pd.isna(val):  return "Sin dato"
    if val >= 5.0:    return "Óptimo"
    if val >= 4.5:    return "En riesgo"
    return "Bajo estándar"

def color_semaforo(val):
    s = semaforo(val)
    return {
        "Óptimo":        COLOR_VERDE,
        "En riesgo":     COLOR_AMARILLO,
        "Bajo estándar": COLOR_ROJO,
        "Sin dato":      COLOR_ND,
    }[s]

def estilo_celda(val):
    c = color_semaforo(val)
    return f"background-color:{c};color:{'white' if c == COLOR_ROJO else 'black'};font-weight:bold;text-align:center"

@st.cache_data
def cargar_datos(archivos_subidos):
    frames = []
    for archivo in archivos_subidos:
        mes = detectar_mes(archivo.name)
        df  = pd.read_csv(archivo, sep="|", low_memory=False)
        df["MES"] = mes
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    datos = pd.concat(frames, ignore_index=True)
    filtro = (
        (datos["GRPO_OCUPACIONAL"] == GRUPO) &
        (datos["SUBACTIVIDAD"].isin(SUBACTIVIDADES))
    )
    datos = datos[filtro].copy()
    datos["ATE"]       = pd.to_numeric(datos["ATE"],       errors="coerce").fillna(0)
    datos["HRAS_PROG"] = pd.to_numeric(datos["HRAS_PROG"], errors="coerce").fillna(0)

    grp = datos.groupby(
        ["MES", "SERVICIO", "PROFESIONAL", "SUBACTIVIDAD"], as_index=False
    ).agg(
        ATENCIONES =("ATE",       "sum"),
        HORAS_PROG =("HRAS_PROG", "sum"),
    )
    grp["MES"] = pd.Categorical(grp["MES"], categories=MESES_ORDER, ordered=True)
    grp.sort_values(["MES", "SERVICIO", "PROFESIONAL"], inplace=True)
    grp["RENDIMIENTO"] = np.where(
        grp["HORAS_PROG"] > 0,
        (grp["ATENCIONES"] / grp["HORAS_PROG"]).round(2),
        np.nan,
    )
    grp["SEMAFORO"] = grp["RENDIMIENTO"].apply(semaforo)
    grp["COLOR"]    = grp["RENDIMIENTO"].apply(color_semaforo)
    return grp

def detectar_mes(nombre_archivo: str) -> str:
    """Extrae el mes desde el nombre del archivo (YYYYMMDD) o lo pide al usuario."""
    name = nombre_archivo.upper()
    mapa = {
        "0101": "Enero",   "0201": "Febrero", "0301": "Marzo",
        "0401": "Abril",   "0501": "Mayo",    "0601": "Junio",
        "0701": "Julio",   "0801": "Agosto",  "0901": "Setiembre",
        "1001": "Octubre", "1101": "Noviembre","1201": "Diciembre",
    }
    for patron, mes in mapa.items():
        if patron in name:
            return mes
    # fallback: busca mes en el nombre
    for mes in MESES_ORDER:
        if mes.upper() in name:
            return mes
    return "Desconocido"

def exportar_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.drop(columns=["COLOR"], errors="ignore").to_excel(
            writer, index=False, sheet_name="Rendimiento"
        )
        wb  = writer.book
        ws  = writer.sheets["Rendimiento"]
        col_rend = df.columns.get_loc("RENDIMIENTO")
        fmt_v = wb.add_format({"bg_color": "#70AD47", "bold": True, "num_format": "0.00", "border": 1})
        fmt_a = wb.add_format({"bg_color": "#FFD966", "bold": True, "num_format": "0.00", "border": 1})
        fmt_r = wb.add_format({"bg_color": "#FF6B6B", "bold": True, "num_format": "0.00", "border": 1, "font_color": "white"})
        for row_num, val in enumerate(df["RENDIMIENTO"], start=1):
            if pd.isna(val): continue
            fmt = fmt_v if val >= 5.0 else (fmt_a if val >= 4.5 else fmt_r)
            ws.write(row_num, col_rend, val, fmt)
    return buf.getvalue()


# ── LAYOUT ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rendimiento Hora Médico",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Rendimiento Hora Médico")
st.caption("H. II Huancavelica 2026 · Grupo Ocupacional: MÉDICO · Estándar: 5 atenc/hora · Cálculo: ATE ÷ HRAS_PROG")

# ── CARGA DE ARCHIVOS ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Cargar archivos")
    archivos = st.file_uploader(
        "Sube los archivos TXT (separados por |)",
        type=["txt", "csv"],
        accept_multiple_files=True,
    )
    st.markdown("---")
    st.markdown(
        "**Semáforo**\n"
        "- 🟢 **Óptimo** ≥ 5.0\n"
        "- 🟡 **En riesgo** 4.5 – 4.99\n"
        "- 🔴 **Bajo estándar** < 4.5"
    )

if not archivos:
    st.info("👈 Sube uno o más archivos TXT desde el panel lateral para comenzar.")
    st.stop()

# ── DATOS ──────────────────────────────────────────────────────────────────────

grp = cargar_datos(tuple(archivos))

if grp.empty:
    st.error("No se encontraron registros con los filtros aplicados. Revisa los archivos.")
    st.stop()

# ── FILTROS ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔍 Filtros")

    meses_disp = [m for m in MESES_ORDER if m in grp["MES"].values]
    sel_mes = st.multiselect("Mes", meses_disp, default=meses_disp)

    servicios_disp = sorted(grp["SERVICIO"].unique())
    sel_srv = st.multiselect("Servicio", servicios_disp, default=servicios_disp)

    medicos_disp = sorted(grp["PROFESIONAL"].unique())
    sel_med = st.multiselect("Médico", medicos_disp, default=medicos_disp)

    subs_disp = sorted(grp["SUBACTIVIDAD"].unique())
    sel_sub = st.multiselect("Subactividad", subs_disp, default=subs_disp)

    sel_sem = st.multiselect(
        "Semáforo",
        ["Óptimo", "En riesgo", "Bajo estándar", "Sin dato"],
        default=["Óptimo", "En riesgo", "Bajo estándar", "Sin dato"],
    )

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

# ── KPIs ───────────────────────────────────────────────────────────────────────

vals = df["RENDIMIENTO"].dropna()
n_verde    = (vals >= 5.0).sum()
n_amarillo = ((vals >= 4.5) & (vals < 5.0)).sum()
n_rojo     = (vals < 4.5).sum()
n_nd       = df["RENDIMIENTO"].isna().sum()
pct_verde  = n_verde / len(vals) * 100 if len(vals) else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📋 Registros",        len(df))
k2.metric("📊 Promedio",         f"{vals.mean():.2f}" if len(vals) else "—")
k3.metric("🟢 Óptimo",           f"{n_verde}  ({pct_verde:.0f}%)")
k4.metric("🟡 En riesgo",        str(n_amarillo))
k5.metric("🔴 Bajo estándar",    str(n_rojo))

st.markdown("---")

# ── GRÁFICOS ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Evolución mensual",
    "🏢 Por servicio",
    "👨‍⚕️ Por médico",
    "📋 Tabla detalle",
])

# Tab 1 — Evolución mensual
with tab1:
    evo = (
        df.groupby("MES", as_index=False, observed=True)
          .agg(ATENCIONES=("ATENCIONES","sum"), HORAS_PROG=("HORAS_PROG","sum"))
    )
    evo["RENDIMIENTO"] = np.where(
        evo["HORAS_PROG"] > 0, (evo["ATENCIONES"] / evo["HORAS_PROG"]).round(2), np.nan
    )
    evo["COLOR"] = evo["RENDIMIENTO"].apply(color_semaforo)

    fig = go.Figure()
    fig.add_hline(y=ESTANDAR,     line_dash="dash", line_color=COLOR_VERDE,    annotation_text="Estándar (5.0)")
    fig.add_hline(y=4.5,          line_dash="dot",  line_color=COLOR_AMARILLO, annotation_text="Mínimo aceptable (4.5)")
    fig.add_trace(go.Scatter(
        x=evo["MES"].astype(str), y=evo["RENDIMIENTO"],
        mode="lines+markers+text",
        marker=dict(color=evo["COLOR"], size=12, line=dict(width=1.5, color="white")),
        line=dict(color="#607D8B", width=2),
        text=evo["RENDIMIENTO"].apply(lambda v: f"{v:.2f}" if not pd.isna(v) else ""),
        textposition="top center",
        name="Rendimiento",
    ))
    fig.update_layout(
        title="Rendimiento promedio mensual (todos los servicios y médicos seleccionados)",
        xaxis_title="Mes", yaxis_title="Atenciones / Hora programada",
        plot_bgcolor="white", height=420,
        yaxis=dict(gridcolor="#ECEFF1"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Por subactividad
    evo_sub = (
        df.groupby(["MES","SUBACTIVIDAD"], as_index=False, observed=True)
          .agg(ATENCIONES=("ATENCIONES","sum"), HORAS_PROG=("HORAS_PROG","sum"))
    )
    evo_sub["RENDIMIENTO"] = np.where(
        evo_sub["HORAS_PROG"] > 0,
        (evo_sub["ATENCIONES"] / evo_sub["HORAS_PROG"]).round(2), np.nan
    )
    fig2 = px.line(
        evo_sub, x="MES", y="RENDIMIENTO", color="SUBACTIVIDAD",
        markers=True, title="Evolución mensual por subactividad",
        labels={"RENDIMIENTO":"Atenciones/hora","MES":"Mes"},
    )
    fig2.add_hline(y=ESTANDAR, line_dash="dash", line_color=COLOR_VERDE)
    fig2.update_layout(plot_bgcolor="white", height=380, yaxis=dict(gridcolor="#ECEFF1"))
    st.plotly_chart(fig2, use_container_width=True)

# Tab 2 — Por servicio
with tab2:
    srv_df = (
        df.groupby(["SERVICIO","MES"], as_index=False, observed=True)
          .agg(ATENCIONES=("ATENCIONES","sum"), HORAS_PROG=("HORAS_PROG","sum"))
    )
    srv_df["RENDIMIENTO"] = np.where(
        srv_df["HORAS_PROG"] > 0,
        (srv_df["ATENCIONES"] / srv_df["HORAS_PROG"]).round(2), np.nan
    )
    srv_df["COLOR"] = srv_df["RENDIMIENTO"].apply(color_semaforo)

    # Resumen acumulado por servicio
    srv_acum = (
        df.groupby("SERVICIO", as_index=False)
          .agg(ATENCIONES=("ATENCIONES","sum"), HORAS_PROG=("HORAS_PROG","sum"))
    )
    srv_acum["RENDIMIENTO"] = np.where(
        srv_acum["HORAS_PROG"] > 0,
        (srv_acum["ATENCIONES"] / srv_acum["HORAS_PROG"]).round(2), np.nan
    )
    srv_acum["COLOR"] = srv_acum["RENDIMIENTO"].apply(color_semaforo)
    srv_acum.sort_values("RENDIMIENTO", ascending=True, inplace=True)

    fig3 = go.Figure(go.Bar(
        x=srv_acum["RENDIMIENTO"], y=srv_acum["SERVICIO"],
        orientation="h",
        marker_color=srv_acum["COLOR"],
        text=srv_acum["RENDIMIENTO"].apply(lambda v: f"{v:.2f}" if not pd.isna(v) else ""),
        textposition="outside",
    ))
    fig3.add_vline(x=ESTANDAR, line_dash="dash", line_color=COLOR_VERDE, annotation_text="Estándar")
    fig3.update_layout(
        title="Rendimiento acumulado por servicio (período seleccionado)",
        xaxis_title="Atenciones / hora", yaxis_title="",
        plot_bgcolor="white", height=max(400, len(srv_acum) * 30),
        xaxis=dict(gridcolor="#ECEFF1"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap servicio x mes
    pivot = srv_df.pivot_table(index="SERVICIO", columns="MES", values="RENDIMIENTO")
    pivot = pivot.reindex(columns=[m for m in MESES_ORDER if m in pivot.columns])

    fig4 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  COLOR_ROJO],
            [0.45, COLOR_ROJO],
            [0.45, COLOR_AMARILLO],
            [0.50, COLOR_AMARILLO],
            [0.50, COLOR_VERDE],
            [1.0,  COLOR_VERDE],
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
        xaxis_title="Mes", yaxis_title="",
    )
    st.plotly_chart(fig4, use_container_width=True)

# Tab 3 — Por médico
with tab3:
    med_df = (
        df.groupby(["PROFESIONAL","SERVICIO"], as_index=False)
          .agg(ATENCIONES=("ATENCIONES","sum"), HORAS_PROG=("HORAS_PROG","sum"))
    )
    med_df["RENDIMIENTO"] = np.where(
        med_df["HORAS_PROG"] > 0,
        (med_df["ATENCIONES"] / med_df["HORAS_PROG"]).round(2), np.nan
    )
    med_df["COLOR"]    = med_df["RENDIMIENTO"].apply(color_semaforo)
    med_df["SEMAFORO"] = med_df["RENDIMIENTO"].apply(semaforo)
    med_df.sort_values("RENDIMIENTO", ascending=True, inplace=True)

    fig5 = px.bar(
        med_df, x="RENDIMIENTO", y="PROFESIONAL", color="SEMAFORO",
        orientation="h",
        color_discrete_map={
            "Óptimo":        COLOR_VERDE,
            "En riesgo":     COLOR_AMARILLO,
            "Bajo estándar": COLOR_ROJO,
            "Sin dato":      COLOR_ND,
        },
        hover_data={"SERVICIO": True, "ATENCIONES": True, "HORAS_PROG": True},
        title="Rendimiento acumulado por médico (período seleccionado)",
        labels={"RENDIMIENTO": "Atenciones/hora", "PROFESIONAL": ""},
    )
    fig5.add_vline(x=ESTANDAR, line_dash="dash", line_color="#333", annotation_text="Estándar")
    fig5.update_layout(
        plot_bgcolor="white", height=max(500, len(med_df) * 22),
        xaxis=dict(gridcolor="#ECEFF1"), showlegend=True,
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Distribución semáforo
    sem_counts = med_df["SEMAFORO"].value_counts().reset_index()
    sem_counts.columns = ["Semáforo", "Médicos"]
    sem_counts["Color"] = sem_counts["Semáforo"].map({
        "Óptimo": COLOR_VERDE, "En riesgo": COLOR_AMARILLO,
        "Bajo estándar": COLOR_ROJO, "Sin dato": COLOR_ND,
    })
    fig6 = px.pie(
        sem_counts, names="Semáforo", values="Médicos",
        color="Semáforo",
        color_discrete_map={
            "Óptimo": COLOR_VERDE, "En riesgo": COLOR_AMARILLO,
            "Bajo estándar": COLOR_ROJO, "Sin dato": COLOR_ND,
        },
        title="Distribución de médicos por semáforo",
        hole=0.4,
    )
    st.plotly_chart(fig6, use_container_width=True)

# Tab 4 — Tabla detalle
with tab4:
    tabla = df[[
        "MES","SERVICIO","PROFESIONAL","SUBACTIVIDAD",
        "ATENCIONES","HORAS_PROG","RENDIMIENTO","SEMAFORO"
    ]].copy()
    tabla["MES"] = tabla["MES"].astype(str)

    def colorear_fila(row):
        c = color_semaforo(row["RENDIMIENTO"])
        texto = "white" if c == COLOR_ROJO else "black"
        return [""] * (len(row) - 2) + [
            f"background-color:{c};color:{texto};font-weight:bold",
            f"background-color:{c};color:{texto};font-weight:bold",
        ]

    st.dataframe(
        tabla.style.apply(colorear_fila, axis=1).format({"RENDIMIENTO": "{:.2f}"}),
        use_container_width=True,
        height=500,
    )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = tabla.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", csv, "rendimiento_medico.csv", "text/csv")
    with col_dl2:
        xlsx_bytes = exportar_excel(tabla)
        st.download_button(
            "⬇️ Descargar Excel", xlsx_bytes,
            "rendimiento_medico.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
