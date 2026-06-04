# Rendimiento Hora Médico — Dashboard Streamlit

Dashboard interactivo para evaluar el rendimiento hora médico del grupo ocupacional MÉDICO.

**Cálculo:** `ATE ÷ HRAS_PROG`  
**Estándar:** 5 atenciones/hora  
**Semáforo:** 🟢 ≥ 5.0 | 🟡 4.5–4.99 | 🔴 < 4.5  

---

## 🚀 Despliegue en Streamlit Community Cloud (gratis)

1. Sube esta carpeta a un repositorio de GitHub (puede ser privado).
2. Ve a [https://share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Clic en **"New app"** → selecciona el repositorio y el archivo `app.py`.
4. Clic en **"Deploy"** — en 1–2 minutos tendrás una URL pública.

> El repositorio debe contener al menos: `app.py` y `requirements.txt`.

---

## 💻 Uso local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

Luego abre `http://localhost:8501` en tu navegador.

---

## 📂 Estructura del repositorio

```
tu_repo/
├── app.py              ← Dashboard principal
├── requirements.txt    ← Dependencias Python
└── README.md
```

Los archivos TXT de horas efectivas **no se suben al repo** — se cargan directamente desde el panel lateral del dashboard cada vez que se use.

---

## 📋 Formato de archivos de entrada

- Separador: `|` (pipe)
- Columnas requeridas: `GRPO_OCUPACIONAL`, `SUBACTIVIDAD`, `ATE`, `HRAS_PROG`, `SERVICIO`, `PROFESIONAL`
- El mes se detecta automáticamente desde el nombre del archivo (ej. `342_20260101_...` → Enero)
