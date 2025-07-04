# 📊 EDAHelper — Análisis Exploratorio Automatizado en Python

Este módulo contiene **herramientas interactivas y automáticas** para realizar análisis exploratorio de datos (EDA) de forma detallada, estructurada y explicativa.

> 💡 Diseñado para facilitar el análisis estadístico, la interpretación de resultados y la generación de informes claros en Jupyter Notebooks o scripts.

---

## 📁 Estructura de la carpeta `EDA`

```
EDA/
│
├── data/ # 📂 Archivos de datos a analizar (CSV, Excel, Parquet...)
├── document/ # 📓 Jupyter Notebooks con teoría y análisis
│ ├── doc-auxiliar.ipynb # Resumen teórico de estadística descriptiva y exploratoria
│ └── plantilla-eda.ipynb # Notebook principal para ejecutar el análisis
├── src/ # 🧠 Código fuente (clases en Python)
│ ├── Class_EDA.py # Clase principal: EDAHelper
│ └── Class_Test.py # Copia auxiliar de clase de tests estadísticos
└── requirements.txt # 📦 Librerías necesarias
```

---

## 🔍 ¿Qué es `EDAHelper`?

`EDAHelper` es una clase Python pensada para ayudarte a realizar **análisis exploratorios completos** de forma semiautomática:

- ✅ Limpieza y resumen inicial de datos
- 🔢 Análisis numérico, categórico y de fechas
- 📊 Visualizaciones interactivas con Plotly
- 🧪 Pruebas estadísticas automáticas
- 🧠 Conclusiones interpretativas en lenguaje natural
- 📋 Generación de informes paso a paso

---

## ✨ Características

- **Fases organizadas**: desde la carga hasta las conclusiones
- **Pruebas estadísticas adaptativas**: t-test, ANOVA, Welch, Chi², etc.
- **Selección interactiva de columnas**
- **Visualizaciones integradas** (histogramas, boxplots, Q-Q, heatmaps…)
- **Conclusiones automáticas por variable o combinación**
- **Detección de outliers, normalidad, simetría, curtosis y más**
- **Compatibilidad con `Class_Test` para tests avanzados**

---

## 🧪 Requisitos

Asegúrate de tener las siguientes librerías instaladas:

```bash
pip install pandas numpy scipy statsmodels plotly openpyxl
```

## ⚙️ Cómo usarlo

Abre el archivo plantilla-eda.ipynb en Jupyter Notebook.

Asegúrate de tener tus datos en la carpeta data/.

Ejecuta las celdas paso a paso para realizar todo el análisis.

Personaliza las fases según tus necesidades.

## 🧩 Integración con otros módulos

Este módulo está pensado para integrarse con el archivo Class_Test.py, que contiene todos los tests estadísticos utilizados por EDAHelper.

También puede ampliarse para incluir:

📁 Exportación de resultados

📈 Visualización avanzada

💬 Generación de informes automáticos

## 🛠️ Autor y Licencia

Creado por David Sequeiro mediante ChatGPT
Licencia MIT — puedes usarlo, modificarlo y distribuirlo libremente.

🙌 Proyecto creado con fines formativos, de análisis y de divulgación estadística aplicada a Python.
