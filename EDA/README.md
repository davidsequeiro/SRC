# 📊 EDA — Análisis Exploratorio Automatizado en Python

Este módulo contiene **herramientas interactivas y automáticas** para realizar análisis exploratorio de datos (EDA) de forma detallada, estructurada y explicativa.

> 💡 Diseñado para facilitar el análisis estadístico, la interpretación de resultados y la generación de informes claros en Jupyter Notebooks o scripts.

---

## 📁 Estructura de la carpeta `EDA`

```
EDA/
│
├── Data/ # 📂 Archivos de datos a analizar (CSV, Excel, Parquet...)
├── Document/ # 📓 Jupyter Notebooks con teoría y análisis
│ ├── doc-auxiliar.ipynb # Resumen teórico de estadística descriptiva y exploratoria
│ └── plantilla-eda.ipynb # Notebook principal para ejecutar el análisis
├── SRC/ # 🧠 Código fuente (clases en Python)
│ ├── Class_EDA.py # Clase principal: EDAHelper
│ └── Class_Test.py # Copia auxiliar de clase de tests estadísticos
└── requirements.txt # 📦 Librerías necesarias
└── environment.yml # 📦 Librerías necesarias (Conda)
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

```batch
pip install numpy pandas matplotlib seaborn scipy plotly statsmodels ipykernel openpyxl nbformat
```

## ⚙️ Cómo usarlo

1.- Clona el repositorio
2.- Añade el dataset que quieres analizar a la carpeta Data
3.- Introduce el nombre.extension del dataset en el sitio indicado del Jupyter.
4.- Ejecuta el análisis

Abre el archivo plantilla-eda.ipynb en Jupyter Notebook.

Asegúrate de tener tus datos en la carpeta Data/.

Ejecuta las celdas paso a paso para realizar todo el análisis.

Personaliza las fases según tus necesidades.

## 🧩 Integración con otros módulos

Este módulo está pensado para integrarse con el archivo Class_Test.py, que contiene todos los tests estadísticos utilizados por Class_EDA.

También puede ampliarse para incluir:

📁 Exportación de resultados

📈 Visualización avanzada

💬 Generación de informes automáticos

## 🛠️ Autor y Licencia

Creado por David Sequeiro usando ChatGPT
Licencia MIT — puedes usarlo, modificarlo y distribuirlo libremente.

🙌 Proyecto creado con fines formativos, de análisis y de divulgación estadística aplicada a Python.
