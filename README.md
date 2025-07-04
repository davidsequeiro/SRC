# SRC

# 📚 Repositorio de Recursos para Análisis de Datos y Estadística

Este repositorio centraliza herramientas, clases y recursos creados para facilitar el análisis de datos, la estadística aplicada y la exploración de datasets de forma profesional, modular y reutilizable.

El objetivo es construir un conjunto de utilidades que permitan:

✅ Automatizar análisis exploratorios (EDA)  
✅ Aplicar tests estadísticos adaptativos  
✅ Exportar resultados de forma organizada  
✅ Documentar y entender cada fase del análisis  
✅ Reutilizar clases y funciones de forma independiente

---

## 🧭 Estructura del repositorio

### 📁 `EDA/`

Contiene todos los recursos necesarios para realizar un análisis exploratorio completo de un dataset.

- `data/`: archivos de datos a analizar (CSV, Excel, Parquet, etc.).
- `document/`: notebooks Jupyter con ejecución y teoría.
  - `plantilla-eda.ipynb`: plantilla ejecutable con el análisis completo.
  - `doc-auxiliar.ipynb`: resumen teórico de estadística y métricas.
- `src/`: código fuente Python:
  - `Class_EDA.py`: clase principal `EDAHelper` para análisis automatizado.
  - `Class_Test.py`: clase `StatisticalTests` integrada para pruebas estadísticas.

📌 _Ideal para proyectos donde se quiera ejecutar un EDA detallado en pocos pasos._

---

### 📁 `Test_Estadísticos/`

Contiene una versión independiente de la clase `Class_Test` para su uso modular.

- `Class_Test.py`: clase con funciones estadísticas reutilizables.

📌 _Pensado para importar directamente en cualquier otro proyecto de análisis._

---

## ✍️ ¿Cómo colaborar o usarlo?

Puedes descargar el repositorio completo o solo las clases que necesites.  
Si quieres contribuir con nuevas funciones, notebooks o recursos, no dudes en hacer un **fork** o enviar un **pull request**.

---

## 🧠 Visión

Este repositorio nace como un espacio para reunir todas las herramientas creadas a lo largo del aprendizaje, aplicarlas a nuevos análisis y compartirlas con quien lo necesite.

📌 **Aprender, aplicar y compartir — en ese orden.**
