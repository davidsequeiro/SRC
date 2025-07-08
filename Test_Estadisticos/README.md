# 📊 Test_Estadísticos

Este módulo contiene la clase `StatisticalTests`, una herramienta avanzada, reutilizable e independiente que implementa una amplia gama de **tests estadísticos automáticos**, organizada por categorías y pensada para **análisis exploratorio, pruebas de hipótesis y validación de supuestos**.

---

## 🔍 ¿Qué es `StatisticalTests.py`?

Una clase modular con **métodos estáticos** para aplicar más de 25 tests estadísticos, todos devuelven resultados en formato estructurado (diccionario) con:

- Nombre del test
- Estadístico
- p-valor
- Resultado (`Rechazar H0` o `No rechazar H0`)
- Conclusión explicativa
- Recomendación práctica

---

## 📚 Tests incluidos

### ✅ Tests de normalidad:

- `shapiro_wilk_test`
- `kolmogorov_smirnov_test`
- `anderson_darling_test`
- `dagostino_test`
- `jarque_bera_test`

### ✅ Homogeneidad de varianzas:

- `levene_test`
- `bartlett_test`
- `brown_forsythe_test`

### ✅ Comparación de medias o distribuciones (2 grupos):

- `ttest_independent`
- `mann_whitney_u_test`
- `wilcoxon_signed_rank_test`

### ✅ Comparación de medias (más de 2 grupos):

- `anova_classic`
- `welch_anova`
- `kruskal_wallis_test`

### ✅ Correlación:

- `pearson_correlation`
- `spearman_correlation`
- `kendall_tau_correlation`

### ✅ Variables categóricas:

- `chi2_test`
- `fisher_exact_test`
- `mcnemar_test`

### ✅ Tests adicionales:

- `skewness_test` (asimetría)
- `kurtosis_test` (curtosis)
- `ks_2sample_test` (Kolmogorov-Smirnov para 2 muestras)
- `durbin_watson_test` (autocorrelación de residuos)

---

## 🧠 ¿Cómo usar este módulo?

Todos los métodos están pensados para:

- 🧪 Ser usados directamente en **Jupyter Notebook**, scripts `.py`, clases personalizadas o pipelines de análisis.
- 🔗 Integrarse fácilmente con otros módulos como `EDAHelper`, facilitando un análisis estadístico automatizado y explicativo.

---

## ✨ Características destacadas

- 📦 **Modular y reutilizable**: puedes importar solo lo que necesites.
- 📊 **Resultados estructurados**: todos los tests devuelven diccionarios con estadístico, p-valor, conclusión y recomendación.
- 💬 **Conclusiones interpretables automáticamente**: ideales para análisis guiados.
- 🧪 **Tests automáticos para cada tipo de análisis** (normalidad, correlación, medias, homocedasticidad...).
- 🧰 **Ideal para análisis exploratorio, validación de supuestos, modelos predictivos y tests A/B**.

---

## 🧩 Requisitos

Este módulo requiere las siguientes librerías de Python:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`

### 📥 Instalación rápida

```bash
pip install numpy pandas scipy statsmodels
```

## 🛠️ Autor y Licencia

Creado por David Sequeiro, usando ChatGPT, como parte de un repositorio de herramientas estadísticas y analíticas en Python.
Distribuido bajo licencia MIT — libre para uso, modificación y distribución.
