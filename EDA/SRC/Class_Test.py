# Class_Test.py

import numpy as np
import pandas as pd
from scipy.stats import (
    shapiro, kstest, anderson,
    pearsonr, spearmanr, kendalltau,
    ttest_ind, mannwhitneyu, f_oneway,
    levene, chi2_contingency, fisher_exact,
    kruskal, wilcoxon, bartlett, skewtest, kurtosistest, normaltest, jarque_bera
)
from statsmodels.stats.oneway import anova_oneway
import statsmodels.api as sm

"""
Implementa una amplia gama de tests estadísticos comunes y avanzados para facilitar
el análisis univariante, bivariante y tests de hipótesis. Cada test está implementado
como método estático que recibe los datos correspondientes y devuelve un diccionario
estructurado con los resultados y una interpretación básica.

Categorías de tests incluidos:

1. Tests de normalidad (evaluar si una muestra sigue una distribución normal):
    - shapiro_wilk: Test clásico para muestras pequeñas a medianas.
    - anderson_darling: Test basado en distancias entre distribuciones.
    - kolmogorov_smirnov: Test general para comparar con una distribución teórica.
    - dagostino_k2 (normaltest): Test combinado basado en skewness y curtosis.
    - jarque_bera: Test basado en skewness y curtosis para normalidad.

2. Tests de homocedasticidad (comprobar igualdad de varianzas entre grupos):
    - levene: Test robusto para igualdad de varianzas.
    - bartlett: Test clásico, sensible a desviaciones de normalidad.
    - brown_forsythe: Variante robusta basada en medianas.

3. Tests paramétricos para comparación de medias o grupos:
    - ttest_ind: Test t de Student para dos grupos con varianzas iguales.
    - ttest_ind_welch: Variante para varianzas desiguales (Welch).
    - anova_oneway: ANOVA clásico para más de dos grupos.
    - welch_anova: ANOVA de Welch para varianzas desiguales, usando statsmodels.

4. Tests no paramétricos para comparación de grupos:
    - mannwhitneyu: Test de rango para dos grupos independientes.
    - kruskal: ANOVA no paramétrico para más de dos grupos.
    - wilcoxon: Test para muestras pareadas no paramétricas.

5. Tests para variables categóricas (asociación e independencia):
    - chi2_contingency: Test Chi-cuadrado para tablas de contingencia.
    - fisher_exact: Test exacto para tablas 2x2 pequeñas.
    - mcnemar: Test para tablas 2x2 dependientes (pares).

6. Tests de correlación (medición de asociación entre variables):
    - pearsonr: Correlación lineal paramétrica.
    - spearmanr: Correlación monotónica no paramétrica.
    - kendalltau: Correlación ordinal no paramétrica.

7. Otros tests estadísticos útiles:
    - durbin_watson: Test para autocorrelación de residuos en modelos.
    - ks_2sample: Test Kolmogorov-Smirnov para comparar dos muestras.
    - skewtest: Test para simetría de la distribución.
    - kurtosistest: Test para curtosis (apuntamiento de la distribución).

Uso:
    - Los métodos se llaman como métodos estáticos, pasando los datos adecuados.
    - Cada método retorna un diccionario con claves:
        * test_name: nombre del test
        * statistic: valor estadístico calculado
        * p_value: valor p del test
        * result: interpretación básica ("Rechazar H0" / "No rechazar H0")
        * conclusion: explicación sencilla del resultado
        * recommendation: sugerencias o contexto para interpretación y uso

Esta clase es útil para automatizar análisis estadísticos en fases univariantes,
bivariantes y para tests de hipótesis en procesos de análisis exploratorio de datos.
"""

class StatisticalTests:
    
    ALPHA_DEFAULT = 0.05

    @staticmethod
    def _format_result(test_name, stat, p, alpha=ALPHA_DEFAULT, greater_is_reject=False):
        """
        Formatea el resultado estándar de un test.
        """
        stat_str = None if stat is None else float(stat)
        if greater_is_reject:
            reject = p > alpha
        else:
            reject = p < alpha
        result = "Rechazar H0" if reject else "No rechazar H0"
        return {
            "test_name": test_name,
            "statistic": stat_str,
            "p_value": float(p) if p is not None else None,
            "result": result,
            "conclusion": "",
            "recommendation": ""
        }

    # --------------------
    # TESTS DE NORMALIDAD
    # --------------------

    @staticmethod
    def shapiro_wilk_test(data, alpha=ALPHA_DEFAULT):
        stat, p = shapiro(data)
        res = StatisticalTests._format_result("Shapiro-Wilk", stat, p, alpha)
        res["conclusion"] = "Los datos parecen normales" if res["result"] == "No rechazar H0" else "Los datos no parecen normales"
        res["recommendation"] = "Si no son normales, considera tests no paramétricos o transformaciones"
        return res

    @staticmethod
    def kolmogorov_smirnov_test(data, cdf='norm', alpha=ALPHA_DEFAULT):
        stat, p = kstest(data, cdf)
        res = StatisticalTests._format_result("Kolmogorov-Smirnov", stat, p, alpha)
        res["conclusion"] = "Los datos parecen seguir la distribución teórica" if res["result"] == "No rechazar H0" else "Los datos no siguen la distribución teórica"
        res["recommendation"] = "Útil para verificar ajustes de distribución"
        return res

    @staticmethod
    def anderson_darling_test(data, dist='norm', alpha=ALPHA_DEFAULT):
        ad = anderson(data, dist)
        stat = ad.statistic
        # Buscar índice de nivel de significancia más cercano a alpha
        crit_idx = None
        for i, sl in enumerate(ad.significance_level):
            if np.isclose(sl/100, alpha):
                crit_idx = i
                break
        if crit_idx is None:
            crit_idx = 2  # default al 5%
        crit_val = ad.critical_values[crit_idx]
        reject = stat > crit_val
        res = {
            "test_name": "Anderson-Darling",
            "statistic": stat,
            "p_value": None,
            "result": "Rechazar H0" if reject else "No rechazar H0",
            "conclusion": "Los datos parecen normales" if not reject else "Los datos no parecen normales",
            "recommendation": "Considerar tests no paramétricos si se rechaza H0"
        }
        return res
    
    # ... otros métodos ...

    @staticmethod
    def skewness_test(data, alpha=0.05):
        stat, p = skewtest(data)
        reject = p < alpha
        result = "Asimetría significativa" if reject else "No asimetría significativa"
        return {
            "test_name": "Test de simetría",
            "statistic": stat,
            "p_value": p,
            "result": result,
            "conclusion": "",
            "recommendation": ""
        }

    @staticmethod
    def kurtosis_test(data, alpha=0.05):
        stat, p = kurtosistest(data)
        reject = p < alpha
        result = "Curtosis significativa" if reject else "No curtosis significativa"
        return {
            "test_name": "Test de curtosis",
            "statistic": stat,
            "p_value": p,
            "result": result,
            "conclusion": "",
            "recommendation": ""
        }

    @staticmethod
    def dagostino_test(data, alpha=0.05):
        stat, p = normaltest(data)
        reject = p < alpha
        result = "No normal según D'Agostino K²" if reject else "Normal según D'Agostino K²"
        return {
            "test_name": "D'Agostino K² Test",
            "statistic": stat,
            "p_value": p,
            "result": result,
            "conclusion": "",
            "recommendation": ""
        }

    @staticmethod
    def jarque_bera_test(data, alpha=0.05):
        stat, p = jarque_bera(data)
        reject = p < alpha
        result = "No normal según Jarque-Bera" if reject else "Normal según Jarque-Bera"
        return {
            "test_name": "Jarque-Bera Test",
            "statistic": stat,
            "p_value": p,
            "result": result,
            "conclusion": "",
            "recommendation": ""
        }

    # --------------------------
    # TESTS DE CORRELACIÓN (BIVARIANTE)
    # --------------------------

    @staticmethod
    def pearson_correlation(x, y, alpha=ALPHA_DEFAULT):
        stat, p = pearsonr(x, y)
        res = StatisticalTests._format_result("Pearson Correlation", stat, p, alpha)
        res["conclusion"] = "Existe correlación lineal significativa" if res["result"] == "Rechazar H0" else "No existe correlación lineal significativa"
        res["recommendation"] = "Útil para relaciones lineales entre variables numéricas"
        return res

    @staticmethod
    def spearman_correlation(x, y, alpha=ALPHA_DEFAULT):
        stat, p = spearmanr(x, y)
        res = StatisticalTests._format_result("Spearman Correlation", stat, p, alpha)
        res["conclusion"] = "Existe correlación monotónica significativa" if res["result"] == "Rechazar H0" else "No existe correlación monotónica significativa"
        res["recommendation"] = "Útil para relaciones no lineales o rangos"
        return res

    @staticmethod
    def kendall_tau_correlation(x, y, alpha=ALPHA_DEFAULT):
        stat, p = kendalltau(x, y)
        res = StatisticalTests._format_result("Kendall Tau Correlation", stat, p, alpha)
        res["conclusion"] = "Existe correlación ordinal significativa" if res["result"] == "Rechazar H0" else "No existe correlación ordinal significativa"
        res["recommendation"] = "Para variables ordinales o rangos"
        return res

    # --------------------------
    # TESTS PARAMÉTRICOS Y NO PARAMÉTRICOS PARA DOS MUESTRAS O MÁS
    # --------------------------

    @staticmethod
    def ttest_independent(x1, x2, alpha=ALPHA_DEFAULT, equal_var=True):
        """
        Test t de Student para muestras independientes
        equal_var=True usa t clásico, False Welch
        """
        stat, p = ttest_ind(x1, x2, equal_var=equal_var)
        test_name = "t-test Student" if equal_var else "Welch's t-test"
        res = StatisticalTests._format_result(test_name, stat, p, alpha)
        res["conclusion"] = "Las medias de los dos grupos son significativamente diferentes" if res["result"] == "Rechazar H0" else "No se encontraron diferencias significativas en las medias"
        res["recommendation"] = "Validar supuestos y tamaño muestral"
        return res

    @staticmethod
    def mann_whitney_u_test(x1, x2, alpha=ALPHA_DEFAULT, alternative='two-sided'):
        """
        Test no paramétrico para dos muestras independientes
        """
        stat, p = mannwhitneyu(x1, x2, alternative=alternative)
        res = StatisticalTests._format_result("Mann-Whitney U Test", stat, p, alpha)
        res["conclusion"] = "Distribuciones significativamente diferentes" if res["result"] == "Rechazar H0" else "No se encontraron diferencias significativas entre distribuciones"
        res["recommendation"] = "Test no paramétrico para dos muestras independientes"
        return res

    @staticmethod
    def wilcoxon_signed_rank_test(x1, x2, alpha=ALPHA_DEFAULT):
        """
        Test no paramétrico para muestras relacionadas (pareadas)
        """
        stat, p = wilcoxon(x1, x2)
        res = StatisticalTests._format_result("Wilcoxon Signed-Rank Test", stat, p, alpha)
        res["conclusion"] = "Diferencias significativas entre muestras pareadas" if res["result"] == "Rechazar H0" else "No se encontraron diferencias significativas"
        res["recommendation"] = "Para muestras pareadas, alternativa no paramétrica al t-test pareado"
        return res

    @staticmethod
    def anova_classic(groups, alpha=ALPHA_DEFAULT):
        """
        ANOVA clásico para más de dos grupos
        """
        stat, p = f_oneway(*groups)
        res = StatisticalTests._format_result("ANOVA clásico", stat, p, alpha)
        res["conclusion"] = "Al menos un grupo difiere significativamente" if res["result"] == "Rechazar H0" else "No se encontraron diferencias significativas entre grupos"
        res["recommendation"] = "Usar si varianzas son iguales y muestras independientes"
        return res

    @staticmethod
    def levene_test(*groups, alpha=ALPHA_DEFAULT):
        """
        Test de Levene para homogeneidad de varianzas
        """
        stat, p = levene(*groups)
        res = StatisticalTests._format_result("Levene Test", stat, p, alpha)
        res["conclusion"] = "Varianzas significativamente diferentes" if res["result"] == "Rechazar H0" else "Varianzas homogéneas"
        res["recommendation"] = "Validar homocedasticidad antes de ANOVA o t-test"
        return res

    @staticmethod
    def bartlett_test(*groups, alpha=ALPHA_DEFAULT):
        """
        Test de Bartlett para homogeneidad de varianzas (más sensible a normalidad)
        """
        stat, p = bartlett(*groups)
        res = StatisticalTests._format_result("Bartlett Test", stat, p, alpha)
        res["conclusion"] = "Varianzas significativamente diferentes" if res["result"] == "Rechazar H0" else "Varianzas homogéneas"
        res["recommendation"] = "Validar homocedasticidad, sensible a desviaciones de normalidad"
        return res

    @staticmethod
    def welch_anova(groups, alpha=ALPHA_DEFAULT):
        """
        ANOVA de Welch para varianzas no iguales
        """
        data = []
        group_labels = []
        for i, g in enumerate(groups):
            data.extend(g)
            group_labels.extend([i]*len(g))
        df_w = pd.DataFrame({"value": data, "group": group_labels})
        mod = anova_oneway(df_w["value"], df_w["group"], use_var='unequal')
        p = mod.pvalue
        stat = mod.statistic
        res = StatisticalTests._format_result("Welch ANOVA", stat, p, alpha)
        res["conclusion"] = "Al menos un grupo difiere significativamente (Welch)" if res["result"] == "Rechazar H0" else "No se encontraron diferencias significativas (Welch)"
        res["recommendation"] = "Usar si varianzas no son iguales"
        return res

    @staticmethod
    def kruskal_wallis_test(groups, alpha=ALPHA_DEFAULT):
        """
        Test no paramétrico para comparar múltiples grupos independientes
        """
        stat, p = kruskal(*groups)
        res = StatisticalTests._format_result("Kruskal-Wallis", stat, p, alpha)
        res["conclusion"] = "Diferencias significativas entre grupos (no paramétrico)" if res["result"] == "Rechazar H0" else "No diferencias significativas"
        res["recommendation"] = "Alternativa no paramétrica a ANOVA"
        return res

    # --------------------------
    # TESTS PARA VARIABLES CATEGÓRICAS
    # --------------------------

    @staticmethod
    def chi2_test(contingency_table, alpha=ALPHA_DEFAULT):
        """
        Test Chi-cuadrado para tablas de contingencia
        """
        stat, p, dof, expected = chi2_contingency(contingency_table)
        res = StatisticalTests._format_result("Chi-cuadrado", stat, p, alpha)
        res["conclusion"] = "Variables asociadas (dependientes)" if res["result"] == "Rechazar H0" else "Variables independientes"
        res["recommendation"] = "No usar si frecuencias esperadas < 5 en >20% celdas"
        return res

    @staticmethod
    def fisher_exact_test(table, alpha=ALPHA_DEFAULT):
        """
        Test exacto de Fisher para tablas 2x2 pequeñas
        """
        stat, p = fisher_exact(table)
        res = StatisticalTests._format_result("Fisher Exact Test", stat, p, alpha)
        res["conclusion"] = "Variables asociadas" if res["result"] == "Rechazar H0" else "Variables independientes"
        res["recommendation"] = "Útil para tablas pequeñas"
        return res

    @staticmethod
    def mcnemar_test(table, alpha=ALPHA_DEFAULT):
        """
        Test de McNemar para tablas 2x2 dependientes (pareadas)
        """
        from statsmodels.stats.contingency_tables import mcnemar
        result = mcnemar(table)
        stat = result.statistic
        p = result.pvalue
        res = StatisticalTests._format_result("McNemar Test", stat, p, alpha)
        res["conclusion"] = "Cambio significativo entre variables pareadas" if res["result"] == "Rechazar H0" else "No cambio significativo"
        res["recommendation"] = "Para tablas 2x2 de datos dependientes"
        return res

    # --------------------------
    # TESTS ADICIONALES Y DE APOYO
    # --------------------------

    @staticmethod
    def brown_forsythe_test(*groups, alpha=ALPHA_DEFAULT):
        """
        Variante del test de Levene usando la mediana
        """
        stat, p = levene(*groups, center='median')
        res = StatisticalTests._format_result("Brown-Forsythe Test", stat, p, alpha)
        res["conclusion"] = "Varianzas significativamente diferentes" if res["result"] == "Rechazar H0" else "Varianzas homogéneas"
        res["recommendation"] = "Validar homocedasticidad"
        return res

    @staticmethod
    def durbin_watson_test(residuals):
        """
        Test de Durbin-Watson para autocorrelación de residuos
        """
        dw = sm.stats.stattools.durbin_watson(residuals)
        res = {
            "test_name": "Durbin-Watson Test",
            "statistic": dw,
            "p_value": None,
            "result": None,
            "conclusion": "Valor cercano a 2 indica ausencia de autocorrelación",
            "recommendation": "Útil para analizar residuos de modelos"
        }
        return res

    @staticmethod
    def ks_2sample_test(x1, x2, alpha=ALPHA_DEFAULT):
        """
        Test de Kolmogorov-Smirnov para comparar dos distribuciones
        """
        stat, p = kstest(x1, x2)
        res = StatisticalTests._format_result("Kolmogorov-Smirnov 2 muestras", stat, p, alpha)
        res["conclusion"] = "Distribuciones significativamente diferentes" if res["result"] == "Rechazar H0" else "No se encontraron diferencias significativas"
        res["recommendation"] = "Test no paramétrico para comparar distribuciones"
        return res

     @staticmethod
    def one_sample_t_test(sample, popmean, alpha=0.05):
        """
        Test t de una muestra para comparar la media con un valor conocido.
        """
        from scipy.stats import ttest_1samp
        stat, p = ttest_1samp(sample, popmean)
        res = StatisticalTests._format_result("t-test una muestra", stat, p, alpha)
        res["conclusion"] = "La media difiere significativamente del valor esperado" if res["result"] == "Rechazar H0" else "La media no difiere significativamente del valor esperado"
        res["recommendation"] = "Útil para contrastar medias frente a un valor de referencia"
        return res