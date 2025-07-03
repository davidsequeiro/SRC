# eda_helper.py

# EDAHelper: Exploración de Datos Automatizada y Modular
import os
import warnings

# Librerias ETL
import pandas as pd
import numpy as np

#Librerias Visualizacion
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

#Libreria de tiempo
from datetime import datetime

#Libreria Estadisticas
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde, entropy, chi2_contingency, ttest_ind, mannwhitneyu, f_oneway, levene, shapiro, pearsonr, spearmanr, iqr, chi2_contingency
from itertools import zip_longest
from statsmodels.stats.oneway import anova_oneway
import statsmodels.api as sm



# CLASE PARA EDA
class EDAHelper:
    def __init__(self, file_path,df=None):
        self.file_path = file_path
        self.df = df
        self.logs = []

    def log(self, msg):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self.logs.append(f"{timestamp} {msg}")
        print(f"\n✅ {msg}\n" + "- "*40)

    def print_logs(self):
        print("\n📋 RESUMEN:")
        for log in self.logs:
            print(log)
        self.logs = []

    def run_fase0_Datos_Crudos(self):
        # self.load_data() (PASO INCLUIDO EN EL JUPYTER)
        self.format_column_names()
        self.show_dataframe()
        
    # --- FASE 1: Análisis Preliminar ---
    def run_fase1_preliminar(self):
        self.show_info()
        self.group_by_dtype()
        self.nulls_and_duplicates()
        self.detect_booleans()
        #self.detect_date_columns()
        self.print_logs()

    def load_data(self):
        extension = os.path.splitext(self.file_path)[1].lower()
        try:
            if extension == '.csv':
                self.df = pd.read_csv(self.file_path)
            elif extension == '.parquet':
                self.df = pd.read_parquet(self.file_path)
            elif extension in ['.xls', '.xlsx']:
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError(f"Extensión no soportada: {extension}")
            print(f"✅ Archivo cargado correctamente: {extension}")
        except Exception as e:
            print(f"❌ Error al cargar archivo: {e}")
            
    def format_column_names(self):
        print("📌 FORMATEAR TITULOS COLUMNAS\n"+ "-"*40)
        self.df.columns = [col.strip().lower().replace(" ", "_").capitalize() for col in self.df.columns]
        self.log("Titulos Columnas formateadas correctamente")
        
    def show_dataframe(self):
        print("\n📌 DATASET CARGADO\n"+ "-"*40)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(self.df.head())
        self.log("DataSet Muestra")
        
    def show_info(self):
        print(f"\n📐 Tamaño del DataFrame:\n Filas: {self.df.shape[0]}\n Columnas: {self.df.shape[1]}")
        print("\n📌 Info DataSet\n"+ "-"*40)
        self.df.info() 
        self.log("DataSet Info")
        
    def group_by_dtype(self):
        print("\n📌 COLUMNAS AGRUPADAS POR TIPO DE DATO\n"+ "-"*40)
        groups = self.df.dtypes.groupby(self.df.dtypes).groups
        for dtype, cols in groups.items():
            print(f"\n{dtype}: {list(cols)}")
        self.log("Columnas Agrupadas")
        
    def nulls_and_duplicates(self):
        print("\n📌 COMPROBACIÓN DE VALORES NULOS, DUPLICADOS, UNICOS POR COLUMNA\n" + "-"*40)
        # --- NULOS ---
        nulls = self.df.isnull().sum()
        percent_nulls = (nulls / len(self.df)) * 100
        nulls_df = pd.DataFrame({
            'Nulos': nulls,
            '% Nulos': percent_nulls.round(2).astype(str) + '%'
        })
        
        # --- DUPLICADOS ---
        column_dupes = {}
        for col in self.df.columns:
            dupes = self.df[col].duplicated(keep=False).sum()
            perc = (dupes / len(self.df)) * 100
            column_dupes[col] = (dupes, round(perc, 2))
            
        column_dupes_df = pd.DataFrame.from_dict(column_dupes, orient='index', columns=['Duplicados', '% Duplicados'])

        # --- VALORES ÚNICOS ---
        unique_vals = self.df.nunique()
        unique_percentage = (unique_vals / len(self.df)) * 100
        unique_df = pd.DataFrame({
            'Únicos': unique_vals,
            '% Únicos': unique_percentage.round(2).astype(str) + '%'
        })

        # --- INTEGRACIÓN DE TODA LA INFORMACIÓN ---
        full_df = pd.concat([nulls_df, column_dupes_df['Duplicados'], column_dupes_df['% Duplicados'], unique_df['Únicos'], unique_df['% Únicos']], axis=1)
        full_df = full_df.rename(columns={
            'Nulos': 'Nulos',
            '% Nulos': '% Nulos',
            'Duplicados': 'Duplicados',
            '% Duplicados': '% Duplicados',
            'Únicos': 'Únicos',
            '% Únicos': '% Únicos'
        })

        print("\n🔍 Reporte Completo:")
        print(full_df)

        # --- MENSAJES DE INTERPRETACIÓN ---
        print("\n📋 Resumen de interpretación:")
        
        # Nulos
        null_columns = nulls[nulls > 0]
        if not null_columns.empty:
            print(f"❗ Se encontraron {null_columns.shape[0]} columnas con valores nulos.")
        else:
            print("✅ Ninguna columna tiene valores nulos.")

        # Duplicados
        columnas_con_duplicados = [col for col, (dupes, _) in column_dupes.items() if dupes > 0]
        num_col_duplicadas = len(columnas_con_duplicados)
        total_columnas = len(self.df.columns)
        pct_col_duplicadas = (num_col_duplicadas / total_columnas) * 100

        if num_col_duplicadas > 0:
            print(f"🔁 Se encontraron {num_col_duplicadas} columnas con valores duplicados ({pct_col_duplicadas:.2f}% del total).")
        else:
            print("✅ No se encontraron columnas con duplicados.")
            
        # Valores únicos
        low_variability = unique_vals[unique_vals < 5]  # Columnas con muy pocos valores únicos
        if not low_variability.empty:
            print(f"⚠️ Las siguientes columnas tienen muy baja variabilidad: {', '.join(low_variability.index)}.")
        else:
            print("✅ Todas las columnas tienen suficiente variabilidad.")
        
        self.log("Reporte de Nulos, Duplicados y Cardinalidad generado con éxito.")
        
     
        # Nulos y Duplicados por fila
        print("\n📌 COMPROBACION DE VALORES NULOS Y DUPLICADOS POR FILA\n"+ "-"*40)
        self.df_nulls = self.df[self.df.isnull().any(axis=1)]
        self.df_dups = self.df[self.df.duplicated()]
        print(f"❗ {len(self.df_nulls)} filas con nulos | 🔁 {len(self.df_dups)} filas duplicadas")
        self.log("Valores nulos | duplicados por fila")
        
        
    def detect_booleans(self):
        candidates = [col for col in self.df.columns if self.df[col].nunique() == 2]
        print("\n📌 POSIBLES COLUMNAS BOOLEANAS\n"+ "-"*40)
        print(f"⚠️ Columnas booleanas candidatas: {candidates}")
        self.log("Posibles Booleanas")
   
    # --- FASE 2: Análisis Numérico ---
    def run_fase2_numericas(self):
        self.numeric_stats()
        self.numeric_distributions_separadas()
        self.print_logs()

    def run_fase2_numericas(self):
        self.numeric_stats()
        self.numeric_distributions_separadas()
        self.print_logs()

    def numeric_stats(self):
        num_df = self.df.select_dtypes(include='number')
        stats_df = num_df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T

        # Agregar métricas adicionales
        stats_df['Mediana'] = num_df.median()
        stats_df['Asimetría'] = num_df.skew()
        stats_df['Curtosis'] = num_df.kurtosis()
        stats_df['Moda'] = num_df.mode().iloc[0]
        stats_df['IQR'] = num_df.apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
        stats_df['Rango'] = num_df.max() - num_df.min()
        stats_df['Normalidad (p-valor)'] = num_df.apply(lambda x: shapiro(x.dropna())[1] if len(x.dropna()) < 5000 else np.nan)
        stats_df['Normalidad'] = stats_df['Normalidad (p-valor)'].apply(lambda p: '✅' if p > 0.05 else '❌')

        # Renombrar columnas al español
        stats_df = stats_df.rename(columns={
            'count': 'Recuento',
            'mean': 'Media',
            'std': 'Desviación estándar',
            'min': 'Mínimo',
            '10%': 'Percentil 10',
            '25%': 'Percentil 25',
            '50%': 'Percentil 50',
            '75%': 'Percentil 75',
            '90%': 'Percentil 90',
            'max': 'Máximo'
        })


        # Reordenar columnas para mostrar primero las más relevantes
        columnas_ordenadas = [
            'Recuento', 'Media', 'Mediana', 'Moda', 'Desviación estándar', 'Asimetría', 'Curtosis', 'IQR', 'Rango',
            'Mínimo', 'Percentil 10', 'Percentil 25', 'Percentil 50', 'Percentil 75', 'Percentil 90', 'Máximo',
            'Normalidad (p-valor)', 'Normalidad'
        ]

        stats_df = stats_df[columnas_ordenadas]

        print("\n📈 Estadísticas básicas para variables numericas:")
        print(stats_df.round(2))
        
        return stats_df.round(2)
        
    def numeric_distributions_separadas(self):
        """
        Genera gráficos individuales (distribución + boxplot) para cada columna numérica.
        Cada gráfico se muestra por separado.
        """
        num_df = self.df.select_dtypes(include='number')

        if num_df.empty:
            print("❌ No hay columnas numéricas en el DataFrame.")
            return

        for col in num_df.columns:
            #fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

            col_data = num_df[col].dropna()

            # Crear figura interactiva con subplots
            fig = go.Figure()

            # Histograma con KDE
            fig.add_trace(go.Histogram(
                x=col_data,
                name="Histograma",
                marker_color='skyblue',
                opacity=0.75,
                nbinsx=30
            ))

            # Agregar línea KDE con Plotly Express
            kde_fig = px.density_contour(num_df, x=col)
            kde_trace = kde_fig['data'][0]
            kde_trace.name = 'Densidad (KDE)'
            kde_trace.line.color = 'red'
            kde_trace.line.width = 2
            fig.add_trace(kde_trace)

            # Layout para histograma
            fig.update_layout(
                title=f"Distribución y KDE de '{col}'",
                xaxis_title=col,
                yaxis_title="Frecuencia",
                bargap=0.1,
                template="plotly_white"
            )

            fig.show()

            # Boxplot separado
            fig2 = go.Figure()
            fig2.add_trace(go.Box(
                x=col_data,
                name=col,
                boxpoints="outliers",
                marker_color="lightcoral",
                orientation="h"
            ))

            fig2.update_layout(
                title=f"Boxplot de '{col}'",
                xaxis_title=col,
                template="plotly_white"
            )

            fig2.show()
        self.log("Graficos Numericas generados")    
        
    
    # --- FASE 3: Análisis Categórico ---
    def run_fase3_categoricas(self,max_categorias=15):
        self.categorical_stats()
        self.categorical_distributions_separadas(max_categorias=max_categorias)
        self.print_logs()
    
    def categorical_stats(self):
        """
        Muestra métricas estadísticas clave para columnas categóricas:
        - Recuento
        - Nulos y %
        - Valores únicos
        - Moda
        - Frecuencia de la moda
        - Cardinalidad (% únicos respecto a total)
        - Entropia
        - Alerta cardinalidad
        """
        cat_df = self.df.select_dtypes(include=['object', 'category', 'bool', 'datetime'])
        if cat_df.empty:
            print("❌ No hay columnas categóricas en el DataFrame.")
            return

        total_rows = len(self.df)
        resumen = []

        for col in cat_df.columns:
            datos = self.df[col]
            nulos = datos.isnull().sum()
            n_unique = datos.nunique(dropna=True)
            moda = datos.mode().iloc[0] if not datos.mode().empty else None
            frecuencia_moda = datos.value_counts(dropna=True).iloc[0] if not datos.value_counts().empty else None
            
            # % Moda
            pct_moda = round(frecuencia_moda / datos.count() * 100, 2) if datos.count() > 0 else None

            # Entropía (Shannon base 2) Para saber cuán dispersa está la variable
            freq_dist = datos.value_counts(normalize=True, dropna=True)
            entropia = round(entropy(freq_dist, base=2), 2) if not freq_dist.empty else None

            # Alerta cardinalidad
            cardinalidad_pct = round(n_unique / total_rows * 100, 2)
            if n_unique == 0:
                alerta_card = "Sin datos"
            elif n_unique < 5:
                alerta_card = "Muy baja cardinalidad"
            elif n_unique > total_rows * 0.5:
                alerta_card = "Alta cardinalidad"
            else:
                alerta_card = "Cardinalidad normal"
            resumen.append({
                'Columna': col,
                'Recuento': datos.count(),
                'Nulos': nulos,
                '% Nulos': round(nulos / total_rows * 100, 2),
                'Valores Únicos': n_unique,
                'Cardinalidad %': round(n_unique / total_rows * 100, 2),
                'Moda': moda,
                'Frecuencia Moda': frecuencia_moda,
                '% Moda': pct_moda,
                'Entropía': entropia,
                'Alerta Cardinalidad': alerta_card
            })

        resumen_df = pd.DataFrame(resumen)
        print("\n📊 Métricas estadísticas para variables categóricas:")
        print(resumen_df.to_string(index=False))
        self.log("Tabla de métricas categóricas generada")
        
        return resumen_df
    
    def categorical_distributions_separadas(self, max_categorias=15):
        """
        Muestra gráficos individuales (barra + pastel) para cada columna categórica.
        Limita a las N categorías más frecuentes (por defecto 15).
        """
        
        # Incluimos también fechas como categóricas (convierten a string)
        cat_df = self.df.select_dtypes(include=['object', 'category', 'datetime'])

        if cat_df.empty:
            print("❌ No hay columnas categóricas en el DataFrame.")
            return

        for col in cat_df.columns:
            # Convertimos a string para evitar problemas con fechas
            conteo = cat_df[col].astype(str).value_counts().sort_values(ascending=False).head(max_categorias)

            # Crear DataFrame auxiliar
            data_plot = pd.DataFrame({'Categoría': conteo.index, 'Frecuencia': conteo.values})

            #fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

            # Gráfico de barras interactivo
            fig_bar = px.bar(
                data_plot,
                x='Frecuencia',
                y='Categoría',
                orientation='h',
                title=f"🔹 Frecuencia de categorías en '{col}' (Top {max_categorias})",
                text='Frecuencia',
                color='Categoría',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig_bar.show()

            # Gráfico de pastel interactivo
            fig_pie = px.pie(
                data_plot,
                names='Categoría',
                values='Frecuencia',
                title=f"🔸 Distribución porcentual de '{col}' (Top {max_categorias})",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05]*len(data_plot))
            fig_pie.show()
        self.log("Graficos Categoricas")
    
    # --- FASE 4: Fechas y Booleanos ---
    def run_fase4_fechas_booleans(self):
        self.date_analysis()
        self.boolean_distribution()
        self.print_logs()

    def date_analysis(self):
        print("\n📌 Analisis Fechas\n"+ "-"*40)
        date_cols = [col for col in self.df.columns if np.issubdtype(self.df[col].dtype, np.datetime64)]
        for col in date_cols:
            print(f"\n⏳ {col}: min = {self.df[col].min()}, max = {self.df[col].max()}")
        self.log("Fechas Analizado")
        
    def boolean_distribution(self):
        print("\n📌 Posibles columnas Booleanas\n"+ "-"*40)
        bool_cols = [col for col in self.df.columns if self.df[col].nunique() == 2]
        for col in bool_cols:
            print(f"\n⚠️ {col}:\n{self.df[col].value_counts()}")
            self.log("Booleanas Analizado")
            
    # --- FASE 5: Correlaciones ---
    def run_fase5_correlaciones(self):
        self.correlation_matrix()
        self.highlight_strong_correlations()
        self.print_logs()

    def correlation_matrix(self):
        num_df = self.df.select_dtypes(include='number')
        corr = num_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            colorbar=dict(title='Correlación'),
            zmin=-1, zmax=1,
            hoverongaps=False,
            text=corr.round(2).values,
            hovertemplate='↔ %{x} vs %{y}<br>Correlación: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title='🔍 Matriz de Correlación (interactiva)',
            xaxis_nticks=len(corr.columns),
            yaxis_nticks=len(corr.index),
            width=800,
            height=700
        )

        fig.show()
        self.log("Mapa de correlaciones interactivo generado")
        """num_df = self.df.select_dtypes(include='number')
        corr = num_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Matriz de Correlaciones")
        plt.tight_layout()
        plt.show()
        self.log("Mapa Correlacciones")"""
        
    def highlight_strong_correlations(self, threshold=0.7):
        print("📌 Correlacciones Fuertes\n"+ "-"*40)
        corr = self.df.select_dtypes(include='number').corr()
        strong_corrs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) >= threshold:
                    pair = (corr.index[i], corr.columns[j], corr.iloc[i, j])
                    strong_corrs.append(pair)
        if strong_corrs:
            print("\n🔍 Correlaciones fuertes (>|0.7|):")
            for col1, col2, value in strong_corrs:
                print(f"{col1} ↔ {col2}: {value:.2f}")
        else:
            print("\n✅ No se detectaron correlaciones fuertes por encima del umbral")
        self.log("Correlacciones Fuertes analizado")
        
     # --- FASE 6: Conclusiones ---
    def run_fase6_conclusiones(self):
        #self.auto_conclusions()
        self.auto_conclusions2()
        self.print_logs()
    
    # Devuelve la informacion agrupada por metricas    
    def auto_conclusions(self):
        # NULOS
        print("\n📌 ANÁLISIS DE NULOS\n"+ "-"*40)
        nulls = self.df.isnull().sum()
        for col in nulls[nulls > 0].index:
            pct = nulls[col] / len(self.df) * 100
            print(f"🔸 {col} tiene {pct:.2f}% de valores nulos")
        self.log("Análisis de valores nulos completado")
        
        # DUPLICADOS
        print("\n📌 ANÁLISIS DE DUPLICADOS\n"+ "-"*40)
        dups = self.df.duplicated().sum()
        pct_dups = (dups / len(self.df)) * 100
        print(f"🔸 Se encontraron {dups} filas duplicadas ({pct_dups:.2f}%)")

        print("\n📌 DUPLICADOS POR COLUMNA\n"+ "-"*40)
        for col in self.df.columns:
            col_dups = self.df[col].duplicated().sum()
            pct_col_dups = col_dups / len(self.df) * 100
            if col_dups > 0:
                print(f"🔸 {col}: {col_dups} duplicados ({pct_col_dups:.2f}%)")
        self.log("Análisis de duplicados por columna completado")
        
        # CONSTANTES
        print("\n📌 COLUMNAS CONSTANTES\n"+ "-"*40)
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                print(f"🔸 {col} es constante (1 único valor). Podría eliminarse.")
        self.log("Detección de columnas constantes finalizada")

        # ASIMETRÍA Y OUTLIERS
        print("\n📌 ASIMETRÍA Y OUTLIERS (IQR)\n"+ "-"*40)
        num_df = self.df.select_dtypes(include='number')
        for col in num_df.columns:
            skewness = num_df[col].skew()
            if skewness > 1:
                print(f"🔸 {col} está sesgada positivamente")
            elif skewness < -1:
                print(f"🔸 {col} está sesgada negativamente")

            Q1 = num_df[col].quantile(0.25)
            Q3 = num_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((num_df[col] < (Q1 - 1.5 * IQR)) | (num_df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                print(f"⚠️ {col} tiene {outliers} posibles outliers detectados por IQR")
        self.log("Asimetría y outliers analizados")

        # DESVIACIÓN ESTÁNDAR Y CURTOSIS
        print("\n📌 DESVIACIÓN Y CURTOSIS\n"+ "-"*40)
        for col in num_df.columns:
            std = num_df[col].std()
            mean = num_df[col].mean()
            if std > mean * 2:
                print(f"📈 {col} tiene una desviación estándar alta ({std:.2f})")
            kurt = num_df[col].kurtosis()
            if kurt > 3:
                print(f"🔺 {col} tiene curtosis alta ({kurt:.2f}) → colas pesadas")
            elif kurt < 1:
                print(f"🔻 {col} tiene curtosis baja ({kurt:.2f}) → distribución plana")
        self.log("Análisis de desviación y curtosis completado")
        
        # CARDINALIDAD
        print("\n📌 CARDINALIDAD DE CATEGÓRICAS\n"+ "-"*40)
        cat_df = self.df.select_dtypes(include='object')
        for col in cat_df.columns:
            n_unique = cat_df[col].nunique()
            if n_unique > 50:
                print(f"🔸 {col} tiene alta cardinalidad ({n_unique} valores únicos)")
        self.log("Análisis de cardinalidad completado")

        # RESUMEN FINAL
        self.log("Conclusiones automáticas generadas")
    
    # Devuelve la informacion agrupada por columnas    
    def auto_conclusions2(self):
        print("\n📌 CONCLUSIONES DETALLADAS POR COLUMNA\n" + "-"*40)
        for col in self.df.columns:
            print(f"\n🔍 COLUMNA: {col}")
            col_data = self.df[col].dropna()
            total = len(self.df)
            dtype = self.df[col].dtype
            print(f"- 🏷️Tipo de dato: {dtype}")

            # Nulos
            nulls = self.df[col].isnull().sum()
            pct_nulls = nulls / total * 100
            if nulls > 0:
                print(f"- ❗ Nulos: {nulls} ({pct_nulls:.2f}%) → Considera imputación o eliminación")

            # Duplicados
            dups = self.df[col].duplicated().sum()
            pct_dups = dups / total * 100
            if dups > 0:
                print(f"- 🔁Duplicados: {dups} ({pct_dups:.2f}%) → Revisa posibles redundancias")

            # Constantes
            nunique = self.df[col].nunique()
            if nunique == 1:
                print("- ❌Valor constante → Puede eliminarse del análisis")

            if np.issubdtype(dtype, np.number):
                # Asimetría
                skew = col_data.skew()
                if skew > 1:
                    skew_text = "Sesgo positivo"
                elif skew < -1:
                    skew_text = "Sesgo negativo"
                else:
                    skew_text = "Distribución simétrica"
                print(f"- 🔸Asimetría: {skew:.2f} → {skew_text}")

                # Outliers
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    print(f"- ⚠️Outliers detectados: {outliers} → Posibles valores extremos")

                # Desviación estándar y media
                std = col_data.std()
                mean = col_data.mean()
                if std > mean * 2:
                    print(f"- 📈Alta desviación estándar ({std:.2f}) respecto a la media ({mean:.2f})")

                # Curtosis
                kurt = col_data.kurtosis()
                if kurt > 3:
                    print(f"- 🔺Curtosis alta ({kurt:.2f}) → Colas pesadas")
                elif kurt < 1:
                    print(f"- 🔻Curtosis baja ({kurt:.2f}) → Distribución plana")

            elif dtype == 'object':
                # Cardinalidad
                if nunique > 50:
                    print(f"- 💡Alta cardinalidad: {nunique} valores únicos → Considera agrupar")

            # Conclusión general
            print("- 📋Conclusión: ", end="")
            if nunique == 1:
                print("Columna constante, poco útil")
            elif pct_nulls > 50:
                print("Demasiados nulos → Considerar eliminación")
            elif np.issubdtype(dtype, np.number) and outliers > total * 0.2:
                print("Muchos outliers → Requiere tratamiento")
            elif np.issubdtype(dtype, np.number) and std > mean * 2:
                print("Alta variabilidad → Requiere normalización o segmentación")
            elif dtype == 'object' and nunique > 50:
                print("Categoría con alta cardinalidad → Considerar reducción")
            else:
                print("Columna válida para análisis")

        self.log("Conclusiones detalladas generadas")
        
    # --- FASE Test: Análisis Univariante con Tests y Visualización ---        
    def run_fase_test_univariante(self):
        self.show_column_number()
        self.test_univariante()  
        self.log("Análisis univariante completado")  
    
    def show_column_number(self):
        # Selección de columnas numéricas
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not numeric_cols:
            print("⚠️ No hay columnas numéricas continuas en el DataFrame.")
            return

        print("🔢 Selecciona la variable numérica que deseas analizar:\n")
        for idx, col in enumerate(numeric_cols):
            print(f"{idx}: {col}")

        try:
            sel = int(input("\nIntroduce el número de la columna: "))
            self.columna_univariante_seleccionada = numeric_cols[sel]
        except (IndexError, ValueError):
            print("❌ Selección inválida.")
            self.columna_univariante_seleccionada = None
        
    def test_univariante(self):
        column = getattr(self, 'columna_univariante_seleccionada', None)

        if not column:
            print("⚠️ No se ha seleccionado ninguna columna válida.")
            return

        series = self.df[column].dropna()
        total_n = self.df[column].shape[0]
        valid_n = series.shape[0]
        null_n = total_n - valid_n

        # Métricas estadísticas
        media = series.mean()
        mediana = series.median()
        moda = series.mode().iloc[0] if not series.mode().empty else np.nan
        std = series.std()
        varianza = series.var()
        skewness = series.skew()
        kurtosis = series.kurtosis()
        minimo = series.min()
        maximo = series.max()
        rango = maximo - minimo
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

        # Test de Shapiro-Wilk
        if 3 <= len(series) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(series)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        # Interpretaciones
        skew_text = "sesgo positivo (cola derecha)" if skewness > 0.5 else "sesgo negativo (cola izquierda)" if skewness < -0.5 else "ligera asimetría"
        kurt_text = "leptocúrtica (picuda, colas largas)" if kurtosis > 3 else "platicúrtica (aplanada, colas cortas)" if kurtosis < 3 else "mesocúrtica"
        normal_text = "✅ Distribución aproximadamente normal" if shapiro_p > 0.05 else "❌ No distribución normal"

        # Output textual estructurado
        print(f"\n🔍 Análisis univariante: columna '{column}'\n")
        print("🧮 Resumen de métricas")
        print(f"- N (valores válidos): {valid_n} → Muestra {'amplia' if valid_n > 100 else 'pequeña'}")
        print(f"- N (nulos): {null_n} → {'Sin' if null_n == 0 else f'{round(100*null_n/total_n,1)}% de'} datos faltantes")
        print(f"- Media: {media:.2f}")
        print(f"- Mediana: {mediana:.2f}")
        print(f"- Moda: {moda:.2f}")
        print(f"- Desviación estándar: {std:.2f}")
        print(f"- Varianza: {varianza:.2f}")
        print(f"- Skewness (Asimetría): {skewness:.2f} 🚨 → {skew_text}")
        print(f"- Kurtosis: {kurtosis:.2f} 🚨 → {kurt_text}")
        print(f"- Mínimo: {minimo:.2f}")
        print(f"- Máximo: {maximo:.2f}")
        print(f"- Rango: {rango:.2f}")
        print(f"- IQR: {iqr:.2f}")
        print(f"- Outliers detectados: {outliers}")

        print("\n🔬📈 Shapiro-Wilk – Test de normalidad")
        if not np.isnan(shapiro_stat):
            print(f"- Estadístico = {shapiro_stat:.3f}")
            print(f"- p = {shapiro_p:.5f} → {normal_text}")
        else:
            print("⚠️ No se pudo calcular el test de Shapiro (requiere entre 3 y 5000 datos).")

        # Conclusión
        print("\n💬 Conclusión:")
        print(f"La variable '{column}' {normal_text.lower()}, muestra un {skew_text} y una forma {kurt_text}.")
        if outliers > 0:
            print(f"Hay presencia de outliers detectados por IQR (n={outliers}).")

        # Interpretación integral
        print("\n🧠 Interpretación integral")
        print(f"La variable '{column}' presenta una distribución {'asimétrica' if abs(skewness) > 0.5 else 'casi simétrica'},")
        print(f"lo que implica que los valores se concentran más en uno de los extremos del rango.")
        print(f"Además, su {kurt_text} indica que la forma de la distribución difiere de la normal en términos de concentración y colas.")
        print("El test de normalidad confirma que no se ajusta a una distribución normal.\n")

        # Por qué es importante
        print("📌 ¿Por qué es importante?")
        print("El comportamiento de esta variable afecta directamente a la validez de los tests estadísticos que se puedan aplicar.")
        print("Si no es normal y presenta asimetría o curtosis alta, conviene evitar tests paramétricos que requieran normalidad.\n")

        # Recomendaciones
        print("✅ Recomendaciones")
        if shapiro_p < 0.05:
            print("- Evitar usar tests paramétricos directamente con esta variable.")
            print("- Aplicar una transformación logarítmica o Box-Cox si se requiere normalidad.")
        else:
            print("- Puede utilizarse en tests paramétricos si otros supuestos se cumplen.")

        if outliers > 0:
            print("- Revisar y tratar los outliers si distorsionan el análisis.")
        print("- Visualizar la variable con histogramas, boxplots y Q-Q plots para una evaluación visual complementaria.")

        # Crear histograma y KDE
        hist_data = series
        kde = gaussian_kde(hist_data)
        x_vals = np.linspace(hist_data.min(), hist_data.max(), 500)
        kde_vals = kde(x_vals)

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=hist_data,
            nbinsx=30,
            name='Histograma',
            marker_color='#1f77b4',
            opacity=0.7
        ))
        fig1.add_trace(go.Scatter(
            x=x_vals,
            y=kde_vals * len(hist_data) * (x_vals[1] - x_vals[0]),  # Escalado
            mode='lines',
            name='KDE',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig1.update_layout(
            title=f'📊 Histograma con KDE – {column}',
            bargap=0.1,
            yaxis_title='Frecuencia',
            legend=dict(x=0.8, y=0.95)
        )
        fig1.show()

        fig2 = go.Figure()
        fig2.add_trace(go.Box(x=series, name='Boxplot', boxpoints='outliers', marker_color='lightcoral'))
        fig2.update_layout(title=f'📦 Boxplot interactivo – {column}')
        fig2.show()

        # Q-Q Plot con statsmodels (estático)
        sm.qqplot(series, line='s')
        plt.title(f"Q-Q Plot – {column}")
        plt.show()

        # Logging si usas sistema
        if hasattr(self, "log"):
            self.log(f"Fase 7 completada: análisis univariante de '{column}'")
        
        
    # --- FASE Test: Análisis Bivariante con Tests y Visualización ---
    def run_fase_test_bivariante(self):
        self.show_column_indices()
        #self.suggest_column_pairs()
        self.test_bivariante()
        self.log("Análisis bivariante completado")

    def show_column_indices(self):
        print("\n📜 ÍNDICES DE COLUMNAS DISPONIBLES")
        for i, col in enumerate(self.df.columns):
            print(f"[{i}] {col} — tipo: {self.df[col].dtype}") 
    """
    def suggest_column_pairs(self):
        print("\n📌 SUGERENCIAS DE VARIABLES BIVARIANTES\n" + "-"*40)
        num_cols = self.df.select_dtypes(include='number').columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        print("\nNuméricas vs Numéricas:")
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                print(f"🔹 {num_cols[i]} ↔ {num_cols[j]}")

        print("\nCategóricas vs Numéricas:")
        for cat in cat_cols:
            for num in num_cols:
                print(f"🔸 {cat} ↔ {num}")

        print("\nCategóricas vs Categóricas:")
        for i in range(len(cat_cols)):
            for j in range(i+1, len(cat_cols)):
                print(f"🔻 {cat_cols[i]} ↔ {cat_cols[j]}")
        self.log("Pares sugeridos para análisis bivariante")
    """
    def test_bivariante(self):
        cols = self.df.columns.tolist()
        while True:
            col_x = input("\n📥 Ingrese la PRIMERA columna (número o nombre, o 'salir'): ")
            if col_x.lower() == 'salir':
                break

            col_y = input("📥 Ingrese la SEGUNDA columna (número o nombre): ")

            # Convertir a nombres si es índice
            try:
                if col_x.isdigit():
                    col_x = cols[int(col_x)]
                if col_y.isdigit():
                    col_y = cols[int(col_y)]
            except (IndexError, ValueError):
                print("❌ Índice fuera de rango o no válido.")
                continue

            if col_x in self.df.columns and col_y in self.df.columns:
                self.run_test_for_pair(col_x, col_y)
                self.visualize_bivariate(col_x, col_y)
            else:
                print("❌ Columnas no válidas.")

    def run_test_for_pair(self, col_x, col_y):
        tipo_x = self.df[col_x].dtype
        tipo_y = self.df[col_y].dtype

        print(f"\n⚙️ Analizando relación entre '{col_x}' y '{col_y}':")

        if np.issubdtype(tipo_x, np.number) and np.issubdtype(tipo_y, np.number):
            x = self.df[col_x].dropna()
            y = self.df[col_y].dropna()
            if len(x) != len(y):
                df_temp = pd.concat([x, y], axis=1).dropna()
                x = df_temp[col_x]
                y = df_temp[col_y]
            r, p = pearsonr(x, y)
            print(f"→ Test de correlación de Pearson:")
            print(f"- Coeficiente r = {r:.3f}, p = {p:.4f}")

            # Interpretación del coeficiente
            if abs(r) < 0.1:
                interpretacion = "muy débil o inexistente"
            elif abs(r) < 0.3:
                interpretacion = "débil"
            elif abs(r) < 0.5:
                interpretacion = "moderada"
            elif abs(r) < 0.7:
                interpretacion = "fuerte"
            else:
                interpretacion = "muy fuerte"

            print(f"📊 Relación {interpretacion} ({'positiva' if r > 0 else 'negativa' if r < 0 else 'nula'})")
            if p < 0.05:
                print("✅ Correlación estadísticamente significativa (p < 0.05)")
            else:
                print("❌ No se detecta correlación significativa")

        elif tipo_x in ['object', 'category', 'bool'] and np.issubdtype(tipo_y, np.number):
            grupos = self.df.groupby(col_x)[col_y].apply(list)

            if len(grupos) == 2:
                group_names = grupos.index.tolist()
                group1, group2 = grupos.iloc[0], grupos.iloc[1]

                # Test de Levene
                stat_levene, p_levene = levene(group1, group2)
                print(f"→ Test de Levene (igualdad de varianzas): p = {p_levene:.4f}")

                if p_levene > 0.05:
                    # Varianzas iguales → t-test estándar
                    stat, p = ttest_ind(group1, group2, equal_var=True)
                    print(f"✅ Varianzas iguales → t-test estándar: p = {p:.4f}")
                else:
                    # Varianzas distintas → Welch
                    stat, p = ttest_ind(group1, group2, equal_var=False)
                    print(f"⚠️ Varianzas diferentes → Welch's t-test: p = {p:.4f}")

                if p < 0.05:
                    print("🔍 Diferencia significativa entre grupos (p < 0.05)")
                else:
                    print("✅ No se detecta diferencia significativa (p ≥ 0.05)")

            elif len(grupos) > 2:
                self.run_anova_or_welch(col_x, col_y)
            else:
                print("❌ No hay suficientes grupos para análisis.")

        elif tipo_x in ['object', 'category', 'bool'] and tipo_y in ['object', 'category', 'bool']:
            tabla = pd.crosstab(self.df[col_x], self.df[col_y])
            chi2, p, dof, expected = chi2_contingency(tabla)
            print(f"→ Test de Chi-cuadrado entre '{col_x}' y '{col_y}'")
            print(f"- Chi² = {chi2:.2f}, p = {p:.4f}, grados de libertad = {dof}")

            if expected.min() < 5:
                print("⚠️ Advertencia: Algunas frecuencias esperadas son < 5. El resultado podría no ser fiable.")

            if p < 0.05:
                print("✅ Asociación estadísticamente significativa (p < 0.05)")
            else:
                print("❌ No se detecta asociación significativa")

            # Tamaño del efecto: Cramér's V
            n = tabla.sum().sum()
            phi2 = chi2 / n
            r, k = tabla.shape
            cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
            print(f"📏 Tamaño del efecto (Cramér's V): {cramers_v:.3f}")

        else:
            print("❌ Combinación de tipos no compatible")
     
     # Análisis ANOVA para {num_col} según {cat_col} si len(grupos) > 2     
    def run_anova_or_welch(self, cat_col, num_col):
        print(f"\n📊 Análisis ANOVA para {num_col} según {cat_col}")

        grupos = self.df.groupby(cat_col)[num_col].apply(list)

        # Test de Levene para igualdad de varianzas
        stat_levene, p_levene = levene(*grupos)
        print(f"📏 Test de Levene: p = {p_levene:.4f} → {'✅ Varianzas iguales' if p_levene > 0.05 else '❌ Varianzas diferentes'}")

        if p_levene > 0.05:
            # Varianzas iguales → ANOVA clásico
            stat_anova, p_anova = f_oneway(*grupos)
            print(f"🔬 ANOVA clásico: p = {p_anova:.4f}")
            interpretacion = "✅ Hay diferencias significativas entre grupos" if p_anova < 0.05 else "❌ No hay diferencias significativas"
        else:
            # Varianzas diferentes → Welch ANOVA
            df_test = self.df[[cat_col, num_col]].dropna()
            df_test[cat_col] = df_test[cat_col].astype(str)  # statsmodels requiere categorías tipo str
            res = anova_oneway(df_test[num_col], groups=df_test[cat_col], use_var='unequal')
            p_anova = res.pvalue
            print(f"🔬 Welch ANOVA: p = {p_anova:.4f}")
            interpretacion = "✅ Hay diferencias significativas entre grupos (Welch)" if p_anova < 0.05 else "❌ No hay diferencias significativas (Welch)"

        print(f"🧠 Interpretación: {interpretacion}")
        
        
    def visualize_bivariate(self, col_x, col_y):
        print(f"\n📊 Gráfico para {col_x} ↔ {col_y}")
        if np.issubdtype(self.df[col_x].dtype, np.number) and np.issubdtype(self.df[col_y].dtype, np.number):
            sns.scatterplot(x=self.df[col_x], y=self.df[col_y])
            plt.title(f"Relación entre {col_x} y {col_y}")
        elif self.df[col_x].dtype in ['object', 'category', 'bool'] and np.issubdtype(self.df[col_y].dtype, np.number):
            sns.boxplot(x=self.df[col_x], y=self.df[col_y])
            plt.title(f"{col_y} por {col_x}")
        elif self.df[col_x].dtype in ['object', 'category', 'bool'] and self.df[col_y].dtype in ['object', 'category', 'bool']:
            tabla = pd.crosstab(self.df[col_x], self.df[col_y])
            tabla.plot(kind='bar', stacked=True)
            plt.title(f"{col_x} vs {col_y}")
        plt.tight_layout()
        plt.show()
        self.log("Visualización bivariante generada")
        
        
        
        """
Siguientes mejoras recomendadas:
0. poder analizar dos columnas ( por ejemplo -> ventas por marca)

2. Fase 8 (Bivariante Avanzado y A/B Testing)
Mejorar grafico

Mostrar resultado del test de Levene para varianzas.





Añadir una conclusión interpretativa automática después de cada test.



3. Exportar informe (opcional)
Exportar a .txt o .pdf el log y/o gráficos.
        """