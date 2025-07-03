# eda_helper.py

# EDAHelper: Exploración de Datos Automatizada y Modular
import os
from IPython.display import display, HTML

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

#Archivo auxiliar de Test estadisticos
from Class_Test import StatisticalTests

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
        self.print_logs()  
    
    def show_column_number(self):
        # Selección de columnas numéricas
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not numeric_cols:
            print("⚠️ No hay columnas numéricas continuas en el DataFrame.")
            return

        print("📜 ÍNDICES DE COLUMNAS DISPONIBLES:")
        for idx, col in enumerate(numeric_cols):
            print(f"{idx}: {col}")
        print("- -"*30)
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

            # --- Tests de normalidad adaptados según tamaño de muestra ---
        if 3 <= len(series) <= 5000:
            # Shapiro-Wilk para muestras pequeñas a medias
            res_shapiro = StatisticalTests.shapiro_wilk_test(series)
            normal_text = "✅ Distribución aproximadamente normal" if res_shapiro["result"] == "No rechazar H0" else "❌ No distribución normal"
            shapiro_stat = res_shapiro["statistic"]
            shapiro_p = res_shapiro["p_value"]

        elif len(series) > 5000:
            # Para muestras grandes, usar KS y Anderson-Darling combinados
            res_ks = StatisticalTests.kolmogorov_smirnov_test(series)
            res_ad = StatisticalTests.anderson_darling_test(series)

            normal_ks = res_ks["result"] == "No rechazar H0"
            normal_ad = res_ad["result"] == "No rechazar H0"

            if normal_ks and normal_ad:
                normal_text = "✅ Distribución aproximadamente normal según KS y AD"
            else:
                normal_text = "❌ No distribución normal según KS y/o AD"

            shapiro_stat = None
            shapiro_p = None

        else:
            normal_text = "⚠️ No se pudo calcular el test de normalidad (menos de 3 datos)"
            shapiro_stat = None
            shapiro_p = None

            # --- Tests adicionales univariantes ---
        # Test de simetría (skewness)
        res_skew = StatisticalTests.skewness_test(series)
        # Test de curtosis (kurtosis)
        res_kurt = StatisticalTests.kurtosis_test(series)
        # Test de normalidad combinada D'Agostino K2
        res_dago = StatisticalTests.dagostino_test(series)
        # Test Jarque-Bera
        res_jb = StatisticalTests.jarque_bera_test(series)

        # --- Interpretación de skewness y kurtosis ---
        skew_text = "sesgo positivo (cola derecha)" if skewness > 0.5 else "sesgo negativo (cola izquierda)" if skewness < -0.5 else "ligera asimetría"
        kurt_text = "leptocúrtica (picuda, colas largas)" if kurtosis > 3 else "platicúrtica (aplanada, colas cortas)" if kurtosis < 3 else "mesocúrtica"

        # --- Salida textual estructurada ---
        print(f"\n🔍 Análisis univariante: columna '{column}'\n" + "-"*60)
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

        # Tests de normalidad
        print("\n🔬📈 Test de normalidad")
        if shapiro_stat is not None:
            print(f"Shapiro-Wilk:")
            print(f"- Estadístico = {shapiro_stat:.3f}")
            print(f"- p = {shapiro_p:.5f} → {normal_text}")
        elif len(series) > 5000:
            print(f"Kolmogorov-Smirnov:")
            print(f"- Estadístico = {res_ks['statistic']:.3f}")
            print(f"- p = {res_ks['p_value']:.5f}")
            print(f"Anderson-Darling:")
            print(f"- Estadístico = {res_ad['statistic']:.3f}")
            print(f"→ {normal_text}")
        else:
            print(normal_text)

        # Tests adicionales (skewness, kurtosis, D'Agostino, Jarque-Bera)
        print("\n🔬📉 Tests adicionales")
        print(f"- Test simetría: estadístico={res_skew['statistic']:.3f}, p={res_skew['p_value']:.5f} → {res_skew['result']}")
        print(f"- Test curtosis: estadístico={res_kurt['statistic']:.3f}, p={res_kurt['p_value']:.5f} → {res_kurt['result']}")
        print(f"- Test D'Agostino K²: estadístico={res_dago['statistic']:.3f}, p={res_dago['p_value']:.5f} → {res_dago['result']}")
        print(f"- Test Jarque-Bera: estadístico={res_jb['statistic']:.3f}, p={res_jb['p_value']:.5f} → {res_jb['result']}")

        # --- Conclusión ---
        print("\n💬 Conclusión:")
        print(f"La variable '{column}' {normal_text.lower()}, muestra un {skew_text} y una forma {kurt_text}.")
        if outliers > 0:
            print(f"Hay presencia de outliers detectados por IQR (n={outliers}).")

        # --- Interpretación integral ---
        print("\n🧠 Interpretación integral")
        print(f"La variable '{column}' presenta una distribución {'asimétrica' if abs(skewness) > 0.5 else 'casi simétrica'},")
        print(f"lo que implica que los valores se concentran más en uno de los extremos del rango.")
        print(f"Además, su {kurt_text} indica que la forma de la distribución difiere de la normal en términos de concentración y colas.")
        print(f"El test de normalidad confirma que {normal_text.lower()}.\n")

        # --- Por qué es importante ---
        print("📌 ¿Por qué es importante?")
        print("El comportamiento de esta variable afecta directamente a la validez de los tests estadísticos que se puedan aplicar.")
        print("Si no es normal y presenta asimetría o curtosis alta, conviene evitar tests paramétricos que requieran normalidad.\n")

        # --- Recomendaciones ---
        print("✅ Recomendaciones")
        if (shapiro_p is not None and shapiro_p < 0.05) or (len(series) > 5000 and normal_text.startswith("❌")):
            print("- Evitar usar tests paramétricos directamente con esta variable.")
            print("- Aplicar una transformación logarítmica o Box-Cox si se requiere normalidad.")
        else:
            print("- Puede utilizarse en tests paramétricos si otros supuestos se cumplen.")

        if outliers > 0:
            print("- Revisar y tratar los outliers si distorsionan el análisis.")
        print("- Visualizar la variable con histogramas, boxplots y Q-Q plots para una evaluación visual complementaria.")

        # --- Visualizaciones ---
        kde = gaussian_kde(series)
        x_vals = np.linspace(series.min(), series.max(), 500)
        kde_vals = kde(x_vals)

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=series, nbinsx=30, name='Histograma', marker_color='#1f77b4', opacity=0.7))
        fig1.add_trace(go.Scatter(x=x_vals, y=kde_vals * len(series) * (x_vals[1] - x_vals[0]), mode='lines', name='KDE', line=dict(color='#ff7f0e', width=2)))
        fig1.update_layout(title=f'📊 Histograma con KDE – {column}', bargap=0.1, yaxis_title='Frecuencia', legend=dict(x=0.8, y=0.95))
        fig1.show()

        fig2 = go.Figure()
        fig2.add_trace(go.Box(x=series, name='Boxplot', boxpoints='outliers', marker_color='lightcoral'))
        fig2.update_layout(title=f'📦 Boxplot interactivo – {column}')
        fig2.show()

        sm.qqplot(series, line='s')
        plt.title(f"Q-Q Plot – {column}")
        plt.show()

        # --- Logging ---
        if hasattr(self, "log"):
            self.log(f"Completado Análisis Univariante de '{column}'") 
    
    
    
    def run_fase_test_bivariante(self):
        col_x, col_y = self.show_column_all()
        if col_x is not None and col_y is not None:
            self.test_bivariante(col_x, col_y)
        self.print_logs()    
    def show_column_all(self):
        print("\n📜 Selección de columnas para análisis bivariante")
        cols = self.df.columns.tolist()
        for idx, col in enumerate(cols):
            print(f"{idx}: {col}")
        print("- -" * 30)

        try:
            sel_x = int(input("Introduce número de la PRIMERA columna: "))
            sel_y = int(input("Introduce número de la SEGUNDA columna: "))
            return cols[sel_x], cols[sel_y]
        except (IndexError, ValueError):
            print("❌ Selección inválida.")
            return None, None
    
    def test_bivariante(self, col_x, col_y):
        tipo_x = self.df[col_x].dtype
        tipo_y = self.df[col_y].dtype

        print(f"\n🔍 Análisis bivariante: '{col_x}' vs '{col_y}'")
        print("-" * 60)

        if np.issubdtype(tipo_x, np.number) and np.issubdtype(tipo_y, np.number):
            self._test_num_vs_num(col_x, col_y)
        elif tipo_x in ['object', 'category', 'bool'] and np.issubdtype(tipo_y, np.number):
            self._test_cat_vs_num(cat_col=col_x, num_col=col_y)
        elif np.issubdtype(tipo_x, np.number) and tipo_y in ['object', 'category', 'bool']:
            self._test_cat_vs_num(cat_col=col_y, num_col=col_x)
        elif tipo_x in ['object', 'category', 'bool'] and tipo_y in ['object', 'category', 'bool']:
            self._test_cat_vs_cat(col_x, col_y)
        else:
            print("❌ Combinación de tipos no soportada para análisis bivariante.")

    def _test_num_vs_num(self, col_x, col_y):
        x = self.df[col_x].dropna()
        y = self.df[col_y].dropna()
        df_temp = pd.concat([x, y], axis=1).dropna()
        x, y = df_temp[col_x], df_temp[col_y]

        print("\n🧮 Resumen de datos")
        print(f"- Valores válidos: {len(x)}")
        if len(x) < 10:
            print("⚠️ Muestra muy pequeña: resultados poco fiables")

        print("\n🔬 Test de correlación de Pearson")
        res = StatisticalTests.pearson_correlation(x, y)
        r = res['statistic']
        p = res['p_value']
        print(f"- Coeficiente r = {r:.3f}, p = {p:.4f}")

        if abs(r) < 0.1:
            intensidad = "muy débil o inexistente"
        elif abs(r) < 0.3:
            intensidad = "débil"
        elif abs(r) < 0.5:
            intensidad = "moderada"
        elif abs(r) < 0.7:
            intensidad = "fuerte"
        else:
            intensidad = "muy fuerte"
        signo = "positiva" if r > 0 else "negativa" if r < 0 else "nula"
        print(f"📏 Relación {intensidad} ({signo})")

        print("\n💬 Conclusión")
        print("✅ Correlación significativa" if p < 0.05 else "❌ No significativa")

        print("\n🧠 Interpretación integral")
        print("La relación entre ambas variables se evalúa mediante correlación lineal de Pearson.")
        print("Recuerda que no implica causalidad. Si hay asimetría o outliers, la correlación puede distorsionarse.")

        print("\n📌 ¿Por qué es importante?")
        print("Entender si dos variables numéricas están relacionadas puede apoyar decisiones y modelos.")

        print("\n📊 Visualización Scatter Plot interactivo")
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#1f77b4')))
        fig.update_layout(title=f"Scatter plot: {col_x} vs {col_y}", xaxis_title=col_x, yaxis_title=col_y)
        fig.show()

        print("\n📋 Resumen del test aplicado:")
        resumen_df = pd.DataFrame([{
            "Test": res['test_name'],
            "p-valor": round(res['p_value'], 4),
            "Significativo": "✅" if res['p_value'] < 0.05 else "❌",
            "Coef. Pearson r": round(res['statistic'], 3),
            "Recomendación": res['recommendation']
        }])
        display(HTML(resumen_df.to_html(index=False)))

        if hasattr(self, "log"):
            self.log(f"Completado Análisis Bivariante de '{col_x}' y '{col_y}'")
    
    def _test_cat_vs_num(self, cat_col, num_col):
        grupos = self.df.groupby(cat_col)[num_col].apply(list)
        grupos_validos = [g for g in grupos if len(g) >= 2]
        grupos_filtrados = grupos[grupos.apply(lambda g: len(g) >= 2)]

        if len(grupos_validos) < 2:
            print("❌ No hay suficientes grupos válidos (mínimo 2 con ≥2 datos)")
            return

        print(f"\n🧮 Grupos válidos: {len(grupos_validos)}")
        excluidos = [g for g in grupos.index if len(grupos[g]) < 2]
        if excluidos:
            print(f"⚠️ Grupos excluidos por tamaño insuficiente: {excluidos}")

        

        # Test de Levene
        print(f"\n🔬 Test de Levene para igualdad de varianzas")
        res_levene = StatisticalTests.levene_test(*grupos_validos)
        p_levene = res_levene['p_value']
        print(f"- p = {p_levene:.4f} → {res_levene['conclusion']}")

        # Elección del test
        if len(grupos_validos) == 2:
            g1, g2 = grupos_validos[0], grupos_validos[1]
            if p_levene > 0.05:
                res_test = StatisticalTests.ttest_independent(g1, g2, equal_var=True)
            else:
                res_test = StatisticalTests.ttest_independent(g1, g2, equal_var=False)

            # Tamaño del efecto: Cohen's d
            d = (np.mean(g1) - np.mean(g2)) / np.sqrt((np.std(g1, ddof=1)**2 + np.std(g2, ddof=1)**2) / 2)
            print(f"→ {res_test['test_name']}: p = {res_test['p_value']:.4f}")
            print("🔍 Conclusión:", res_test['conclusion'])
            print(f"🎯 Tamaño del efecto (Cohen's d): {d:.3f}")

            # Si p > 0.05, test no paramétrico
            if res_test['p_value'] > 0.05:
                print("\n🧪 Validación con test no paramétrico (Mann-Whitney)")
                res_nonparam = StatisticalTests.mann_whitney_u_test(g1, g2)
                print(f"→ {res_nonparam['test_name']}: p = {res_nonparam['p_value']:.4f}")
                print("🔍 Conclusión:", res_nonparam['conclusion'])

        else:
            if p_levene > 0.05:
                res_test = StatisticalTests.anova_classic(grupos_validos)
                eta_sq = res_test['statistic'] * (len(self.df) - 1) / (res_test['statistic'] * (len(self.df) - 1) + len(grupos_validos))
            else:
                res_test = StatisticalTests.welch_anova(grupos_validos)
                eta_sq = None  # Welch no da eta² directamente

            print(f"→ {res_test['test_name']}: p = {res_test['p_value']:.4f}")
            print("🔍 Conclusión:", res_test['conclusion'])
            if eta_sq is not None:
                print(f"🎯 Tamaño del efecto (eta²): {eta_sq:.3f}")

            # Si p > 0.05, test no paramétrico
            if res_test['p_value'] > 0.05:
                print("\n🧪 Validación con test no paramétrico (Kruskal-Wallis)")
                res_nonparam = StatisticalTests.kruskal_wallis_test(grupos_validos)
                print(f"→ {res_nonparam['test_name']}: p = {res_nonparam['p_value']:.4f}")
                print("🔍 Conclusión:", res_nonparam['conclusion'])

        print("\n🧠 Interpretación integral")
        print("Se evaluaron diferencias entre grupos usando tests adecuados según homocedasticidad.")
        print("Los grupos con menos de 2 datos fueron excluidos del análisis.")

        print("\n📌 ¿Por qué es importante?")
        print("Comprobar si los valores de una variable numérica difieren según categorías ayuda a descubrir patrones.")

        
            
        print("\n📊 Visualización Boxplot + Stripplot interactivos")
        df_viz = self.df[[cat_col, num_col]].dropna()
        fig = px.strip(df_viz, x=cat_col, y=num_col, stripmode='overlay', color=cat_col,
                    title=f"Distribución de {num_col} por grupos de {cat_col}",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(jitter=0.35, marker=dict(opacity=0.5, size=5), selector=dict(type='scatter'))
        fig.update_layout(yaxis_title=num_col, xaxis_title=cat_col)
        fig.show()

        print("\n📋 Resumen del test aplicado:")
        resumen_data = {
            "Test": res_test['test_name'],
            "p-valor": round(res_test['p_value'], 4),
            "Significativo": "✅" if res_test['p_value'] < 0.05 else "❌",
            "Tamaño del efecto": round(d if len(grupos_validos) == 2 else (eta_sq if eta_sq else 0), 3),
            "Recomendación": res_test['recommendation']
        }
        if 'res_nonparam' in locals():
            resumen_data["Test alternativo"] = res_nonparam['test_name']
            resumen_data["p-alt"] = round(res_nonparam['p_value'], 4)
        resumen_df = pd.DataFrame([resumen_data])
        display(HTML(resumen_df.to_html(index=False)))

        
        # Outliers por grupo
        print("\n🔎 Outliers por grupo (basado en IQR):")
        for grupo, datos in grupos_filtrados.items():
            q1 = np.percentile(datos, 25)
            q3 = np.percentile(datos, 75)
            iqr = q3 - q1
            outliers = [x for x in datos if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]
            print(f"- {grupo}: {len(outliers)} outliers")
        if hasattr(self, "log"):
            self.log(f"Completado Análisis Bivariante de '{cat_col}' y '{num_col}'")
    
    
    def _test_cat_vs_cat(self, col_x, col_y):
        tabla = pd.crosstab(self.df[col_x], self.df[col_y])
        print(f"\n🧮 Tabla de contingencia de '{col_x}' vs '{col_y}'")
        print(f"- Tamaño total: {tabla.values.sum()}")

        chi2_res = StatisticalTests.chi2_test(tabla)
        p = chi2_res['p_value']
        stat = chi2_res['statistic']

        print("\n🔬 Test Chi-cuadrado")
        print(f"- Chi² = {stat:.2f}, p = {p:.4f}")
        print("🔍 Conclusión:", chi2_res['conclusion'])

        from scipy.stats import chi2_contingency
        expected = chi2_contingency(tabla)[3]
        if (expected < 5).sum() / expected.size > 0.2:
            print("⚠️ Más del 20% de celdas tienen frecuencia esperada < 5 → resultado poco fiable")

        n = tabla.values.sum()
        phi2 = stat / n
        r, k = tabla.shape
        cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
        print(f"🎯 Tamaño del efecto (Cramér's V): {cramers_v:.3f}")

        print("\n🧠 Interpretación integral")
        print("Este test evalúa si existe asociación significativa entre dos variables categóricas.")
        print("Cramér's V ayuda a interpretar la fuerza de la relación.")

        print("\n📌 ¿Por qué es importante?")
        print("Entender si dos variables categóricas están relacionadas puede revelar patrones o dependencias.")

        print("\n📊 Visualización Heatmap interactivo de frecuencias")
        fig = go.Figure(data=go.Heatmap(
            z=tabla.values,
            x=tabla.columns.astype(str),
            y=tabla.index.astype(str),
            colorscale='Viridis'))
        fig.update_layout(title=f"Heatmap de contingencia: {col_x} vs {col_y}",
                        xaxis_title=col_y, yaxis_title=col_x)
        fig.show()

        print("\n📋 Resumen del test aplicado:")
        resumen_df = pd.DataFrame([{
            "Test": chi2_res['test_name'],
            "p-valor": round(chi2_res['p_value'], 4),
            "Significativo": "✅" if chi2_res['p_value'] < 0.05 else "❌",
            "Tamaño del efecto": round(cramers_v, 3),
            "Recomendación": chi2_res['recommendation']
        }])
        display(HTML(resumen_df.to_html(index=False)))

        if hasattr(self, "log"):
            self.log(f"Completado Análisis Bivariante de '{col_x}' y '{col_y}'")


"""    
Siguientes mejoras recomendadas:
0. poder analizar dos columnas ( por ejemplo -> ventas por marca)

2. Fase A/B Testing)
Mejorar grafico

Mostrar resultado del test de Levene para varianzas.

3. Exportar informe (opcional)
Exportar a .txt o .pdf el log y/o gráficos.
"""