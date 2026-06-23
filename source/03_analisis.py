from __future__ import annotations

import os
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")

RUTA_BASE: Path = Path(__file__).resolve().parents[1]
RUTA_DATASET: Path = RUTA_BASE / "dataset"
RUTA_DOCS: Path = RUTA_BASE / "docs"
RUTA_FIGURAS: Path = RUTA_DOCS / "figures"
RUTA_MODELO_TIPADO: Path = RUTA_DATASET / "bibliotecas_barcelona_modelo_tipado.csv"
RUTA_RESULTADOS_SUPERVISADO: Path = RUTA_DOCS / "resultados_modelo_supervisado.xlsx"
RUTA_RESULTADOS_NO_SUPERVISADO: Path = RUTA_DOCS / "resultados_modelo_no_supervisado.xlsx"
RUTA_CONTRASTE_HIPOTESIS: Path = RUTA_DOCS / "contraste_hipotesis_4_2.xlsx"
RUTA_MATRIZ_CONFUSION: Path = RUTA_FIGURAS / "matriz_confusion_logistica.svg"
RUTA_COEFICIENTES: Path = RUTA_FIGURAS / "coeficientes_logistica.svg"
RUTA_CLUSTERS_PCA: Path = RUTA_FIGURAS / "clusters_kmeans_pca.svg"
RUTA_PERFIL_CLUSTERS: Path = RUTA_FIGURAS / "perfil_clusters_kmeans.svg"
RUTA_CONTRASTE_TIPOLOGIA: Path = RUTA_FIGURAS / "contraste_catalogo_tipologia.svg"

VARIABLE_OBJETIVO: str = "tiene_catalogo"

VARIABLES_NUMERICAS: list[str] = [
    "antiguedad_biblioteca",
    "bibliotecas_municipio",
    "poblacion_municipio",
    "densidad_hab_km2",
]

VARIABLES_BINARIAS: list[str] = [
    "tiene_web",
    "anio_fundacion_missing",
    "tiene_poblacion_externa",
    "antiguedad_biblioteca_imputada",
    "poblacion_municipio_imputada",
    "densidad_hab_km2_imputada",
]

VARIABLES_CATEGORICAS: list[str] = [
    "tipologia_simplificada",
    "titularidad_simplificada",
    "gestion_simplificada",
]

VARIABLES_EXPLICATIVAS: list[str] = (
    VARIABLES_NUMERICAS + VARIABLES_BINARIAS + VARIABLES_CATEGORICAS
)


def cargar_dataset_analisis(ruta_entrada: Path = RUTA_MODELO_TIPADO) -> pd.DataFrame:
    """Carga el dataset tipado que se usara en el analisis supervisado."""
    return pd.read_csv(ruta_entrada)


def separar_variables_supervisado(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables explicativas y variable objetivo.

    Se usan las variables simplificadas de tipologia, titularidad y gestion
    para evitar alta cardinalidad y redundancia con las categorias originales.
    """
    x: pd.DataFrame = df.loc[:, VARIABLES_EXPLICATIVAS].copy()
    y: pd.Series = df[VARIABLE_OBJETIVO].astype(int)
    return x, y


def obtener_categorias_posibles(x: pd.DataFrame) -> list[list[str]]:
    """Obtiene los niveles posibles de cada categorica antes del split.

    Se usan solo los nombres de categorias, no la variable objetivo. Esto evita
    que el one-hot encoder encuentre categorias desconocidas en test cuando una
    categoria minoritaria no aparece en el subconjunto de entrenamiento.
    """
    return [sorted(x[variable].dropna().astype(str).unique()) for variable in VARIABLES_CATEGORICAS]


def crear_pipeline_regresion_logistica(categorias_posibles: list[list[str]]) -> Pipeline:
    """Crea el pipeline de preprocesado y regresion logistica.

    Las numericas se estandarizan porque la regresion logistica es sensible a
    escalas diferentes. Las binarias se pasan sin transformar. Las categoricas
    se codifican con one-hot encoding para convertir los factores en variables
    numericas. El modelo usa `class_weight="balanced"` porque la clase objetivo
    esta desbalanceada: hay muchas mas bibliotecas con catalogo que sin catalogo.
    """
    preprocesador: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), VARIABLES_NUMERICAS),
            ("bin", "passthrough", VARIABLES_BINARIAS),
            (
                "cat",
                OneHotEncoder(
                    categories=categorias_posibles,
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                VARIABLES_CATEGORICAS,
            ),
        ]
    )
    modelo: LogisticRegression = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="liblinear",
    )
    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", modelo),
        ]
    )


def crear_preprocesador_no_supervisado(categorias_posibles: list[list[str]]) -> ColumnTransformer:
    """Crea el preprocesador usado para el modelo no supervisado.

    En clustering no hay variable objetivo. Por eso se transforma solo la matriz
    de variables explicativas: numericas escaladas, binarias como 0/1 y
    categoricas codificadas con one-hot. En este caso no se elimina ninguna
    categoria de referencia porque K-Means trabaja con distancias y no hay un
    problema de multicolinealidad asociado a coeficientes interpretables.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), VARIABLES_NUMERICAS),
            ("bin", "passthrough", VARIABLES_BINARIAS),
            (
                "cat",
                OneHotEncoder(
                    categories=categorias_posibles,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                VARIABLES_CATEGORICAS,
            ),
        ]
    )


def preparar_matriz_no_supervisada(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str], ColumnTransformer]:
    """Prepara la matriz numerica para clustering.

    La variable `tiene_catalogo` no se incluye en el ajuste del modelo no
    supervisado. Se conserva fuera del modelo para describir posteriormente los
    clusters y comprobar si los grupos encontrados muestran diferencias en la
    disponibilidad de catalogo.
    """
    x = df.loc[:, VARIABLES_EXPLICATIVAS].copy()
    categorias_posibles = obtener_categorias_posibles(x)
    preprocesador = crear_preprocesador_no_supervisado(categorias_posibles)
    matriz = preprocesador.fit_transform(x)
    nombres_variables = preprocesador.get_feature_names_out().tolist()
    return x, matriz, nombres_variables, preprocesador


def evaluar_numero_clusters(
    matriz: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
) -> pd.DataFrame:
    """Evalua varios valores de k mediante inercia y silueta.

    La inercia mide compactacion interna de los clusters y siempre baja al
    aumentar k. La silueta ayuda a elegir una particion con grupos cohesionados
    y separados, por lo que se usa como criterio principal en este ejercicio.
    """
    resultados: list[dict[str, float | int]] = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            etiquetas = modelo.fit_predict(matriz)
        resultados.append(
            {
                "k": k,
                "inercia": modelo.inertia_,
                "silueta": silhouette_score(matriz, etiquetas),
            }
        )
    return pd.DataFrame(resultados)


def seleccionar_k(evaluacion_k: pd.DataFrame) -> int:
    """Selecciona el numero de clusters con mayor coeficiente de silueta."""
    fila_mejor = evaluacion_k.sort_values(["silueta", "k"], ascending=[False, True]).iloc[0]
    return int(fila_mejor["k"])


def entrenar_kmeans(matriz: np.ndarray, n_clusters: int) -> KMeans:
    """Entrena el modelo K-Means final con el numero de clusters seleccionado."""
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        modelo.fit(matriz)
    return modelo


def describir_clusters(df: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Resume el perfil de cada cluster para facilitar su interpretacion.

    Aunque `tiene_catalogo` no participa en el ajuste del modelo, se incorpora
    aqui como variable descriptiva para comprobar si los grupos no supervisados
    se diferencian tambien en disponibilidad de catalogo.
    """
    datos = df.copy()
    datos["cluster"] = etiquetas
    total = len(datos)
    resumen: list[dict[str, object]] = []
    for cluster, grupo in datos.groupby("cluster"):
        fila: dict[str, object] = {
            "cluster": int(cluster),
            "registros": len(grupo),
            "porcentaje_registros": round(len(grupo) / total * 100, 2),
            "pct_con_catalogo": round(grupo[VARIABLE_OBJETIVO].mean() * 100, 2),
            "pct_con_web": round(grupo["tiene_web"].mean() * 100, 2),
            "media_antiguedad": round(grupo["antiguedad_biblioteca"].mean(), 2),
            "media_bibliotecas_municipio": round(grupo["bibliotecas_municipio"].mean(), 2),
            "media_poblacion_municipio": round(grupo["poblacion_municipio"].mean(), 2),
            "media_densidad_hab_km2": round(grupo["densidad_hab_km2"].mean(), 2),
            "tipologia_mas_frecuente": grupo["tipologia_simplificada"].mode().iat[0],
            "titularidad_mas_frecuente": grupo["titularidad_simplificada"].mode().iat[0],
            "gestion_mas_frecuente": grupo["gestion_simplificada"].mode().iat[0],
        }
        resumen.append(fila)
    return pd.DataFrame(resumen).sort_values("cluster").reset_index(drop=True)


def calcular_componentes_pca(matriz: np.ndarray, etiquetas: np.ndarray) -> pd.DataFrame:
    """Reduce la matriz transformada a dos componentes para visualizar clusters."""
    pca = PCA(n_components=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        componentes = pca.fit_transform(matriz)
    return pd.DataFrame(
        {
            "cluster": etiquetas,
            "pca_1": componentes[:, 0],
            "pca_2": componentes[:, 1],
            "varianza_pca_1": pca.explained_variance_ratio_[0],
            "varianza_pca_2": pca.explained_variance_ratio_[1],
        }
    )


def crear_tabla_contingencia(
    df: pd.DataFrame,
    variable_filas: str = "tipologia_simplificada",
    variable_columnas: str = VARIABLE_OBJETIVO,
) -> pd.DataFrame:
    """Crea la tabla de contingencia para el contraste chi-cuadrado.

    Se cruzan categorias de biblioteca con disponibilidad de catalogo. Esta
    tabla contiene frecuencias observadas y es la base del contraste de
    independencia.
    """
    tabla = pd.crosstab(df[variable_filas], df[variable_columnas])
    tabla = tabla.rename(columns={0: "sin_catalogo", 1: "con_catalogo"})
    return tabla.reset_index()


def calcular_porcentajes_contingencia(
    tabla_contingencia: pd.DataFrame,
    variable_filas: str,
) -> pd.DataFrame:
    """Calcula porcentajes por fila para interpretar diferencias por tipologia."""
    porcentajes = tabla_contingencia.copy()
    total_fila = porcentajes[["sin_catalogo", "con_catalogo"]].sum(axis=1)
    porcentajes["pct_sin_catalogo"] = (porcentajes["sin_catalogo"] / total_fila * 100).round(2)
    porcentajes["pct_con_catalogo"] = (porcentajes["con_catalogo"] / total_fila * 100).round(2)
    porcentajes["total"] = total_fila
    return porcentajes[
        [
            variable_filas,
            "total",
            "sin_catalogo",
            "con_catalogo",
            "pct_sin_catalogo",
            "pct_con_catalogo",
        ]
    ]


def calcular_v_cramer(chi2: float, total: int, filas: int, columnas: int) -> float:
    """Calcula V de Cramer como medida de tamano del efecto."""
    denominador = total * min(filas - 1, columnas - 1)
    if denominador == 0:
        return 0.0
    return float(np.sqrt(chi2 / denominador))


def interpretar_v_cramer(valor: float) -> str:
    """Devuelve una interpretacion orientativa de V de Cramer."""
    if valor < 0.10:
        return "asociacion muy debil"
    if valor < 0.30:
        return "asociacion debil"
    if valor < 0.50:
        return "asociacion moderada"
    return "asociacion fuerte"


def crear_tipologia_contraste(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa categorias pequenas para cumplir mejor el supuesto chi-cuadrado."""
    datos = df.copy()

    def agrupar_tipologia(valor: str) -> str:
        if valor in {"Bibliotecas publicas", "Bibliotecas universitarias"}:
            return valor
        return "Bibliotecas especializadas, especificas y otras"

    datos["tipologia_contraste"] = datos["tipologia_simplificada"].map(agrupar_tipologia)
    return datos


def ejecutar_chi_cuadrado_tipologia_catalogo(
    df: pd.DataFrame,
    variable_filas: str = "tipologia_simplificada",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Contrasta independencia entre tipologia simplificada y catalogo.

    H0: la disponibilidad de catalogo es independiente de la tipologia.
    H1: existe asociacion entre tipologia y disponibilidad de catalogo.
    """
    tabla_contingencia = crear_tabla_contingencia(df, variable_filas=variable_filas)
    observados = tabla_contingencia.set_index(variable_filas)
    chi2, p_valor, grados_libertad, esperados = chi2_contingency(observados)
    total = int(observados.to_numpy().sum())
    v_cramer = calcular_v_cramer(
        chi2=chi2,
        total=total,
        filas=observados.shape[0],
        columnas=observados.shape[1],
    )
    frecuencias_esperadas = pd.DataFrame(
        esperados,
        index=observados.index,
        columns=observados.columns,
    ).reset_index()
    porcentajes = calcular_porcentajes_contingencia(tabla_contingencia, variable_filas=variable_filas)
    min_esperada = float(np.min(esperados))
    celdas_esperadas_menor_5 = int((esperados < 5).sum())
    resultado = pd.DataFrame(
        [
            {
                "contraste": "Chi-cuadrado de independencia",
                "variable_filas": variable_filas,
                "variable_columnas": VARIABLE_OBJETIVO,
                "hipotesis_nula": "La disponibilidad de catalogo es independiente de la tipologia.",
                "hipotesis_alternativa": "Existe asociacion entre tipologia y disponibilidad de catalogo.",
                "chi2": chi2,
                "p_valor": p_valor,
                "grados_libertad": int(grados_libertad),
                "alfa": 0.05,
                "decision": "rechazar H0" if p_valor < 0.05 else "no rechazar H0",
                "v_cramer": v_cramer,
                "interpretacion_v_cramer": interpretar_v_cramer(v_cramer),
                "frecuencia_esperada_minima": min_esperada,
                "celdas_esperadas_menor_5": celdas_esperadas_menor_5,
                "supuesto_frecuencias": "cumplido" if celdas_esperadas_menor_5 == 0 else "interpretar con cautela",
            }
        ]
    )
    interpretacion = pd.DataFrame(
        [
            {
                "aspecto": "normalidad_homocedasticidad",
                "comentario": "No aplica porque el contraste usa dos variables categoricas y trabaja con frecuencias.",
            },
            {
                "aspecto": "resultado",
                "comentario": (
                    "Se rechaza la independencia entre tipologia y catalogo."
                    if p_valor < 0.05
                    else "No hay evidencia suficiente para rechazar la independencia entre tipologia y catalogo."
                ),
            },
            {
                "aspecto": "tamano_efecto",
                "comentario": f"V de Cramer = {v_cramer:.3f}, interpretado como {interpretar_v_cramer(v_cramer)}.",
            },
        ]
    )
    return tabla_contingencia, porcentajes, frecuencias_esperadas, resultado, interpretacion


def dividir_entrenamiento_prueba(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en entrenamiento y prueba con estratificacion.

    La estratificacion conserva una proporcion similar de bibliotecas con y sin
    catalogo en ambos subconjuntos, algo importante por el desbalance de clases.
    """
    return train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )


def entrenar_modelo_supervisado(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    categorias_posibles: list[list[str]],
) -> Pipeline:
    """Entrena una regresion logistica sobre el conjunto de entrenamiento."""
    pipeline: Pipeline = crear_pipeline_regresion_logistica(categorias_posibles)
    pipeline.fit(x_train, y_train)
    return pipeline


def evaluar_modelo_supervisado(
    modelo: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula metricas y matriz de confusion del modelo supervisado."""
    # Algunas combinaciones de numpy/scikit-learn pueden emitir RuntimeWarning
    # durante el producto matricial interno aunque las predicciones sean finitas.
    # Se silencian solo esos avisos y se valida explicitamente el resultado.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        predicciones = modelo.predict(x_test)
        probabilidades = modelo.predict_proba(x_test)[:, 1]

    if not np.isfinite(probabilidades).all():
        raise ValueError("El modelo ha generado probabilidades no finitas.")

    matriz = confusion_matrix(y_test, predicciones, labels=[0, 1])

    metricas: pd.DataFrame = pd.DataFrame(
        [
            {
                "metrica": "accuracy",
                "valor": accuracy_score(y_test, predicciones),
                "interpretacion": "Proporcion total de clasificaciones correctas.",
            },
            {
                "metrica": "precision_macro",
                "valor": precision_score(y_test, predicciones, average="macro", zero_division=0),
                "interpretacion": "Precision media entre clases; util con clases desbalanceadas.",
            },
            {
                "metrica": "recall_macro",
                "valor": recall_score(y_test, predicciones, average="macro", zero_division=0),
                "interpretacion": "Sensibilidad media entre clases.",
            },
            {
                "metrica": "f1_macro",
                "valor": f1_score(y_test, predicciones, average="macro", zero_division=0),
                "interpretacion": "Equilibrio entre precision y recall medio por clase.",
            },
            {
                "metrica": "roc_auc",
                "valor": roc_auc_score(y_test, probabilidades),
                "interpretacion": "Capacidad de separar bibliotecas con y sin catalogo.",
            },
        ]
    )
    matriz_df: pd.DataFrame = pd.DataFrame(
        matriz,
        index=["real_sin_catalogo", "real_con_catalogo"],
        columns=["pred_sin_catalogo", "pred_con_catalogo"],
    ).reset_index(names="clase_real")
    return metricas, matriz_df


def extraer_coeficientes_modelo(modelo: Pipeline) -> pd.DataFrame:
    """Extrae coeficientes de la regresion logistica con nombres de variables.

    Los coeficientes positivos se asocian con mayor probabilidad estimada de
    tener catalogo, mientras que los negativos se asocian con menor probabilidad.
    """
    preprocesador: ColumnTransformer = modelo.named_steps["preprocesador"]
    regresion: LogisticRegression = modelo.named_steps["modelo"]
    nombres_variables = preprocesador.get_feature_names_out()
    coeficientes = regresion.coef_[0]
    resultado = pd.DataFrame(
        {
            "variable": nombres_variables,
            "coeficiente": coeficientes,
            "abs_coeficiente": abs(coeficientes),
        }
    )
    return resultado.sort_values("abs_coeficiente", ascending=False).reset_index(drop=True)


def nombre_columna_excel(indice: int) -> str:
    """Convierte un indice numerico en nombre de columna Excel."""
    nombre = ""
    while indice:
        indice, resto = divmod(indice - 1, 26)
        nombre = chr(65 + resto) + nombre
    return nombre


def celda_excel_xml(fila: int, columna: int, valor: object, estilo: int | None = None) -> str:
    """Genera una celda XML de texto para un XLSX sencillo."""
    referencia = f"{nombre_columna_excel(columna)}{fila}"
    atributo_estilo = f' s="{estilo}"' if estilo is not None else ""
    texto = escape(str(valor))
    return f'<c r="{referencia}" t="inlineStr"{atributo_estilo}><is><t>{texto}</t></is></c>'


def hoja_excel_xml(df: pd.DataFrame) -> str:
    """Convierte un DataFrame en XML de hoja de calculo."""
    filas = [list(df.columns)] + df.astype(object).values.tolist()
    filas_xml: list[str] = []
    for indice_fila, fila in enumerate(filas, start=1):
        estilo = 1 if indice_fila == 1 else None
        celdas = "".join(
            celda_excel_xml(indice_fila, indice_columna, valor, estilo)
            for indice_columna, valor in enumerate(fila, start=1)
        )
        filas_xml.append(f'<row r="{indice_fila}">{celdas}</row>')

    ultima_celda = f"{nombre_columna_excel(len(df.columns))}{len(filas)}"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:{ultima_celda}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols><col min="1" max="{len(df.columns)}" width="38" customWidth="1"/></cols>
  <sheetData>{''.join(filas_xml)}</sheetData>
  <autoFilter ref="A1:{ultima_celda}"/>
</worksheet>'''


def guardar_resultados_xlsx(
    hojas: dict[str, pd.DataFrame],
    ruta_salida: Path = RUTA_RESULTADOS_SUPERVISADO,
) -> None:
    """Guarda las tablas principales del modelo en un archivo XLSX."""
    content_types_overrides = "\n".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, len(hojas) + 1)
    )
    sheets_xml = "\n".join(
        f'<sheet name="{escape(nombre[:31])}" sheetId="{i}" r:id="rId{i}"/>'
        for i, nombre in enumerate(hojas, start=1)
    )
    rels_sheets = "\n".join(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(hojas) + 1)
    )
    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  {content_types_overrides}
</Types>'''
    workbook = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{sheets_xml}</sheets>
</workbook>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
    rels_root = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    rels_workbook = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {rels_sheets}
  <Relationship Id="rId{len(hojas) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    with ZipFile(ruta_salida, "w", ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", rels_root)
        xlsx.writestr("xl/workbook.xml", workbook)
        xlsx.writestr("xl/_rels/workbook.xml.rels", rels_workbook)
        xlsx.writestr("xl/styles.xml", styles)
        for i, dataframe in enumerate(hojas.values(), start=1):
            xlsx.writestr(f"xl/worksheets/sheet{i}.xml", hoja_excel_xml(dataframe))


def guardar_matriz_confusion_svg(matriz: pd.DataFrame, ruta_salida: Path) -> None:
    """Guarda una matriz de confusion sencilla en formato SVG."""
    valores = matriz.set_index("clase_real")
    tn = int(valores.loc["real_sin_catalogo", "pred_sin_catalogo"])
    fp = int(valores.loc["real_sin_catalogo", "pred_con_catalogo"])
    fn = int(valores.loc["real_con_catalogo", "pred_sin_catalogo"])
    tp = int(valores.loc["real_con_catalogo", "pred_con_catalogo"])
    maximo = max(tn, fp, fn, tp, 1)
    celdas = [
        (210, 110, tn, "Real sin catalogo / Pred sin catalogo"),
        (470, 110, fp, "Real sin catalogo / Pred con catalogo"),
        (210, 290, fn, "Real con catalogo / Pred sin catalogo"),
        (470, 290, tp, "Real con catalogo / Pred con catalogo"),
    ]
    rects: list[str] = []
    for x, y, valor, etiqueta in celdas:
        intensidad = 0.2 + 0.7 * valor / maximo
        color = f"rgba(76,120,168,{intensidad:.2f})"
        rects.append(
            f'<rect x="{x}" y="{y}" width="220" height="140" fill="{color}" stroke="#333"/>'
            f'<text x="{x + 110}" y="{y + 65}" font-size="32" text-anchor="middle">{valor}</text>'
            f'<text x="{x + 110}" y="{y + 95}" font-size="12" text-anchor="middle">{escape(etiqueta)}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="500" viewBox="0 0 760 500">
<rect width="100%" height="100%" fill="white"/>
<text x="380" y="45" font-size="24" text-anchor="middle" font-family="Arial">Matriz de confusion - regresion logistica</text>
<text x="320" y="90" font-size="15" text-anchor="middle">Prediccion: sin catalogo</text>
<text x="580" y="90" font-size="15" text-anchor="middle">Prediccion: con catalogo</text>
<text x="110" y="185" font-size="15" text-anchor="middle">Real: sin catalogo</text>
<text x="110" y="365" font-size="15" text-anchor="middle">Real: con catalogo</text>
{''.join(rects)}
</svg>'''
    ruta_salida.write_text(svg, encoding="utf-8")


def guardar_coeficientes_svg(coeficientes: pd.DataFrame, ruta_salida: Path, n: int = 12) -> None:
    """Guarda los coeficientes mas relevantes de la regresion logistica."""
    top = coeficientes.head(n).sort_values("coeficiente")
    ancho = 1100
    alto = 620
    margen_izquierdo = 420
    margen_derecho = 80
    margen_superior = 70
    paso = 40
    max_abs = max(top["coeficiente"].abs().max(), 0.1)
    centro_x = margen_izquierdo + (ancho - margen_izquierdo - margen_derecho) / 2
    escala = (ancho - margen_izquierdo - margen_derecho) / 2 / max_abs
    barras: list[str] = []
    for i, row in enumerate(top.itertuples(index=False), start=0):
        y = margen_superior + i * paso
        coef = float(row.coeficiente)
        longitud = abs(coef) * escala
        x = centro_x if coef >= 0 else centro_x - longitud
        color = "#54a24b" if coef >= 0 else "#e45756"
        barras.append(
            f'<text x="{margen_izquierdo - 15}" y="{y + 18}" font-size="12" text-anchor="end">{escape(str(row.variable))}</text>'
            f'<rect x="{x:.2f}" y="{y}" width="{max(longitud, 1):.2f}" height="24" fill="{color}"/>'
            f'<text x="{x + (longitud if coef >= 0 else -8):.2f}" y="{y + 18}" font-size="12" '
            f'text-anchor="{"start" if coef >= 0 else "end"}">{coef:.3f}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">
<rect width="100%" height="100%" fill="white"/>
<text x="{ancho / 2}" y="35" font-size="24" text-anchor="middle" font-family="Arial">Coeficientes principales - regresion logistica</text>
<line x1="{centro_x:.2f}" y1="55" x2="{centro_x:.2f}" y2="{alto - 70}" stroke="#333"/>
{''.join(barras)}
<text x="{centro_x - 160}" y="{alto - 25}" font-size="13" text-anchor="middle">Menor probabilidad de catalogo</text>
<text x="{centro_x + 160}" y="{alto - 25}" font-size="13" text-anchor="middle">Mayor probabilidad de catalogo</text>
</svg>'''
    ruta_salida.write_text(svg, encoding="utf-8")


def guardar_clusters_pca_svg(componentes: pd.DataFrame, ruta_salida: Path) -> None:
    """Guarda una visualizacion 2D de los clusters mediante PCA."""
    ancho = 900
    alto = 640
    margen = 80
    colores = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    x_min, x_max = componentes["pca_1"].min(), componentes["pca_1"].max()
    y_min, y_max = componentes["pca_2"].min(), componentes["pca_2"].max()
    rango_x = x_max - x_min or 1
    rango_y = y_max - y_min or 1

    puntos: list[str] = []
    for fila in componentes.itertuples(index=False):
        x = margen + (float(fila.pca_1) - x_min) / rango_x * (ancho - 2 * margen)
        y = alto - margen - (float(fila.pca_2) - y_min) / rango_y * (alto - 2 * margen)
        color = colores[int(fila.cluster) % len(colores)]
        puntos.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" fill-opacity="0.72"/>'
        )

    clusters = sorted(componentes["cluster"].unique())
    leyenda = []
    for i, cluster in enumerate(clusters):
        color = colores[int(cluster) % len(colores)]
        y = 86 + i * 24
        leyenda.append(
            f'<rect x="{ancho - 165}" y="{y - 12}" width="14" height="14" fill="{color}"/>'
            f'<text x="{ancho - 145}" y="{y}" font-size="13">Cluster {int(cluster)}</text>'
        )

    var_1 = componentes["varianza_pca_1"].iat[0] * 100
    var_2 = componentes["varianza_pca_2"].iat[0] * 100
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">
<rect width="100%" height="100%" fill="white"/>
<text x="{ancho / 2}" y="38" font-size="24" text-anchor="middle" font-family="Arial">Clusters K-Means proyectados con PCA</text>
<line x1="{margen}" y1="{alto - margen}" x2="{ancho - margen}" y2="{alto - margen}" stroke="#333"/>
<line x1="{margen}" y1="{margen}" x2="{margen}" y2="{alto - margen}" stroke="#333"/>
<text x="{ancho / 2}" y="{alto - 25}" font-size="13" text-anchor="middle">Componente 1 ({var_1:.1f}% varianza)</text>
<text x="25" y="{alto / 2}" font-size="13" text-anchor="middle" transform="rotate(-90 25 {alto / 2})">Componente 2 ({var_2:.1f}% varianza)</text>
{''.join(puntos)}
{''.join(leyenda)}
</svg>'''
    ruta_salida.write_text(svg, encoding="utf-8")


def guardar_perfil_clusters_svg(perfil: pd.DataFrame, ruta_salida: Path) -> None:
    """Guarda un grafico comparativo del tamano y catalogo por cluster."""
    ancho = 900
    alto = 520
    margen_izquierdo = 90
    margen_inferior = 80
    margen_superior = 70
    colores = {"registros": "#4c78a8", "catalogo": "#54a24b"}
    max_valor = max(perfil["porcentaje_registros"].max(), perfil["pct_con_catalogo"].max(), 1)
    escala = (alto - margen_superior - margen_inferior) / max_valor
    ancho_grupo = (ancho - margen_izquierdo - 70) / len(perfil)
    ancho_barra = min(48, ancho_grupo / 3)
    barras: list[str] = []

    for i, fila in enumerate(perfil.itertuples(index=False)):
        centro = margen_izquierdo + ancho_grupo * i + ancho_grupo / 2
        valores = [
            ("registros", float(fila.porcentaje_registros)),
            ("catalogo", float(fila.pct_con_catalogo)),
        ]
        for j, (tipo, valor) in enumerate(valores):
            x = centro + (j - 1) * ancho_barra
            y = alto - margen_inferior - valor * escala
            barras.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{ancho_barra:.2f}" height="{valor * escala:.2f}" fill="{colores[tipo]}"/>'
                f'<text x="{x + ancho_barra / 2:.2f}" y="{y - 6:.2f}" font-size="11" text-anchor="middle">{valor:.1f}%</text>'
            )
        barras.append(
            f'<text x="{centro:.2f}" y="{alto - 48}" font-size="13" text-anchor="middle">Cluster {int(fila.cluster)}</text>'
            f'<text x="{centro:.2f}" y="{alto - 30}" font-size="11" text-anchor="middle">{int(fila.registros)} casos</text>'
        )

    eje_y = "".join(
        f'<line x1="{margen_izquierdo - 5}" y1="{alto - margen_inferior - v * escala:.2f}" x2="{ancho - 70}" y2="{alto - margen_inferior - v * escala:.2f}" stroke="#ddd"/>'
        f'<text x="{margen_izquierdo - 12}" y="{alto - margen_inferior - v * escala + 4:.2f}" font-size="11" text-anchor="end">{v}%</text>'
        for v in range(0, 101, 20)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">
<rect width="100%" height="100%" fill="white"/>
<text x="{ancho / 2}" y="35" font-size="24" text-anchor="middle" font-family="Arial">Perfil de clusters K-Means</text>
{eje_y}
<line x1="{margen_izquierdo}" y1="{alto - margen_inferior}" x2="{ancho - 70}" y2="{alto - margen_inferior}" stroke="#333"/>
<line x1="{margen_izquierdo}" y1="{margen_superior}" x2="{margen_izquierdo}" y2="{alto - margen_inferior}" stroke="#333"/>
{''.join(barras)}
<rect x="{ancho - 260}" y="58" width="14" height="14" fill="{colores["registros"]}"/>
<text x="{ancho - 240}" y="70" font-size="13">Peso del cluster</text>
<rect x="{ancho - 260}" y="82" width="14" height="14" fill="{colores["catalogo"]}"/>
<text x="{ancho - 240}" y="94" font-size="13">% con catalogo</text>
</svg>'''
    ruta_salida.write_text(svg, encoding="utf-8")


def guardar_contraste_tipologia_svg(porcentajes: pd.DataFrame, ruta_salida: Path) -> None:
    """Guarda un grafico de barras apiladas para el contraste de hipotesis."""
    datos = porcentajes.sort_values("pct_con_catalogo", ascending=True)
    variable_filas = datos.columns[0]
    ancho = 980
    alto = 520
    margen_izquierdo = 300
    margen_derecho = 70
    margen_superior = 70
    paso = 70
    ancho_barras = ancho - margen_izquierdo - margen_derecho
    colores = {"sin": "#e45756", "con": "#54a24b"}
    filas_svg: list[str] = []

    for i, fila in enumerate(datos.itertuples(index=False), start=0):
        y = margen_superior + i * paso
        ancho_sin = float(fila.pct_sin_catalogo) / 100 * ancho_barras
        ancho_con = float(fila.pct_con_catalogo) / 100 * ancho_barras
        filas_svg.append(
            f'<text x="{margen_izquierdo - 15}" y="{y + 24}" font-size="12" text-anchor="end">{escape(str(getattr(fila, variable_filas)))}</text>'
            f'<rect x="{margen_izquierdo}" y="{y}" width="{ancho_sin:.2f}" height="30" fill="{colores["sin"]}"/>'
            f'<rect x="{margen_izquierdo + ancho_sin:.2f}" y="{y}" width="{ancho_con:.2f}" height="30" fill="{colores["con"]}"/>'
            f'<text x="{margen_izquierdo + ancho_barras + 8}" y="{y + 21}" font-size="12">{float(fila.pct_con_catalogo):.1f}% con catalogo</text>'
        )

    marcas = "".join(
        f'<line x1="{margen_izquierdo + ancho_barras * v / 100:.2f}" y1="{margen_superior - 12}" '
        f'x2="{margen_izquierdo + ancho_barras * v / 100:.2f}" y2="{alto - 90}" stroke="#ddd"/>'
        f'<text x="{margen_izquierdo + ancho_barras * v / 100:.2f}" y="{alto - 58}" font-size="11" text-anchor="middle">{v}%</text>'
        for v in range(0, 101, 20)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">
<rect width="100%" height="100%" fill="white"/>
<text x="{ancho / 2}" y="35" font-size="24" text-anchor="middle" font-family="Arial">Catalogo por tipologia simplificada</text>
{marcas}
{''.join(filas_svg)}
<rect x="{margen_izquierdo}" y="{alto - 34}" width="14" height="14" fill="{colores["sin"]}"/>
<text x="{margen_izquierdo + 20}" y="{alto - 22}" font-size="13">Sin catalogo</text>
<rect x="{margen_izquierdo + 130}" y="{alto - 34}" width="14" height="14" fill="{colores["con"]}"/>
<text x="{margen_izquierdo + 150}" y="{alto - 22}" font-size="13">Con catalogo</text>
</svg>'''
    ruta_salida.write_text(svg, encoding="utf-8")


def ejecutar_modelo_supervisado() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta el flujo completo del modelo supervisado."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    dataset = cargar_dataset_analisis()
    x, y = separar_variables_supervisado(dataset)
    categorias_posibles = obtener_categorias_posibles(x)
    x_train, x_test, y_train, y_test = dividir_entrenamiento_prueba(x, y)
    modelo = entrenar_modelo_supervisado(x_train, y_train, categorias_posibles)
    metricas, matriz = evaluar_modelo_supervisado(modelo, x_test, y_test)
    coeficientes = extraer_coeficientes_modelo(modelo)

    resumen_particion = pd.DataFrame(
        [
            {"conjunto": "entrenamiento", "registros": len(y_train), "con_catalogo": int(y_train.sum()), "sin_catalogo": int((y_train == 0).sum())},
            {"conjunto": "prueba", "registros": len(y_test), "con_catalogo": int(y_test.sum()), "sin_catalogo": int((y_test == 0).sum())},
        ]
    )
    guardar_resultados_xlsx(
        {
            "metricas": metricas,
            "matriz_confusion": matriz,
            "coeficientes": coeficientes,
            "particion": resumen_particion,
        }
    )
    guardar_matriz_confusion_svg(matriz, RUTA_MATRIZ_CONFUSION)
    guardar_coeficientes_svg(coeficientes, RUTA_COEFICIENTES)

    return metricas, matriz, coeficientes


def ejecutar_modelo_no_supervisado() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta el flujo completo del modelo no supervisado K-Means."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    dataset = cargar_dataset_analisis()
    _, matriz, nombres_variables, _ = preparar_matriz_no_supervisada(dataset)
    evaluacion_k = evaluar_numero_clusters(matriz)
    k_seleccionado = seleccionar_k(evaluacion_k)
    modelo = entrenar_kmeans(matriz, k_seleccionado)
    etiquetas = modelo.labels_

    perfil_clusters = describir_clusters(dataset, etiquetas)
    componentes = calcular_componentes_pca(matriz, etiquetas)
    asignaciones = pd.DataFrame(
        {
            "indice_registro": dataset.index,
            "cluster": etiquetas,
            VARIABLE_OBJETIVO: dataset[VARIABLE_OBJETIVO],
            "tipologia_simplificada": dataset["tipologia_simplificada"],
            "titularidad_simplificada": dataset["titularidad_simplificada"],
            "gestion_simplificada": dataset["gestion_simplificada"],
        }
    )
    variables_transformadas = pd.DataFrame({"variable_transformada": nombres_variables})

    guardar_resultados_xlsx(
        {
            "evaluacion_k": evaluacion_k,
            "perfil_clusters": perfil_clusters,
            "asignaciones": asignaciones,
            "variables_transformadas": variables_transformadas,
            "pca": componentes,
        },
        ruta_salida=RUTA_RESULTADOS_NO_SUPERVISADO,
    )
    guardar_clusters_pca_svg(componentes, RUTA_CLUSTERS_PCA)
    guardar_perfil_clusters_svg(perfil_clusters, RUTA_PERFIL_CLUSTERS)

    return evaluacion_k, perfil_clusters, componentes


def ejecutar_contraste_hipotesis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el contraste de hipotesis del apartado 4.2."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    dataset = cargar_dataset_analisis()
    tabla_original, porcentajes_original, esperadas_original, resultado_original, interpretacion_original = (
        ejecutar_chi_cuadrado_tipologia_catalogo(dataset)
    )
    dataset_agrupado = crear_tipologia_contraste(dataset)
    tabla_agrupada, porcentajes_agrupados, esperadas_agrupadas, resultado_agrupado, interpretacion_agrupada = (
        ejecutar_chi_cuadrado_tipologia_catalogo(dataset_agrupado, variable_filas="tipologia_contraste")
    )
    guardar_resultados_xlsx(
        {
            "tabla_original": tabla_original,
            "porcentajes_original": porcentajes_original,
            "esperadas_original": esperadas_original,
            "resultado_original": resultado_original,
            "tabla_agrupada": tabla_agrupada,
            "porcentajes_agrupados": porcentajes_agrupados,
            "esperadas_agrupadas": esperadas_agrupadas,
            "resultado_agrupado": resultado_agrupado,
            "interpretacion": pd.concat([interpretacion_original, interpretacion_agrupada], ignore_index=True),
        },
        ruta_salida=RUTA_CONTRASTE_HIPOTESIS,
    )
    guardar_contraste_tipologia_svg(porcentajes_agrupados, RUTA_CONTRASTE_TIPOLOGIA)
    return resultado_agrupado, porcentajes_agrupados


def main() -> None:
    metricas, matriz, coeficientes = ejecutar_modelo_supervisado()
    evaluacion_k, perfil_clusters, _ = ejecutar_modelo_no_supervisado()
    resultado_chi2, porcentajes_chi2 = ejecutar_contraste_hipotesis()
    print(f"Resultados guardados en: {RUTA_RESULTADOS_SUPERVISADO}")
    print(f"Resultados no supervisados guardados en: {RUTA_RESULTADOS_NO_SUPERVISADO}")
    print(f"Contraste de hipotesis guardado en: {RUTA_CONTRASTE_HIPOTESIS}")
    print(f"Figura matriz de confusion: {RUTA_MATRIZ_CONFUSION}")
    print(f"Figura coeficientes: {RUTA_COEFICIENTES}")
    print(f"Figura clusters PCA: {RUTA_CLUSTERS_PCA}")
    print(f"Figura perfil clusters: {RUTA_PERFIL_CLUSTERS}")
    print(f"Figura contraste tipologia/catalogo: {RUTA_CONTRASTE_TIPOLOGIA}")
    print("Metricas del modelo supervisado:")
    print(metricas.to_string(index=False))
    print("Matriz de confusion:")
    print(matriz.to_string(index=False))
    print("Coeficientes principales:")
    print(coeficientes.head(12).to_string(index=False))
    print("Evaluacion de k en modelo no supervisado:")
    print(evaluacion_k.to_string(index=False))
    print("Perfil de clusters:")
    print(perfil_clusters.to_string(index=False))
    print("Contraste chi-cuadrado:")
    print(resultado_chi2.to_string(index=False))
    print("Porcentajes por tipologia:")
    print(porcentajes_chi2.to_string(index=False))


if __name__ == "__main__":
    main()
