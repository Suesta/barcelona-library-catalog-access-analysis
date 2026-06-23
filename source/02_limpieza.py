from __future__ import annotations

import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

import pandas as pd


RUTA_BASE: Path = Path(__file__).resolve().parents[1]
RUTA_DATASET: Path = RUTA_BASE / "dataset"
RUTA_FIGURAS: Path = RUTA_BASE / "docs" / "figures"
RUTA_DIAGNOSTICO_3_4: Path = RUTA_BASE / "docs" / "diagnostico_limpieza_3_4.xlsx"
RUTA_INTEGRADO: Path = RUTA_DATASET / "bibliotecas_barcelona_integrado_provisional.csv"
RUTA_MODELO_BASE: Path = RUTA_DATASET / "bibliotecas_barcelona_modelo_base.csv"
RUTA_MODELO_LIMPIO: Path = RUTA_DATASET / "bibliotecas_barcelona_modelo_limpio.csv"
RUTA_MODELO_TIPADO: Path = RUTA_DATASET / "bibliotecas_barcelona_modelo_tipado.csv"

VARIABLES_BINARIAS: list[str] = [
    "tiene_catalogo",
    "tiene_web",
    "anio_fundacion_missing",
    "tiene_poblacion_externa",
    "antiguedad_biblioteca_imputada",
    "poblacion_municipio_imputada",
    "densidad_hab_km2_imputada",
]

VARIABLES_NUMERICAS_A_IMPUTAR: list[str] = [
    "antiguedad_biblioteca",
    "poblacion_municipio",
    "densidad_hab_km2",
]

VARIABLES_MUNICIPALES: set[str] = {
    "poblacion_municipio",
    "densidad_hab_km2",
}

VARIABLES_NUMERICAS: list[str] = [
    "antiguedad_biblioteca",
    "bibliotecas_municipio",
    "poblacion_municipio",
    "densidad_hab_km2",
]

VARIABLES_CATEGORICAS_ORIGINALES: list[str] = [
    "tipologia",
    "titularidad",
    "gestion",
]

VARIABLES_CATEGORICAS_SIMPLIFICADAS: list[str] = [
    "tipologia_simplificada",
    "titularidad_simplificada",
    "gestion_simplificada",
]

VARIABLES_OUTLIERS: list[str] = [
    "antiguedad_biblioteca",
    "bibliotecas_municipio",
    "poblacion_municipio",
    "densidad_hab_km2",
]

CATEGORIAS_ESPERADAS: dict[str, set[str]] = {
    "tipologia_simplificada": {
        "Bibliotecas publicas",
        "Bibliotecas universitarias",
        "Bibliotecas especializadas",
        "Bibliotecas para grupos especificos",
        "Otras tipologias",
    },
    "titularidad_simplificada": {
        "Administracion local",
        "Administracion autonomica",
        "Administracion general del estado",
        "Universitaria publica",
        "Universitaria privada",
        "Privada",
        "Poder judicial",
        "Otras titularidades",
    },
    "gestion_simplificada": {
        "Administracion local",
        "Administracion autonomica",
        "Administracion general del estado",
        "Universitaria publica",
        "Universitaria privada",
        "Privada",
        "Poder judicial",
        "Gestion compartida",
        "Otras gestiones",
    },
}


def cargar_dataset_modelo(ruta_entrada: Path = RUTA_MODELO_BASE) -> pd.DataFrame:
    """Carga el dataset seleccionado para el modelo.

    Este archivo ya contiene solo las variables consideradas utiles para el
    analisis supervisado. La limpieza de este bloque se centra en gestionar
    los valores ausentes de las variables numericas seleccionadas.
    """
    return pd.read_csv(ruta_entrada)


def crear_indicadores_imputacion(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Crea indicadores binarios que marcan que valores seran imputados.

    Cada variable numerica con nulos recibe una columna adicional terminada en
    `_imputada`. Esta columna vale 1 cuando el dato original estaba ausente y
    0 cuando el dato ya estaba informado. Asi se conserva informacion sobre la
    perdida de datos original despues de aplicar la imputacion.
    """
    limpio: pd.DataFrame = df.copy()

    for variable in variables:
        columna_indicador: str = f"{variable}_imputada"
        limpio[columna_indicador] = limpio[variable].isna().astype(int)

    return limpio


def imputar_numericas_con_mediana(
    df: pd.DataFrame,
    variables: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Imputa variables numericas con su mediana.

    Se usa la mediana porque es mas robusta que la media ante distribuciones
    asimetricas y valores extremos, algo esperable en variables municipales
    como poblacion y densidad. En las variables municipales se calcula la
    mediana sobre valores municipales unicos para no sobreponderar municipios
    con muchas bibliotecas en el dataset. La funcion devuelve el dataset
    imputado y las medianas utilizadas para documentar la limpieza.
    """
    limpio: pd.DataFrame = df.copy()
    medianas: dict[str, float] = {}

    for variable in variables:
        if variable in VARIABLES_MUNICIPALES:
            valores_para_mediana: pd.Series = limpio[variable].dropna().drop_duplicates()
        else:
            valores_para_mediana = limpio[variable].dropna()

        mediana: float = float(valores_para_mediana.median())
        medianas[variable] = mediana
        limpio[variable] = limpio[variable].fillna(mediana)

    return limpio, medianas


def generar_dataset_modelo_limpio(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Genera el dataset limpio del modelo para el apartado 3.1.

    Los ceros de las variables binarias no se imputan porque tienen significado
    valido: ausencia de catalogo, ausencia de web, ausencia de poblacion externa
    o ano de fundacion informado segun el caso. Las variables categoricas
    seleccionadas tampoco se imputan porque no presentan nulos ni cadenas
    vacias en el diagnostico realizado.
    """
    limpio: pd.DataFrame = crear_indicadores_imputacion(
        df,
        VARIABLES_NUMERICAS_A_IMPUTAR,
    )
    limpio, medianas = imputar_numericas_con_mediana(
        limpio,
        VARIABLES_NUMERICAS_A_IMPUTAR,
    )

    limpio.to_csv(RUTA_MODELO_LIMPIO, index=False, encoding="utf-8-sig")

    return limpio, medianas


def validar_variables_binarias(df: pd.DataFrame, variables: list[str]) -> None:
    """Comprueba que las variables binarias solo contienen valores 0 y 1.

    Esta validacion evita que una variable indicadora se utilice como numerica
    ordinaria si contiene codigos inesperados. Si se encuentra algun valor
    distinto de 0 o 1, se detiene el proceso para revisar el origen del dato.
    """
    for variable in variables:
        valores: set[int] = set(df[variable].dropna().astype(int).unique())
        valores_no_validos: set[int] = valores - {0, 1}
        if valores_no_validos:
            raise ValueError(
                f"La variable binaria {variable} contiene valores no validos: "
                f"{sorted(valores_no_validos)}"
            )


def simplificar_tipologia(valor: str) -> str:
    """Agrupa la tipologia original en familias bibliotecarias principales."""
    texto: str = str(valor)

    if texto.startswith("Bibliotecas Públicas"):
        return "Bibliotecas publicas"
    if "instituciones de enseñanza superior" in texto:
        return "Bibliotecas universitarias"
    if texto.startswith("Bibliotecas especializadas"):
        return "Bibliotecas especializadas"
    if texto.startswith("Bibliotecas para grupos específicos"):
        return "Bibliotecas para grupos especificos"

    return "Otras tipologias"


def simplificar_titularidad(valor: str) -> str:
    """Agrupa la titularidad en grandes tipos institucionales."""
    texto: str = str(valor)

    if "Administración Local" in texto:
        return "Administracion local"
    if "Administración Autonómica" in texto:
        return "Administracion autonomica"
    if "Administración General del Estado" in texto:
        return "Administracion general del estado"
    if "Universitaria - Pública" in texto:
        return "Universitaria publica"
    if "Universitaria - Privada" in texto:
        return "Universitaria privada"
    if texto.startswith("Privada"):
        return "Privada"
    if "Poder Judicial" in texto:
        return "Poder judicial"

    return "Otras titularidades"


def simplificar_gestion(valor: str) -> str:
    """Agrupa la gestion en categorias institucionales interpretables."""
    texto: str = str(valor)

    if "Gestión compartida" in texto:
        return "Gestion compartida"
    if "Administración Local" in texto:
        return "Administracion local"
    if "Administración Autonómica" in texto:
        return "Administracion autonomica"
    if "Administración General del Estado" in texto:
        return "Administracion general del estado"
    if "Universitaria - Pública" in texto:
        return "Universitaria publica"
    if "Universitaria - Privada" in texto:
        return "Universitaria privada"
    if texto.startswith("Privada"):
        return "Privada"
    if "Poder Judicial" in texto:
        return "Poder judicial"

    return "Otras gestiones"


def gestionar_tipos_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Gestiona los tipos de datos para el apartado 3.2.

    Las binarias se validan y se mantienen como enteros 0/1 porque los modelos
    de clasificacion las pueden usar directamente. Las numericas se fuerzan a
    tipo numerico. Las categoricas originales se convierten a `category` durante
    el proceso y se crean versiones simplificadas para reducir cardinalidad y
    facilitar la futura codificacion mediante variables dummy.
    """
    tipado: pd.DataFrame = df.copy()

    validar_variables_binarias(tipado, VARIABLES_BINARIAS)
    for variable in VARIABLES_BINARIAS:
        tipado[variable] = tipado[variable].astype(int)

    for variable in VARIABLES_NUMERICAS:
        tipado[variable] = pd.to_numeric(tipado[variable], errors="raise")

    for variable in VARIABLES_CATEGORICAS_ORIGINALES:
        tipado[variable] = tipado[variable].astype("category")

    tipado["tipologia_simplificada"] = tipado["tipologia"].apply(simplificar_tipologia)
    tipado["titularidad_simplificada"] = tipado["titularidad"].apply(simplificar_titularidad)
    tipado["gestion_simplificada"] = tipado["gestion"].apply(simplificar_gestion)

    for variable in VARIABLES_CATEGORICAS_SIMPLIFICADAS:
        tipado[variable] = tipado[variable].astype("category")

    tipado.to_csv(RUTA_MODELO_TIPADO, index=False, encoding="utf-8-sig")

    return tipado


def calcular_resumen_outliers_iqr(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Detecta posibles valores extremos mediante el metodo IQR.

    El metodo del rango intercuartilico no presupone normalidad y por eso es
    adecuado para variables asimetricas como poblacion o densidad. La funcion
    solo marca posibles outliers; no elimina ni modifica registros.
    """
    resumen: list[dict[str, float | int | str]] = []

    for variable in variables:
        columna_imputacion: str = f"{variable}_imputada"
        if columna_imputacion in df.columns:
            # Los valores imputados no deben condicionar el calculo de extremos:
            # son estimaciones creadas en 3.1, no observaciones originales.
            serie: pd.Series = pd.to_numeric(
                df.loc[df[columna_imputacion].eq(0), variable],
                errors="raise",
            ).dropna()
        else:
            serie = pd.to_numeric(df[variable], errors="raise").dropna()

        q1: float = float(serie.quantile(0.25))
        mediana: float = float(serie.quantile(0.50))
        q3: float = float(serie.quantile(0.75))
        iqr: float = q3 - q1
        limite_inferior: float = q1 - 1.5 * iqr
        limite_superior: float = q3 + 1.5 * iqr
        mascara_outliers: pd.Series = (serie < limite_inferior) | (serie > limite_superior)
        outliers: int = int(mascara_outliers.sum())

        resumen.append(
            {
                "variable": variable,
                "registros_analizados": int(len(serie)),
                "minimo": float(serie.min()),
                "q1": q1,
                "mediana": mediana,
                "q3": q3,
                "maximo": float(serie.max()),
                "iqr": iqr,
                "limite_inferior": limite_inferior,
                "limite_superior": limite_superior,
                "posibles_outliers": outliers,
                "porcentaje_outliers": round(outliers / len(serie) * 100, 2),
            }
        )

    return pd.DataFrame(resumen)


def escalar_valor(valor: float, minimo: float, maximo: float, inicio: float, fin: float) -> float:
    """Escala un valor numerico a una coordenada grafica."""
    if math.isclose(minimo, maximo):
        return (inicio + fin) / 2
    return inicio + (valor - minimo) / (maximo - minimo) * (fin - inicio)


def guardar_histograma_svg(serie: pd.Series, variable: str, ruta_salida: Path) -> None:
    """Guarda un histograma SVG simple para revisar la distribucion."""
    valores: list[float] = [float(valor) for valor in serie.dropna()]
    minimo: float = min(valores)
    maximo: float = max(valores)
    bins: int = min(12, max(5, int(math.sqrt(len(valores)))))
    ancho: int = 900
    alto: int = 520
    margen_izquierdo: int = 80
    margen_derecho: int = 40
    margen_superior: int = 70
    margen_inferior: int = 80
    plot_ancho: int = ancho - margen_izquierdo - margen_derecho
    plot_alto: int = alto - margen_superior - margen_inferior

    if math.isclose(minimo, maximo):
        conteos = [len(valores)]
        limites = [minimo, maximo]
    else:
        paso: float = (maximo - minimo) / bins
        conteos = [0] * bins
        limites = [minimo + i * paso for i in range(bins + 1)]
        for valor in valores:
            indice: int = min(int((valor - minimo) / paso), bins - 1)
            conteos[indice] += 1

    max_conteo: int = max(conteos)
    barras: list[str] = []
    for indice, conteo in enumerate(conteos):
        barra_ancho: float = plot_ancho / len(conteos) * 0.86
        x: float = margen_izquierdo + indice * plot_ancho / len(conteos) + plot_ancho / len(conteos) * 0.07
        altura_barra: float = 0 if max_conteo == 0 else conteo / max_conteo * plot_alto
        y: float = margen_superior + plot_alto - altura_barra
        barras.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{barra_ancho:.2f}" '
            f'height="{altura_barra:.2f}" fill="#4c78a8"/>'
        )

    etiquetas_x: str = (
        f'<text x="{margen_izquierdo}" y="{alto - 35}" font-size="12" text-anchor="middle">{minimo:.2f}</text>'
        f'<text x="{margen_izquierdo + plot_ancho}" y="{alto - 35}" font-size="12" text-anchor="middle">{maximo:.2f}</text>'
    )
    svg: str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">
<rect width="100%" height="100%" fill="white"/>
<text x="{ancho / 2}" y="35" font-size="22" text-anchor="middle" font-family="Arial">{escape(variable)} - histograma</text>
<line x1="{margen_izquierdo}" y1="{margen_superior + plot_alto}" x2="{margen_izquierdo + plot_ancho}" y2="{margen_superior + plot_alto}" stroke="#333"/>
<line x1="{margen_izquierdo}" y1="{margen_superior}" x2="{margen_izquierdo}" y2="{margen_superior + plot_alto}" stroke="#333"/>
{''.join(barras)}
<text x="{margen_izquierdo - 12}" y="{margen_superior + 5}" font-size="12" text-anchor="end">{max_conteo}</text>
<text x="{margen_izquierdo - 12}" y="{margen_superior + plot_alto}" font-size="12" text-anchor="end">0</text>
{etiquetas_x}
<text x="{ancho / 2}" y="{alto - 10}" font-size="14" text-anchor="middle">Valor</text>
<text x="20" y="{alto / 2}" font-size="14" text-anchor="middle" transform="rotate(-90 20 {alto / 2})">Frecuencia</text>
</svg>'''
    ruta_salida.write_text(svg, encoding="utf-8")


def guardar_boxplot_svg(
    serie: pd.Series,
    variable: str,
    resumen_variable: pd.Series,
    ruta_salida: Path,
) -> None:
    """Guarda un boxplot SVG horizontal con limites IQR y puntos extremos."""
    valores: pd.Series = pd.to_numeric(serie, errors="raise").dropna()
    minimo: float = float(resumen_variable["minimo"])
    maximo: float = float(resumen_variable["maximo"])
    q1: float = float(resumen_variable["q1"])
    mediana: float = float(resumen_variable["mediana"])
    q3: float = float(resumen_variable["q3"])
    limite_inferior: float = float(resumen_variable["limite_inferior"])
    limite_superior: float = float(resumen_variable["limite_superior"])
    whisker_min: float = max(minimo, limite_inferior)
    whisker_max: float = min(maximo, limite_superior)
    outliers: list[float] = [
        float(valor)
        for valor in valores[(valores < limite_inferior) | (valores > limite_superior)]
    ]

    ancho: int = 900
    alto: int = 340
    inicio: int = 90
    fin: int = 840
    eje_y: int = 170
    caja_alto: int = 80

    x_min: float = escalar_valor(whisker_min, minimo, maximo, inicio, fin)
    x_max: float = escalar_valor(whisker_max, minimo, maximo, inicio, fin)
    x_q1: float = escalar_valor(q1, minimo, maximo, inicio, fin)
    x_q3: float = escalar_valor(q3, minimo, maximo, inicio, fin)
    x_mediana: float = escalar_valor(mediana, minimo, maximo, inicio, fin)

    puntos_outliers: list[str] = []
    for valor in outliers:
        x: float = escalar_valor(valor, minimo, maximo, inicio, fin)
        puntos_outliers.append(f'<circle cx="{x:.2f}" cy="{eje_y}" r="4" fill="#e45756" opacity="0.75"/>')

    svg: str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">
<rect width="100%" height="100%" fill="white"/>
<text x="{ancho / 2}" y="35" font-size="22" text-anchor="middle" font-family="Arial">{escape(variable)} - boxplot IQR</text>
<line x1="{inicio}" y1="{eje_y}" x2="{fin}" y2="{eje_y}" stroke="#333"/>
<line x1="{x_min:.2f}" y1="{eje_y - 35}" x2="{x_min:.2f}" y2="{eje_y + 35}" stroke="#333"/>
<line x1="{x_max:.2f}" y1="{eje_y - 35}" x2="{x_max:.2f}" y2="{eje_y + 35}" stroke="#333"/>
<rect x="{x_q1:.2f}" y="{eje_y - caja_alto / 2}" width="{max(x_q3 - x_q1, 1):.2f}" height="{caja_alto}" fill="#72b7b2" stroke="#333"/>
<line x1="{x_mediana:.2f}" y1="{eje_y - caja_alto / 2}" x2="{x_mediana:.2f}" y2="{eje_y + caja_alto / 2}" stroke="#111" stroke-width="3"/>
{''.join(puntos_outliers)}
<text x="{inicio}" y="{alto - 55}" font-size="12" text-anchor="middle">{minimo:.2f}</text>
<text x="{fin}" y="{alto - 55}" font-size="12" text-anchor="middle">{maximo:.2f}</text>
<text x="{x_q1:.2f}" y="{eje_y - 55}" font-size="12" text-anchor="middle">Q1 {q1:.2f}</text>
<text x="{x_mediana:.2f}" y="{eje_y + 72}" font-size="12" text-anchor="middle">Mediana {mediana:.2f}</text>
<text x="{x_q3:.2f}" y="{eje_y - 55}" font-size="12" text-anchor="middle">Q3 {q3:.2f}</text>
<text x="{ancho / 2}" y="{alto - 15}" font-size="13" text-anchor="middle">Puntos rojos: posibles outliers segun limites IQR</text>
</svg>'''
    ruta_salida.write_text(svg, encoding="utf-8")


def generar_visualizaciones_outliers(
    df: pd.DataFrame,
    resumen_outliers: pd.DataFrame,
    variables: list[str],
    ruta_figuras: Path = RUTA_FIGURAS,
) -> list[Path]:
    """Genera histogramas y boxplots para variables numericas susceptibles.

    Las figuras se guardan como SVG para que puedan abrirse en navegador, VS
    Code o incluirse en la memoria sin depender de librerias graficas externas.
    """
    ruta_figuras.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []

    for variable in variables:
        columna_imputacion: str = f"{variable}_imputada"
        if columna_imputacion in df.columns:
            serie_visualizacion: pd.Series = df.loc[df[columna_imputacion].eq(0), variable]
        else:
            serie_visualizacion = df[variable]

        resumen_variable: pd.Series = resumen_outliers.loc[
            resumen_outliers["variable"] == variable
        ].iloc[0]
        ruta_histograma: Path = ruta_figuras / f"histograma_{variable}.svg"
        ruta_boxplot: Path = ruta_figuras / f"boxplot_{variable}.svg"

        guardar_histograma_svg(serie_visualizacion, variable, ruta_histograma)
        guardar_boxplot_svg(serie_visualizacion, variable, resumen_variable, ruta_boxplot)

        rutas.extend([ruta_histograma, ruta_boxplot])

    return rutas


def validar_calidad_dataset_final(
    integrado: pd.DataFrame,
    tipado: pd.DataFrame,
) -> pd.DataFrame:
    """Genera un diagnostico de limpieza adicional para el apartado 3.4.

    La validacion revisa duplicados, coherencia de variables derivadas, rangos
    numericos, categorias simplificadas, codificacion binaria y posible fuga de
    informacion. La funcion no modifica ningun valor del dataset: solo devuelve
    un informe con el resultado y la decision metodologica tomada.
    """
    filas: list[dict[str, str | int]] = []

    def agregar(
        bloque: str,
        comprobacion: str,
        resultado: str | int,
        decision: str,
        justificacion: str,
    ) -> None:
        filas.append(
            {
                "bloque": bloque,
                "comprobacion": comprobacion,
                "resultado": resultado,
                "decision": decision,
                "justificacion": justificacion,
            }
        )

    duplicados_completos: int = int(integrado.duplicated().sum())
    agregar(
        "Duplicados",
        "Filas duplicadas completas en el dataset integrado",
        duplicados_completos,
        "No eliminar registros",
        "No hay duplicados completos; no existe evidencia para borrar filas.",
    )

    duplicados_url: int = int(integrado["url_ficha"].duplicated().sum())
    agregar(
        "Duplicados",
        "Duplicados por url_ficha",
        duplicados_url,
        "No eliminar registros",
        "La URL de ficha actua como identificador tecnico y no aparece repetida.",
    )

    duplicados_nombre: int = int(integrado["nombre"].duplicated().sum())
    agregar(
        "Duplicados",
        "Duplicados por nombre",
        duplicados_nombre,
        "No eliminar automaticamente",
        "El nombre por si solo no identifica inequivocamente una biblioteca; puede haber sedes o denominaciones similares.",
    )

    coherencia_web: int = int(
        (integrado["tiene_web"] != integrado["pagina_web"].notna().astype(int)).sum()
    )
    agregar(
        "Coherencia derivadas",
        "tiene_web frente a pagina_web informada",
        coherencia_web,
        "Sin correccion",
        "La variable derivada es coherente si el numero de discrepancias es cero.",
    )

    coherencia_catalogo: int = int(
        (integrado["tiene_catalogo"] != integrado["acceso_catalogo"].notna().astype(int)).sum()
    )
    agregar(
        "Coherencia derivadas",
        "tiene_catalogo frente a acceso_catalogo informado",
        coherencia_catalogo,
        "Sin correccion",
        "La variable objetivo reproduce la presencia o ausencia del enlace de catalogo.",
    )

    coherencia_anio: int = int(
        (
            integrado["anio_fundacion_missing"]
            != integrado["anio_fundacion"].isna().astype(int)
        ).sum()
    )
    agregar(
        "Coherencia derivadas",
        "anio_fundacion_missing frente a anio_fundacion",
        coherencia_anio,
        "Sin correccion",
        "El indicador de ausencia del ano de fundacion coincide con los nulos originales.",
    )

    for variable in VARIABLES_BINARIAS:
        valores: set[int] = set(tipado[variable].dropna().astype(int).unique())
        no_validos: set[int] = valores - {0, 1}
        agregar(
            "Codificacion binaria",
            f"Valores permitidos en {variable}",
            ", ".join(map(str, sorted(valores))),
            "Sin correccion" if not no_validos else "Revisar codificacion",
            "La variable solo debe contener 0 y 1; no se detectan codigos inesperados."
            if not no_validos
            else f"Se detectan valores no validos: {sorted(no_validos)}.",
        )

    reglas_rango: dict[str, tuple[int | float, str]] = {
        "antiguedad_biblioteca": (0, "La antiguedad no puede ser negativa."),
        "bibliotecas_municipio": (1, "Debe existir al menos una biblioteca por municipio registrado."),
        "poblacion_municipio": (0, "La poblacion municipal debe ser positiva."),
        "densidad_hab_km2": (0, "La densidad municipal debe ser positiva."),
    }
    for variable, (limite, justificacion) in reglas_rango.items():
        if variable == "bibliotecas_municipio":
            invalidos: int = int((tipado[variable] < limite).sum())
        else:
            invalidos = int((tipado[variable] <= limite).sum())
        agregar(
            "Rangos numericos",
            f"Valores fuera de rango en {variable}",
            invalidos,
            "Sin correccion",
            f"{justificacion} No se detectan incumplimientos si el resultado es cero.",
        )

    for variable, esperadas in CATEGORIAS_ESPERADAS.items():
        observadas: set[str] = set(tipado[variable].dropna().astype(str).unique())
        inesperadas: set[str] = observadas - esperadas
        agregar(
            "Categorias simplificadas",
            f"Categorias inesperadas en {variable}",
            ", ".join(sorted(inesperadas)) if inesperadas else "0",
            "Sin correccion" if not inesperadas else "Revisar reglas de agrupacion",
            "Las categorias simplificadas coinciden con la agrupacion definida."
            if not inesperadas
            else "Existen categorias no previstas por las reglas de simplificacion.",
        )

    for variable in VARIABLES_CATEGORICAS_SIMPLIFICADAS:
        nulos: int = int(tipado[variable].isna().sum())
        agregar(
            "Categorias simplificadas",
            f"Nulos en {variable}",
            nulos,
            "Sin correccion",
            "La simplificacion no debe generar valores ausentes.",
        )

    agregar(
        "Redundancia",
        "Variables categoricas originales y simplificadas",
        "Se conservan ambas versiones",
        "Usar simplificadas en modelado",
        "Las originales se mantienen para trazabilidad; las simplificadas reducen cardinalidad y redundancia.",
    )

    agregar(
        "Fuga de informacion",
        "Presencia de acceso_catalogo en dataset tipado",
        "No incluida" if "acceso_catalogo" not in tipado.columns else "Incluida",
        "Sin correccion" if "acceso_catalogo" not in tipado.columns else "Eliminar antes de modelar",
        "acceso_catalogo no debe entrar en el modelo porque de esa variable deriva tiene_catalogo.",
    )

    agregar(
        "Fuga de informacion",
        "Uso de tiene_web",
        "Incluida como explicativa",
        "Mantener con cautela interpretativa",
        "No deriva del catalogo, pero representa presencia digital general y puede estar asociada al objetivo.",
    )

    return pd.DataFrame(filas)


def nombre_columna_excel(indice: int) -> str:
    """Convierte un indice de columna a notacion Excel."""
    nombre: str = ""
    while indice:
        indice, resto = divmod(indice - 1, 26)
        nombre = chr(65 + resto) + nombre
    return nombre


def celda_excel_xml(fila: int, columna: int, valor: object, estilo: int | None = None) -> str:
    """Crea una celda XML de tipo texto para un archivo XLSX sencillo."""
    referencia: str = f"{nombre_columna_excel(columna)}{fila}"
    atributo_estilo: str = f' s="{estilo}"' if estilo is not None else ""
    texto: str = escape(str(valor))
    return f'<c r="{referencia}" t="inlineStr"{atributo_estilo}><is><t>{texto}</t></is></c>'


def guardar_dataframe_xlsx(df: pd.DataFrame, ruta_salida: Path, nombre_hoja: str) -> None:
    """Guarda un DataFrame en formato XLSX sin dependencias externas."""
    cabecera: list[str] = list(df.columns)
    filas: list[list[object]] = [cabecera] + df.astype(object).values.tolist()
    filas_xml: list[str] = []

    for indice_fila, fila in enumerate(filas, start=1):
        estilo: int | None = 1 if indice_fila == 1 else None
        celdas: str = "".join(
            celda_excel_xml(indice_fila, indice_columna, valor, estilo)
            for indice_columna, valor in enumerate(fila, start=1)
        )
        filas_xml.append(f'<row r="{indice_fila}">{celdas}</row>')

    ultima_celda: str = f"{nombre_columna_excel(len(cabecera))}{len(filas)}"
    worksheet: str = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:{ultima_celda}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols><col min="1" max="{len(cabecera)}" width="36" customWidth="1"/></cols>
  <sheetData>{''.join(filas_xml)}</sheetData>
  <autoFilter ref="A1:{ultima_celda}"/>
</worksheet>'''
    workbook: str = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="{escape(nombre_hoja)}" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    styles: str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
    content_types: str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''
    rels_root: str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    rels_workbook: str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    with ZipFile(ruta_salida, "w", ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", rels_root)
        xlsx.writestr("xl/workbook.xml", workbook)
        xlsx.writestr("xl/_rels/workbook.xml.rels", rels_workbook)
        xlsx.writestr("xl/worksheets/sheet1.xml", worksheet)
        xlsx.writestr("xl/styles.xml", styles)


def main() -> None:
    dataset_modelo: pd.DataFrame = cargar_dataset_modelo()
    dataset_limpio, medianas = generar_dataset_modelo_limpio(dataset_modelo)
    dataset_tipado: pd.DataFrame = gestionar_tipos_atributos(dataset_limpio)
    dataset_integrado: pd.DataFrame = pd.read_csv(RUTA_INTEGRADO)

    print(f"Dataset base analizado: {RUTA_MODELO_BASE}")
    print(f"Dataset limpio guardado en: {RUTA_MODELO_LIMPIO}")
    print(f"Dataset tipado guardado en: {RUTA_MODELO_TIPADO}")
    print(f"Registros: {dataset_limpio.shape[0]}")
    print(f"Columnas dataset limpio: {dataset_limpio.shape[1]}")
    print(f"Columnas dataset tipado: {dataset_tipado.shape[1]}")
    print("Medianas utilizadas en la imputacion:")
    for variable, mediana in medianas.items():
        print(f"- {variable}: {mediana:.2f}")
    print("Nulos restantes por columna:")
    print(dataset_limpio.isna().sum().to_string())
    print("Categorias simplificadas:")
    for variable in VARIABLES_CATEGORICAS_SIMPLIFICADAS:
        print(f"- {variable}: {dataset_tipado[variable].nunique()} categorias")

    resumen_outliers: pd.DataFrame = calcular_resumen_outliers_iqr(
        dataset_tipado,
        VARIABLES_OUTLIERS,
    )
    rutas_figuras: list[Path] = generar_visualizaciones_outliers(
        dataset_tipado,
        resumen_outliers,
        VARIABLES_OUTLIERS,
    )
    print("Resumen de posibles valores extremos por IQR:")
    print(resumen_outliers.to_string(index=False))
    print("Figuras generadas:")
    for ruta in rutas_figuras:
        print(f"- {ruta}")

    diagnostico_3_4: pd.DataFrame = validar_calidad_dataset_final(
        dataset_integrado,
        dataset_tipado,
    )
    guardar_dataframe_xlsx(diagnostico_3_4, RUTA_DIAGNOSTICO_3_4, "Limpieza 3.4")
    print(f"Diagnostico adicional 3.4 guardado en: {RUTA_DIAGNOSTICO_3_4}")
    print("Resumen de validaciones adicionales:")
    print(diagnostico_3_4.to_string(index=False))


if __name__ == "__main__":
    main()
