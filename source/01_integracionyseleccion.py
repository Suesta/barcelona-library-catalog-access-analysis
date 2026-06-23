from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


RUTA_BASE: Path = Path(__file__).resolve().parents[1]
RUTA_DATASET: Path = RUTA_BASE / "dataset"
RUTA_ORIGINAL: Path = RUTA_DATASET / "bibliotecas_barcelona_original.csv"
RUTA_MUNICIPIOS: Path = RUTA_DATASET / "municipios_idescat_mas_20000_2025.csv"
RUTA_SALIDA: Path = RUTA_DATASET / "bibliotecas_barcelona_integrado_provisional.csv"
RUTA_MODELO_BASE: Path = RUTA_DATASET / "bibliotecas_barcelona_modelo_base.csv"
ANIO_REFERENCIA: int = 2026


# Variables seleccionadas para el modelo supervisado.
#
# La seleccion conserva la variable objetivo y variables explicativas con
# relacion conceptual con la disponibilidad de catalogo: presencia digital,
# caracteristicas institucionales y contexto territorial. Se excluyen
# identificadores, URL, metadatos, variables constantes y campos redundantes.
VARIABLES_MODELO: list[str] = [
    "tiene_catalogo",
    "tiene_web",
    "tipologia",
    "titularidad",
    "gestion",
    "antiguedad_biblioteca",
    "anio_fundacion_missing",
    "bibliotecas_municipio",
    "poblacion_municipio",
    "densidad_hab_km2",
    "tiene_poblacion_externa",
]


MUNICIPIOS_IDESCAT: list[tuple[str, int, float, float, int]] = [
    ("Barcelona", 1713247, 101.4, 16904.3, 1),
    ("Hospitalet de Llobregat, l'", 292161, 12.4, 23561.4, 2),
    ("Terrassa", 232676, 70.2, 3316.4, 3),
    ("Badalona", 230642, 21.2, 10889.6, 4),
    ("Sabadell", 224589, 37.8, 5943.1, 5),
    ("Lleida", 147369, 212.3, 694.2, 6),
    ("Tarragona", 143260, 57.9, 2474.7, 7),
    ("Mataró", 131370, 22.5, 5830.9, 8),
    ("Santa Coloma de Gramenet", 123250, 7.0, 17607.1, 9),
    ("Reus", 111911, 52.8, 2118.7, 10),
    ("Girona", 108352, 39.1, 2769.7, 11),
    ("Sant Cugat del Vallès", 97959, 48.2, 2031.1, 12),
    ("Cornellà de Llobregat", 92255, 7.0, 13198.1, 13),
    ("Sant Boi de Llobregat", 85317, 21.5, 3973.8, 14),
    ("Rubí", 82825, 32.3, 2564.2, 15),
    ("Manresa", 80692, 41.6, 1937.8, 16),
    ("Vilanova i la Geltrú", 71305, 34.0, 2097.8, 17),
    ("Castelldefels", 69689, 12.9, 5414.8, 18),
    ("Viladecans", 67536, 20.4, 3310.6, 19),
    ("Prat de Llobregat, el", 66251, 31.4, 2109.2, 20),
    ("Granollers", 65123, 14.9, 4379.5, 21),
    ("Cerdanyola del Vallès", 58264, 30.6, 1906.5, 22),
    ("Mollet del Vallès", 52759, 10.8, 4898.7, 23),
    ("Vic", 50405, 30.6, 1648.3, 24),
    ("Figueres", 49556, 19.3, 2567.7, 25),
    ("Esplugues de Llobregat", 48424, 4.6, 10527.0, 26),
    ("Gavà", 48181, 30.8, 1566.9, 27),
    ("Sant Feliu de Llobregat", 46688, 11.8, 3949.9, 28),
    ("Vilafranca del Penedès", 42561, 19.7, 2166.0, 29),
    ("Igualada", 42449, 8.1, 5234.2, 30),
    ("Blanes", 42446, 17.7, 2403.5, 31),
    ("Lloret de Mar", 41944, 48.7, 861.1, 32),
    ("Vendrell, el", 41195, 36.8, 1119.4, 33),
    ("Ripollet", 39773, 4.3, 9185.5, 34),
    ("Olot", 39516, 29.0, 1361.2, 35),
    ("Sant Adrià de Besòs", 38980, 3.8, 10204.2, 36),
    ("Montcada i Reixac", 37446, 23.5, 1595.5, 37),
    ("Cambrils", 37068, 35.2, 1052.8, 38),
    ("Tortosa", 36258, 218.5, 165.9, 39),
    ("Sant Joan Despí", 35811, 6.2, 5804.1, 40),
    ("Salt", 34603, 6.6, 5211.3, 41),
    ("Barberà del Vallès", 34031, 8.3, 4095.2, 42),
    ("Sitges", 32690, 43.9, 745.5, 43),
    ("Sant Pere de Ribes", 32613, 40.8, 799.3, 44),
    ("Calafell", 32429, 20.4, 1591.2, 45),
    ("Salou", 31018, 15.1, 2050.1, 46),
    ("Pineda de Mar", 30071, 10.7, 2799.9, 47),
    ("Premià de Mar", 29348, 2.1, 13909.0, 48),
    ("Martorell", 28989, 12.8, 2271.9, 49),
    ("Sant Vicenç dels Horts", 28760, 9.1, 3153.5, 50),
    ("Molins de Rei", 27291, 15.9, 1712.1, 51),
    ("Sant Andreu de la Barca", 27069, 5.5, 4921.6, 52),
    ("Santa Perpètua de Mogoda", 26101, 15.8, 1648.8, 53),
    ("Valls", 25540, 55.3, 462.0, 54),
    ("Castellar del Vallès", 25395, 44.9, 565.5, 55),
    ("Olesa de Montserrat", 25000, 16.6, 1503.3, 56),
    ("Masnou, el", 24727, 3.4, 7294.1, 57),
    ("Palafrugell", 24556, 26.9, 913.5, 58),
    ("Vila-seca", 24158, 21.6, 1116.4, 59),
    ("Sant Feliu de Guíxols", 23060, 16.2, 1420.8, 60),
    ("Amposta", 23047, 138.3, 166.6, 61),
    ("Esparreguera", 22639, 27.4, 826.2, 62),
    ("Manlleu", 21321, 17.2, 1241.8, 63),
    ("Vilassar de Mar", 21234, 4.0, 5308.5, 64),
    ("Sant Just Desvern", 21045, 7.8, 2694.6, 65),
    ("Banyoles", 20887, 11.1, 1890.2, 66),
    ("Franqueses del Vallès, les", 20855, 29.1, 715.7, 67),
    ("Calella", 20775, 8.0, 2596.9, 68),
    ("Roses", 20175, 45.9, 439.4, 69),
    ("Sant Quirze del Vallès", 20175, 14.1, 1433.9, 70),
]


def normalizar_municipio(valor: str) -> str:
    # deja los nombres en una forma comparable aunque haya acentos, articulos o signos distintos.
    texto: str = str(valor).strip()

    articulo_final = re.match(r"^(.*),\s*(l'|el|la|els|les)$", texto, flags=re.IGNORECASE)
    if articulo_final:
        texto = f"{articulo_final.group(2)} {articulo_final.group(1)}"

    texto = texto.lower()
    texto = texto.replace("l' ", "l ").replace("l'", "l ")
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def crear_dataset_municipal() -> pd.DataFrame:
    # se usa una tabla externa sencilla de idescat con municipios catalanes de mas de 20.000 habitantes.
    municipios: pd.DataFrame = pd.DataFrame(
        MUNICIPIOS_IDESCAT,
        columns=[
            "municipio_idescat",
            "poblacion_municipio",
            "superficie_km2",
            "densidad_hab_km2",
            "rango_poblacion_cat",
        ],
    )
    municipios["anio_poblacion"] = 2025
    municipios["fuente_poblacion"] = "Idescat, Censo de población anual del INE"
    municipios["municipio_norm"] = municipios["municipio_idescat"].apply(normalizar_municipio)

    return municipios


def preparar_bibliotecas(df: pd.DataFrame) -> pd.DataFrame:
    # se crean variables derivadas que despues serviran para limpieza y analisis.
    bibliotecas: pd.DataFrame = df.copy()

    bibliotecas["codigo_postal"] = bibliotecas["codigo_postal"].astype(str).str.zfill(5)
    bibliotecas["municipio_norm"] = bibliotecas["municipio"].apply(normalizar_municipio)
    bibliotecas["tiene_web"] = bibliotecas["pagina_web"].notna().astype(int)
    bibliotecas["tiene_catalogo"] = bibliotecas["acceso_catalogo"].notna().astype(int)
    bibliotecas["anio_fundacion_missing"] = bibliotecas["anio_fundacion"].isna().astype(int)
    bibliotecas["anio_fundacion_num"] = pd.to_numeric(bibliotecas["anio_fundacion"], errors="coerce")
    bibliotecas["antiguedad_biblioteca"] = ANIO_REFERENCIA - bibliotecas["anio_fundacion_num"]

    conteo_municipal: pd.Series = bibliotecas.groupby("municipio_norm")["nombre"].transform("count")
    bibliotecas["bibliotecas_municipio"] = conteo_municipal

    return bibliotecas


def integrar_datos() -> pd.DataFrame:
    # se integra la informacion municipal externa por nombre normalizado de municipio.
    bibliotecas_originales: pd.DataFrame = pd.read_csv(RUTA_ORIGINAL)
    bibliotecas: pd.DataFrame = preparar_bibliotecas(bibliotecas_originales)
    municipios: pd.DataFrame = crear_dataset_municipal()

    RUTA_MUNICIPIOS.parent.mkdir(parents=True, exist_ok=True)
    municipios.drop(columns=["municipio_norm"]).to_csv(RUTA_MUNICIPIOS, index=False, encoding="utf-8-sig")

    integrado: pd.DataFrame = bibliotecas.merge(municipios, on="municipio_norm", how="left")
    integrado["tiene_poblacion_externa"] = integrado["poblacion_municipio"].notna().astype(int)
    integrado["bibliotecas_por_10000_habitantes"] = (
        integrado["bibliotecas_municipio"] / integrado["poblacion_municipio"] * 10000
    )

    integrado = integrado.drop(columns=["municipio_norm"])
    integrado.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig")

    return integrado


def generar_dataset_modelo(integrado: pd.DataFrame) -> pd.DataFrame:
    # se crea una subseleccion de variables orientada al objetivo analitico:
    # estudiar que factores se asocian con la disponibilidad de catalogo online.
    #
    # No se incluyen columnas como nombre, direccion o url_ficha porque son
    # identificadores; tampoco acceso_catalogo porque de el se deriva la
    # variable objetivo y produciria fuga de informacion en el modelo.
    columnas_no_encontradas: list[str] = [
        columna for columna in VARIABLES_MODELO if columna not in integrado.columns
    ]
    if columnas_no_encontradas:
        raise ValueError(
            "No se pueden seleccionar las variables del modelo. "
            f"Columnas no encontradas: {columnas_no_encontradas}"
        )

    modelo_base: pd.DataFrame = integrado.loc[:, VARIABLES_MODELO].copy()
    modelo_base.to_csv(RUTA_MODELO_BASE, index=False, encoding="utf-8-sig")

    return modelo_base


def main() -> None:
    integrado: pd.DataFrame = integrar_datos()
    modelo_base: pd.DataFrame = generar_dataset_modelo(integrado)
    registros: int = len(integrado)
    columnas: int = integrado.shape[1]
    cobertura: float = integrado["tiene_poblacion_externa"].mean() * 100

    print(f"Dataset integrado guardado en: {RUTA_SALIDA}")
    print(f"Registros: {registros}")
    print(f"Columnas: {columnas}")
    print(f"Cobertura de poblacion externa: {cobertura:.2f}%")
    print(f"Dataset para modelo base guardado en: {RUTA_MODELO_BASE}")
    print(f"Columnas seleccionadas para modelo: {modelo_base.shape[1]}")


if __name__ == "__main__":
    main()
