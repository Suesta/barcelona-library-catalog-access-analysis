# PRACT2 - Limpieza y análisis del dataset de bibliotecas de Barcelona

## Integrantes del grupo

- Víctor Suesta Arribas
- Antonio José Paredes de la Sal

## Objetivo de la práctica

Este repositorio contiene el trabajo realizado para la Práctica 2 de la asignatura **Tipología y ciclo de vida de los datos**.

El objetivo de esta práctica es continuar el trabajo iniciado en la Práctica 1, aplicando procesos de integración, selección, limpieza, validación, análisis y visualización sobre el dataset de bibliotecas de la provincia de Barcelona obtenido mediante web scraping.

La pregunta principal del análisis es:

**¿Qué características territoriales e institucionales se asocian con que una biblioteca disponga de acceso a catálogo en línea?**

Para ello se ha definido como variable objetivo `tiene_catalogo`, creada a partir de la variable original `acceso_catalogo`.

## Estructura del repositorio

El repositorio se organiza en tres carpetas principales:

- `dataset/`: contiene los datasets originales, intermedios y finales.
- `source/`: contiene los scripts utilizados para la integración, limpieza y análisis.
- `docs/`: contiene borradores de memoria, diagnósticos, resultados y figuras.

## Archivos incluidos

### Carpeta `dataset/`

- `bibliotecas_barcelona_original.csv`: dataset original obtenido en la Práctica 1.
- `municipios_idescat_mas_20000_2025.csv`: tabla municipal externa usada para enriquecer el dataset.
- `bibliotecas_barcelona_integrado_provisional.csv`: dataset tras la integración inicial.
- `bibliotecas_barcelona_modelo_base.csv`: dataset con la selección inicial de variables para el modelo.
- `bibliotecas_barcelona_modelo_limpio.csv`: dataset tras la gestión inicial de valores ausentes.
- `bibliotecas_barcelona_modelo_tipado.csv`: dataset tras la gestión de tipos y simplificación de variables categóricas.
- `bibliotecas_barcelona_final_analizado.csv`: dataset final preparado para la entrega.

### Carpeta `source/`

- `01_integracionyseleccion.py`: genera el dataset integrado y selecciona las variables iniciales de análisis.
- `02_limpieza.py`: gestiona valores ausentes, tipos de datos, categorías, duplicados y valores extremos.
- `03_analisis.py`: aplica el modelo supervisado, el modelo no supervisado y el contraste de hipótesis.

### Carpeta `docs/`

- `borrador_apartados_1_2_3.md`: borrador de los apartados iniciales de la memoria.
- `diagnostico_limpieza_3_1.xlsx`: diagnóstico de nulos, ceros y valores vacíos.
- `diagnostico_tipos_3_2.xlsx`: diagnóstico y decisiones sobre tipos de datos.
- `diagnostico_limpieza_3_4.xlsx`: validaciones adicionales de limpieza.
- `resultados_modelo_supervisado.xlsx`: métricas y resultados del modelo supervisado.
- `resultados_modelo_no_supervisado.xlsx`: resultados del modelo no supervisado.
- `contraste_hipotesis_4_2.xlsx`: resultados del contraste chi-cuadrado.
- `figures/`: carpeta con las figuras generadas durante la limpieza y el análisis.

## Ejecución del código

Para reproducir el proyecto desde la raíz del repositorio:

```bash
pip install -r requirements.txt
python source/01_integracionyseleccion.py
python source/02_limpieza.py
python source/03_analisis.py
```

## Flujo de trabajo

El flujo seguido en la práctica es el siguiente:

1. Cargar el dataset original de bibliotecas obtenido en la Práctica 1.
2. Integrar información municipal externa procedente de Idescat.
3. Crear variables derivadas útiles para el análisis.
4. Seleccionar las variables de interés.
5. Gestionar valores ausentes y tipos de datos.
6. Simplificar variables categóricas.
7. Revisar duplicados, coherencia interna y valores extremos.
8. Aplicar un modelo supervisado para analizar la disponibilidad de catálogo en línea.
9. Aplicar un modelo no supervisado para identificar grupos de bibliotecas.
10. Aplicar un contraste chi-cuadrado entre tipología de biblioteca y disponibilidad de catálogo.
11. Generar tablas y figuras para interpretar los resultados.

## Variables principales creadas

* `tiene_web`: indica si la biblioteca tiene página web informada.
* `tiene_catalogo`: indica si la biblioteca tiene acceso a catálogo en línea.
* `anio_fundacion_missing`: indica si falta el año de fundación.
* `anio_fundacion_num`: versión numérica del año de fundación.
* `antiguedad_biblioteca`: antigüedad aproximada de la biblioteca.
* `bibliotecas_municipio`: número de bibliotecas del dataset en el mismo municipio.
* `poblacion_municipio`: población municipal procedente de Idescat, cuando está disponible.
* `densidad_hab_km2`: densidad de población municipal.
* `tiene_poblacion_externa`: indica si el registro se ha podido enriquecer con información municipal externa.
* `bibliotecas_por_10000_habitantes`: tasa de bibliotecas por cada 10.000 habitantes.

## Modelos y análisis realizados

En el análisis se han aplicado tres procedimientos principales:

1. **Modelo supervisado**: regresión logística para estudiar la disponibilidad de catálogo en línea.
2. **Modelo no supervisado**: K-Means para identificar perfiles de bibliotecas.
3. **Contraste de hipótesis**: prueba chi-cuadrado para analizar la asociación entre tipología de biblioteca y disponibilidad de catálogo.

## Nota sobre la integración externa

La tabla municipal externa utilizada contiene municipios catalanes de más de 20.000 habitantes. Por este motivo, no todos los municipios del dataset original tienen información poblacional integrada. Esta limitación se conserva explícitamente mediante la variable `tiene_poblacion_externa` y se gestiona posteriormente en la fase de limpieza.

## Dataset final

El dataset final preparado para la entrega se encuentra en:

```text
dataset/bibliotecas_barcelona_final_analizado.csv
```
