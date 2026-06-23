# Barcelona Library Catalog Access Analysis

Data cleaning, feature engineering and statistical analysis project focused on understanding which institutional and territorial factors are associated with online catalog availability across libraries in the province of Barcelona.

## Project Overview

This project analyzes a dataset of libraries located in Barcelona province. Each record represents one library and includes information such as municipality, library type, ownership, management model, website availability, online catalog access, foundation year and additional municipal indicators.

The main analytical question is:

> Which institutional and territorial characteristics are associated with a library having online catalog access?

To answer this question, the original dataset was cleaned, enriched with municipal-level information, transformed into a modeling-ready dataset and analyzed using supervised learning, unsupervised learning and statistical hypothesis testing.

## Main Techniques Used

- Data cleaning and validation
- Missing value analysis
- Median imputation
- Feature engineering
- Categorical variable simplification
- Outlier detection using the IQR method
- Logistic Regression
- K-Means clustering
- Chi-square hypothesis testing
- Data visualization
- Reproducible Python workflow

## Repository Structure

```text
barcelona-library-catalog-access-analysis/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── dataset/
│   ├── bibliotecas_barcelona_original.csv
│   ├── municipios_idescat_mas_20000_2025.csv
│   ├── bibliotecas_barcelona_integrado_provisional.csv
│   ├── bibliotecas_barcelona_modelo_base.csv
│   ├── bibliotecas_barcelona_modelo_limpio.csv
│   ├── bibliotecas_barcelona_modelo_tipado.csv
│   └── bibliotecas_barcelona_final_analizado.csv
│
├── source/
│   ├── 01_integracionyseleccion.py
│   ├── 02_limpieza.py
│   └── 03_analisis.py
│
└── docs/
    └── figures/
        ├── boxplot_antiguedad_biblioteca.svg
        ├── boxplot_bibliotecas_municipio.svg
        ├── boxplot_densidad_hab_km2.svg
        ├── boxplot_poblacion_municipio.svg
        ├── clusters_kmeans_pca.svg
        ├── coeficientes_logistica.svg
        ├── contraste_catalogo_tipologia.svg
        ├── histograma_antiguedad_biblioteca.svg
        ├── histograma_bibliotecas_municipio.svg
        ├── histograma_densidad_hab_km2.svg
        ├── histograma_poblacion_municipio.svg
        ├── matriz_confusion_logistica.svg
        └── perfil_clusters_kmeans.svg
```

## Dataset

The project starts from a library dataset containing 519 records from Barcelona province.

The final modeling dataset is available at:

```text
dataset/bibliotecas_barcelona_final_analizado.csv
```

The dataset combines the original library information with additional municipal-level indicators from Idescat.

## Main Features Created

Several variables were created during the feature engineering process:

- `tiene_web`: indicates whether the library has a website.
- `tiene_catalogo`: indicates whether the library has online catalog access.
- `anio_fundacion_missing`: indicates whether the foundation year is missing.
- `anio_fundacion_num`: numeric version of the foundation year.
- `antiguedad_biblioteca`: estimated library age using 2026 as reference year.
- `bibliotecas_municipio`: number of libraries in the same municipality.
- `poblacion_municipio`: municipal population when available.
- `densidad_hab_km2`: population density by municipality.
- `tiene_poblacion_externa`: indicates whether external municipal data was available.
- `bibliotecas_por_10000_habitantes`: number of libraries per 10,000 inhabitants.

## Workflow

The project follows three main steps.

### 1. Data Integration and Feature Selection

Implemented in:

```text
source/01_integracionyseleccion.py
```

This script:

- Loads the original library dataset.
- Adds derived variables.
- Integrates external municipal data.
- Normalizes municipality names before joining.
- Selects the initial variables for modeling.
- Prevents data leakage by excluding variables directly derived from the target.

### 2. Data Cleaning and Validation

Implemented in:

```text
source/02_limpieza.py
```

This script:

- Reviews missing values, empty values and zeros.
- Handles missing numerical values using median imputation.
- Creates imputation indicator variables.
- Validates binary variables.
- Simplifies high-cardinality categorical variables.
- Checks duplicates and internal consistency.
- Detects potential outliers using the IQR method.

### 3. Modeling and Statistical Analysis

Implemented in:

```text
source/03_analisis.py
```

This script applies:

- Logistic Regression as a supervised model.
- K-Means clustering as an unsupervised model.
- Chi-square hypothesis testing to evaluate the association between library type and catalog availability.
- Visualizations for model interpretation and exploratory analysis.

## Modeling Approach

### Supervised Learning

A Logistic Regression model was used to predict whether a library has online catalog access.

The target variable was:

```text
tiene_catalogo
```

The model used numerical, binary and simplified categorical variables. Categorical features were encoded using one-hot encoding, and numerical features were standardized.

Main evaluation metrics:

```text
Accuracy: 0.756
F1 Macro: 0.686
ROC AUC: 0.887
```

The model showed a reasonable ability to distinguish between libraries with and without online catalog access.

### Unsupervised Learning

K-Means clustering was used to identify library profiles without using the target variable during training.

Different values of `k` were tested, and `k = 3` was selected based on the silhouette score.

The clustering results suggested three broad profiles:

1. A small group of older and more specialized libraries.
2. A large group mainly composed of public local libraries.
3. A more urban and specialized group, especially concentrated in dense municipalities.

### Hypothesis Testing

A chi-square test was used to analyze the association between library type and online catalog availability.

The test showed a statistically significant association between both variables.

Main result:

```text
Chi-square statistic: 117.370
p-value: 3.26e-26
Cramer's V: 0.476
```

This suggests that catalog availability is not evenly distributed across library types.

## Key Insights

- Public and university libraries show very high online catalog availability.
- Specialized libraries present a more heterogeneous situation.
- Website availability is positively associated with online catalog access.
- Library type is one of the most relevant factors in explaining catalog availability.
- External municipal data adds useful territorial context, although it is only available for municipalities above 20,000 inhabitants.

## Visual Outputs

The folder `docs/figures/` contains the main visual outputs generated during the analysis, including:

- Histograms and boxplots for numerical variables.
- Logistic Regression confusion matrix.
- Logistic Regression coefficient plot.
- K-Means PCA visualization.
- Cluster profile visualization.
- Catalog availability by library type.

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/Suesta/barcelona-library-catalog-access-analysis.git
cd barcelona-library-catalog-access-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the scripts in order:

```bash
python source/01_integracionyseleccion.py
python source/02_limpieza.py
python source/03_analisis.py
```

## Dependencies

The main Python libraries used are:

- pandas
- scikit-learn
- scipy

Exact versions are listed in:

```text
requirements.txt
```

## Data Sources

The dataset was built from institutional open data sources:

- Spanish Ministry of Culture: Spanish libraries directory.
- Idescat: municipal indicators for Catalonia.

## Notes

The external municipal dataset only includes Catalan municipalities with more than 20,000 inhabitants. For this reason, not all libraries could be enriched with population and density indicators. This limitation is explicitly tracked using the variable `tiene_poblacion_externa`.

## Author

Víctor Suesta Arribas

Mathematics graduate and Data Science master's student, focused on data analysis, machine learning and applied data projects.
