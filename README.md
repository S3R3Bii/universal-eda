# Universal EDA Pipeline

`universal_eda_pipeline.py` contains one copy-pasteable Python function for exploratory data analysis, preprocessing, report generation, and early modeling diagnostics.

The function is intended for quick starts across analytics, data science, machine learning, and business intelligence projects.

## What It Does

`universal_eda_pipeline(...)` can:

- Load a `pandas.DataFrame`, CSV file, or Excel file.
- Run basic EDA: shape, dtypes, missingness, duplicates, descriptive stats, unique counts, inferred schema, histograms, density plots, boxplots, count plots, missingness plots, correlation heatmaps, pairplots, and target analysis.
- Ask data-driven questions after the basic EDA when `interactive=True`.
- Automatically handle non-numeric columns with safe one-hot encoding, frequency encoding for high-cardinality columns, datetime feature extraction, and text length features.
- Impute missing values with practical defaults, including KNN/median fallbacks.
- Detect outliers and either flag, cap, remove, or leave them depending on the selected mode.
- Run optional PCA, clustering diagnostics, VIF checks, normality/skewness checks, and quick baseline models.
- Return cleaned data, transformed data, fitted preprocessing objects, summaries, warnings, and report metadata.
- Render results inline in a notebook or save a formal PDF report.

## Dependencies

Install the common data science stack:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```

`statsmodels` is optional. If it is missing, VIF checks are skipped.

## Basic Import

```python
from universal_eda_pipeline import universal_eda_pipeline
```

## Use Case 1: Notebook-Only EDA

Best when working in Jupyter, Colab, or VS Code notebooks and you want results printed/rendered inline instead of saved as a PDF.

```python
result = universal_eda_pipeline(
    df,
    target_col="target",
    notebook_mode=True,
)
```

With `notebook_mode=True`, PDF creation is skipped by default. The returned `report_path` will be `None`.

## Use Case 2: Notebook Output Plus PDF

Use this when you want inline notebook output and a saved formal report.

```python
result = universal_eda_pipeline(
    df,
    target_col="target",
    notebook_mode=True,
    create_pdf=True,
    output_dir="eda_output",
    report_name="project_eda_report.pdf",
)
```

## Use Case 3: PDF Report From a CSV

Best for a script or VS Code workflow where you want a saved report.

```python
result = universal_eda_pipeline(
    "data/customer_churn.csv",
    target_col="churn",
    problem_type="classification",
    output_dir="eda_output",
    report_name="churn_eda_report.pdf",
)
```

The PDF path is available here:

```python
result["report_path"]
```

## Use Case 4: Fully Automatic Run

Use this when you do not want prompts.

```python
result = universal_eda_pipeline(
    df,
    target_col="target",
    fully_auto=True,
    interactive=False,
)
```

This uses conservative automatic decisions for missing values, outliers, encoding, scaling, PCA, and baseline modeling.

## Use Case 5: Interactive Decision Checkpoints

Use this when you want the function to inspect the data first and then ask what to do.

```python
result = universal_eda_pipeline(
    df,
    target_col="sale_price",
    problem_type="regression",
    interactive=True,
    fully_auto=False,
)
```

After basic EDA, the function may ask questions such as:

- Should missing values be imputed, dropped, or left unchanged?
- Should outliers be flagged, capped, removed, or left unchanged?
- Should likely ID columns be dropped?
- Should datetime columns be converted into date parts?
- Should skewed numeric variables be transformed?
- Should PCA, clustering diagnostics, or baseline models be run?

It prefers pop-up prompts through `tkinter` when available and falls back to terminal input.

## Use Case 6: Classification Project

```python
result = universal_eda_pipeline(
    df,
    target_col="defaulted",
    problem_type="classification",
    notebook_mode=True,
)
```

For classification, the baseline model section reports:

- Accuracy
- Weighted F1
- Classification report
- Confusion matrix
- Feature importance

## Use Case 7: Regression Project

```python
result = universal_eda_pipeline(
    df,
    target_col="price",
    problem_type="regression",
    notebook_mode=True,
)
```

For regression, the baseline model section reports:

- RMSE
- MAE
- R-squared
- Feature importance

## Use Case 8: Clustering or Unsupervised EDA

When there is no target column:

```python
result = universal_eda_pipeline(
    df,
    problem_type="clustering",
    notebook_mode=True,
    run_clustering=True,
)
```

The function can produce:

- PCA explained variance
- PCA 2D projection
- KMeans elbow diagnostics
- Silhouette diagnostics

## Use Case 9: Large Datasets

For expensive plots and diagnostics, the function samples automatically.

```python
result = universal_eda_pipeline(
    "large_dataset.csv",
    target_col="target",
    max_plot_rows=5000,
    sample_count=3,
    fully_auto=True,
    interactive=False,
)
```

Dataset size buckets:

- Small: under 10,000 rows
- Medium: 10,000 to 100,000 rows
- Large: over 100,000 rows

For very large data, multiple samples may be used for clustering diagnostics.

## Use Case 10: Save Cleaned and Transformed CSVs

```python
result = universal_eda_pipeline(
    df,
    target_col="target",
    save_cleaned_csv=True,
    output_dir="eda_output",
)
```

Returned paths:

```python
result["cleaned_csv_path"]
result["transformed_csv_path"]
```

## Important Outputs

The function returns a dictionary.

```python
result.keys()
```

Important entries:

```python
cleaned_df = result["cleaned_df"]
transformed_df = result["transformed_df"]
preprocessing_objects = result["preprocessing_objects"]
eda_summary = result["eda_summary"]
missingness_summary = result["missingness_summary"]
outlier_summary = result["outlier_summary"]
modeling_summary = result["modeling_summary"]
pca_summary = result["pca_summary"]
clustering_summary = result["clustering_summary"]
warnings = result["warnings"]
```

## Cleaned vs Transformed Data

`cleaned_df` is easier to inspect. It keeps the data closer to the original form while applying selected cleaning steps.

`transformed_df` is model-oriented. It is numeric, encoded, imputed, and optionally scaled.

## Leakage Caution

The returned `transformed_df` is fitted on the dataset passed into the function. That is useful for EDA, but do not treat it as a final leakage-safe train/test modeling pipeline if the input includes holdout or future rows.

For final modeling:

- Split train/test first.
- Fit preprocessing on the training data only.
- Apply the fitted preprocessing to validation/test/future data.

The quick baseline model section follows this principle internally by fitting preprocessing on its training split only.

## Recommended Defaults

For notebooks:

```python
result = universal_eda_pipeline(
    df,
    target_col="target",
    notebook_mode=True,
)
```

For script-based PDF reports:

```python
result = universal_eda_pipeline(
    "data.csv",
    target_col="target",
    fully_auto=True,
    interactive=False,
    output_dir="eda_output",
)
```

For maximum control:

```python
result = universal_eda_pipeline(
    df,
    target_col="target",
    interactive=True,
    fully_auto=False,
    notebook_mode=True,
    create_pdf=True,
)
```

## Scope

Independent exploratory tooling. Fit preprocessing using training rows only when evaluating future or held-out data. No current performance or production-readiness claim is made.

[Project gallery](https://github.com/S3R3Bii/project-portfolio)

## Current validation

On 28 October 2026, the source passed Python syntax compilation and a 40-row mixed-type smoke run with interactive prompts, PDF output and optional modeling disabled. The returned report structure was checked and the input DataFrame remained unchanged. This check does not validate every optional modeling or rendering path. For predictive studies, keep this exploratory output separate from the train-only preprocessing used in evaluation.
