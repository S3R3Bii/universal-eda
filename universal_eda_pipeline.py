def universal_eda_pipeline(
    data,
    target_col=None,
    problem_type="auto",
    interactive=True,
    fully_auto=False,
    output_dir="eda_output",
    report_name="eda_report.pdf",
    random_state=42,
    max_plot_rows=5000,
    max_pairplot_features=8,
    max_categories_for_countplot=30,
    one_hot_max_categories=20,
    high_cardinality_threshold=50,
    rare_category_min_freq=0.01,
    imputation_strategy="auto",
    outlier_method="iqr",
    sample_count=3,
    notebook_mode=False,
    create_pdf=None,
    display_plots=False,
    use_popup_prompts=True,
    run_pca="auto",
    run_clustering="auto",
    run_baseline_models="auto",
    run_vif=True,
    save_cleaned_csv=False,
    verbose=True,
):
    """
    Run a reusable end-to-end EDA, preprocessing, and early modeling pipeline.

    Parameters
    ----------
    data : pandas.DataFrame or str/path-like
        Input data. Supports a DataFrame, CSV path, or Excel path.
    target_col : str, optional
        Target/dependent variable for supervised analysis.
    problem_type : {"classification", "regression", "clustering", "auto", None}
        Analysis type. "auto" infers classification/regression when a target is
        supplied, otherwise uses unsupervised/clustering-oriented analysis.
    interactive : bool
        If True, ask data-driven questions after basic EDA.
    fully_auto : bool
        If True, skip questions and use conservative automatic decisions.
    output_dir : str
        Directory where the PDF report and optional cleaned CSVs are saved.
    report_name : str
        PDF report filename.
    random_state : int
        Random seed for sampling, PCA, clustering, and baseline models.
    max_plot_rows : int
        Maximum rows used for expensive plots and diagnostics.
    max_pairplot_features : int
        Maximum numeric features included in pairplots.
    max_categories_for_countplot : int
        Maximum categories shown in categorical count plots.
    one_hot_max_categories : int
        Maximum unique categories for safe one-hot encoding.
    high_cardinality_threshold : int
        Unique category count above which categorical columns are treated as
        high-cardinality.
    rare_category_min_freq : float
        Minimum frequency used when grouping rare categorical levels.
    imputation_strategy : {"auto", "simple", "knn", "iterative", "none"}
        Numeric imputation preference. "auto" uses KNN when practical, else
        median imputation.
    outlier_method : {"iqr", "zscore"}
        Outlier detection method used for reporting and decisions.
    sample_count : int
        Number of random samples used for very large clustering diagnostics.
    notebook_mode : bool
        If True, render report text, tables, and plots inline in a notebook.
        When create_pdf is left as None, notebook_mode=True creates a notebook-
        only report and skips PDF generation.
    create_pdf : bool, optional
        If True, save a PDF report. If None, defaults to False when
        notebook_mode=True and True otherwise.
    display_plots : bool
        If True, show plots interactively as they are generated.
    use_popup_prompts : bool
        If True, prefer tkinter pop-up prompts and fall back to terminal input.
    run_pca, run_clustering, run_baseline_models : {"auto", True, False}
        Controls optional advanced analysis.
    run_vif : bool
        If True, compute VIF when statsmodels is installed and the feature count
        is manageable.
    save_cleaned_csv : bool
        If True, save cleaned and transformed CSV files.
    verbose : bool
        If True, print an EDA summary and important decisions.

    Returns
    -------
    dict
        Dictionary containing cleaned/transformed data, fitted preprocessing
        objects, summaries, user decisions, report path, and warnings.

    Notes
    -----
    This function does not mutate the original DataFrame. The returned
    transformed data is fitted on the provided dataset for exploratory use.
    Baseline modeling uses a separate train/test split and fits preprocessing
    objects on the training split only to reduce data leakage in model metrics.
    """
    import math
    import os
    import re
    import textwrap
    import warnings as py_warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        raise ImportError("matplotlib is required to generate the EDA report.") from exc

    try:
        from IPython import get_ipython
        from IPython.display import Markdown as _NotebookMarkdown
        from IPython.display import display as _notebook_display

        has_ipython_display = get_ipython() is not None
    except Exception:
        _NotebookMarkdown = None
        _notebook_display = None
        has_ipython_display = False

    if create_pdf is None:
        create_pdf = not notebook_mode

    try:
        import seaborn as sns

        has_seaborn = True
        sns.set_theme(style="whitegrid")
    except Exception:
        sns = None
        has_seaborn = False

    try:
        from scipy import stats

        has_scipy = True
    except Exception:
        stats = None
        has_scipy = False

    try:
        from sklearn.base import clone
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            adjusted_rand_score,
            classification_report,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            silhouette_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler

        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
        except Exception:
            IterativeImputer = None
    except Exception as exc:
        raise ImportError("scikit-learn is required for preprocessing and modeling support.") from exc

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        has_statsmodels = True
    except Exception:
        variance_inflation_factor = None
        has_statsmodels = False

    py_warnings.filterwarnings("ignore", category=RuntimeWarning)

    warning_messages = []
    pdf_sections = []
    rng = np.random.default_rng(random_state)

    def _warn(message):
        warning_messages.append(str(message))
        if verbose:
            print(f"WARNING: {message}")

    def _as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in {"y", "yes", "true", "1", "run", "enable"}
        return bool(value)

    def _is_auto(value):
        return isinstance(value, str) and value.strip().lower() == "auto"

    def _safe_filename(name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")

    def _load_data(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.copy(deep=True), "DataFrame"
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path), str(path)
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            return pd.read_excel(path), str(path)
        raise ValueError("Unsupported input type. Use a DataFrame, CSV path, or Excel path.")

    def _sample_df(df, max_rows=max_plot_rows, seed=random_state):
        if len(df) <= max_rows:
            return df.copy()
        return df.sample(n=max_rows, random_state=seed)

    def _clean_numeric_text(series):
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan})
        )
        return cleaned

    def _numeric_conversion_ratio(series):
        non_missing = series.dropna()
        if non_missing.empty:
            return 0.0
        converted = pd.to_numeric(_clean_numeric_text(non_missing), errors="coerce")
        return float(converted.notna().mean())

    def _datetime_conversion_ratio(series):
        non_missing = series.dropna()
        if len(non_missing) < 3:
            return 0.0
        sample = non_missing.astype(str).sample(
            n=min(len(non_missing), 500),
            random_state=random_state,
        )
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean())

    def _likely_problem_type(df, target):
        if target is None or target not in df.columns:
            return "clustering" if problem_type in {None, "auto"} else problem_type
        if problem_type not in {None, "auto"}:
            return str(problem_type).lower()
        y = df[target].dropna()
        if y.empty:
            return "classification"
        unique_count = y.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(y):
            if unique_count <= 20 or unique_count / max(len(y), 1) <= 0.05:
                return "classification"
            return "regression"
        return "classification"

    def _infer_schema(df):
        n_rows = len(df)
        schema = {
            "numeric_cols": [],
            "categorical_cols": [],
            "boolean_cols": [],
            "datetime_cols": [],
            "convertible_numeric_cols": [],
            "likely_datetime_cols": [],
            "likely_text_cols": [],
            "likely_id_cols": [],
            "high_cardinality_cols": [],
            "low_cardinality_categoricals": [],
            "unique_counts": {},
            "missing_pct": {},
            "skewed_numeric_cols": [],
        }

        for col in df.columns:
            s = df[col]
            schema["unique_counts"][col] = int(s.nunique(dropna=True))
            schema["missing_pct"][col] = float(s.isna().mean() * 100)

            if pd.api.types.is_bool_dtype(s):
                schema["boolean_cols"].append(col)
            elif pd.api.types.is_numeric_dtype(s):
                schema["numeric_cols"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(s):
                schema["datetime_cols"].append(col)
            else:
                schema["categorical_cols"].append(col)

            if col == target_col:
                continue

            unique_count = schema["unique_counts"][col]
            unique_ratio = unique_count / max(n_rows, 1)
            name_lower = str(col).lower()

            if unique_count > 0 and unique_ratio >= 0.90 and re.search(
                r"(^id$|_id$|id_|uuid|guid|key$|record_id|row_id)",
                name_lower,
            ):
                schema["likely_id_cols"].append(col)

            if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
                num_ratio = _numeric_conversion_ratio(s)
                if num_ratio >= 0.90:
                    schema["convertible_numeric_cols"].append(col)

                dt_ratio = _datetime_conversion_ratio(s)
                if dt_ratio >= 0.80 and num_ratio < 0.80:
                    schema["likely_datetime_cols"].append(col)

                non_missing_text = s.dropna().astype(str)
                avg_len = float(non_missing_text.str.len().mean()) if len(non_missing_text) else 0.0
                avg_words = float(non_missing_text.str.split().map(len).mean()) if len(non_missing_text) else 0.0
                if avg_len >= 50 or avg_words >= 8:
                    schema["likely_text_cols"].append(col)

            if not pd.api.types.is_numeric_dtype(s) and col not in schema["likely_id_cols"]:
                if unique_count > high_cardinality_threshold or (
                    unique_count > one_hot_max_categories and unique_ratio > 0.10
                ):
                    schema["high_cardinality_cols"].append(col)
                elif unique_count <= one_hot_max_categories:
                    schema["low_cardinality_categoricals"].append(col)

        numeric_for_skew = [c for c in schema["numeric_cols"] if c != target_col]
        for col in numeric_for_skew:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if values.nunique() > 2 and len(values) >= 8:
                skew_value = float(values.skew())
                if abs(skew_value) >= 1.0:
                    schema["skewed_numeric_cols"].append(col)

        schema["datetime_cols"] = sorted(set(schema["datetime_cols"] + schema["likely_datetime_cols"]))
        schema["likely_id_cols"] = sorted(set(schema["likely_id_cols"]))
        schema["likely_text_cols"] = sorted(set(schema["likely_text_cols"]) - set(schema["likely_datetime_cols"]))
        schema["high_cardinality_cols"] = sorted(
            set(schema["high_cardinality_cols"])
            - set(schema["likely_id_cols"])
            - set(schema["likely_text_cols"])
            - set(schema["likely_datetime_cols"])
        )
        schema["convertible_numeric_cols"] = sorted(
            set(schema["convertible_numeric_cols"])
            - set(schema["likely_datetime_cols"])
            - set(schema["likely_text_cols"])
        )
        return schema

    def _detect_outliers(df, numeric_cols):
        rows = []
        bounds = {}
        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(values) < 4 or values.nunique() <= 1:
                continue
            if outlier_method == "zscore" and has_scipy:
                z = np.abs(stats.zscore(values, nan_policy="omit"))
                mask = z > 3
                lower = float(values.mean() - 3 * values.std())
                upper = float(values.mean() + 3 * values.std())
                count = int(mask.sum())
            else:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0 or pd.isna(iqr):
                    continue
                lower = float(q1 - 1.5 * iqr)
                upper = float(q3 + 1.5 * iqr)
                count = int(((values < lower) | (values > upper)).sum())
            pct = float(count / max(len(values), 1) * 100)
            if count:
                rows.append({"column": col, "outlier_count": count, "outlier_pct": pct, "lower": lower, "upper": upper})
                bounds[col] = (lower, upper)
        summary = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False) if rows else pd.DataFrame()
        return summary, bounds

    def _format_columns(cols, limit=12):
        cols = list(cols)
        if not cols:
            return "None"
        shown = ", ".join(map(str, cols[:limit]))
        if len(cols) > limit:
            shown += f", ... ({len(cols)} total)"
        return shown

    def _ask_choice(title, message, choices, default):
        normalized = {str(choice).strip().lower(): choice for choice in choices}
        default = default if default in choices else choices[0]
        if fully_auto or not interactive:
            return default

        prompt_lines = [message, "", "Options:"]
        for i, choice in enumerate(choices, start=1):
            marker = " [default]" if choice == default else ""
            prompt_lines.append(f"{i}. {choice}{marker}")
        prompt_lines.append("")
        prompt_lines.append("Enter a number or option name.")
        prompt = "\n".join(prompt_lines)

        answer = None
        if use_popup_prompts:
            try:
                import tkinter as tk
                from tkinter import simpledialog

                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                answer = simpledialog.askstring(title, prompt, parent=root)
                root.destroy()
            except Exception:
                answer = None

        if answer is None:
            try:
                answer = input(f"\n{title}\n{prompt}\n> ")
            except Exception:
                answer = ""

        answer = str(answer).strip()
        if not answer:
            return default
        if answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        answer_lower = answer.lower()
        if answer_lower in normalized:
            return normalized[answer_lower]
        for choice in choices:
            if str(choice).lower().startswith(answer_lower):
                return choice
        _warn(f"Unrecognized answer for '{title}'. Using default: {default}")
        return default

    def _make_decisions(df, schema, outlier_summary, inferred_problem_type):
        decisions = {}
        missing_cols = [c for c, pct in schema["missing_pct"].items() if pct > 0 and c != target_col]
        target_missing = bool(target_col in df.columns and df[target_col].isna().any()) if target_col else False

        if missing_cols:
            decisions["missing_action"] = _ask_choice(
                "Missing Values",
                f"Missing values were found in {_format_columns(missing_cols)}. How should feature missingness be handled?",
                ["impute", "drop_rows", "drop_columns", "leave"],
                "impute",
            )
        else:
            decisions["missing_action"] = "none"

        if decisions["missing_action"] == "impute":
            strategy_default = "knn_or_median" if imputation_strategy == "auto" else imputation_strategy
            decisions["numeric_imputation"] = _ask_choice(
                "Numeric Imputation",
                "Choose the numeric imputation method. KNN/iterative methods may fall back to median on large data.",
                ["knn_or_median", "iterative_or_median", "median", "mean"],
                strategy_default if strategy_default in {"knn_or_median", "iterative_or_median", "median", "mean"} else "knn_or_median",
            )
        else:
            decisions["numeric_imputation"] = "none"

        decisions["target_missing_action"] = "drop_rows_for_modeling" if target_missing else "none"
        if target_missing:
            _warn("Target column has missing values. Baseline modeling will drop those rows.")

        if not outlier_summary.empty:
            decisions["outlier_action"] = _ask_choice(
                "Outliers",
                f"Outliers were detected in {len(outlier_summary)} numeric columns. How should they be handled?",
                ["flag", "cap", "remove_rows", "leave"],
                "flag",
            )
        else:
            decisions["outlier_action"] = "none"

        if schema["high_cardinality_cols"]:
            decisions["high_cardinality_action"] = _ask_choice(
                "High Cardinality",
                f"High-cardinality categorical columns were detected: {_format_columns(schema['high_cardinality_cols'])}.",
                ["frequency_encode", "group_rare_then_one_hot", "drop", "leave"],
                "frequency_encode",
            )
        else:
            decisions["high_cardinality_action"] = "none"

        if schema["likely_id_cols"]:
            decisions["id_action"] = _ask_choice(
                "ID Columns",
                f"These columns look like identifiers: {_format_columns(schema['likely_id_cols'])}.",
                ["drop", "keep"],
                "drop",
            )
        else:
            decisions["id_action"] = "none"

        if schema["datetime_cols"]:
            decisions["datetime_action"] = _ask_choice(
                "Datetime Columns",
                f"These columns look like dates/times: {_format_columns(schema['datetime_cols'])}.",
                ["extract_parts", "drop", "leave"],
                "extract_parts",
            )
        else:
            decisions["datetime_action"] = "none"

        if schema["likely_text_cols"]:
            decisions["text_action"] = _ask_choice(
                "Text Columns",
                f"These columns look like free text: {_format_columns(schema['likely_text_cols'])}.",
                ["length_encode", "tfidf_vectorize", "drop", "leave"],
                "length_encode",
            )
        else:
            decisions["text_action"] = "none"

        if schema["skewed_numeric_cols"]:
            decisions["skew_action"] = _ask_choice(
                "Skewed Numeric Features",
                f"Skew was detected in {_format_columns(schema['skewed_numeric_cols'])}. Apply Yeo-Johnson transforms?",
                ["yeo_johnson", "leave"],
                "yeo_johnson" if fully_auto else "leave",
            )
        else:
            decisions["skew_action"] = "none"

        decisions["scaling_action"] = _ask_choice(
            "Scaling",
            "Should numeric model features be scaled with RobustScaler?",
            ["robust_scale", "standard_scale", "none"],
            "robust_scale",
        )

        pca_default = "run" if (_is_auto(run_pca) and len(df.columns) >= 3) or _as_bool(run_pca) else "skip"
        decisions["pca_action"] = _ask_choice(
            "PCA",
            "Should PCA analysis and PCA plots be generated?",
            ["run", "skip"],
            pca_default,
        )

        clustering_default = "run" if (
            (_is_auto(run_clustering) and inferred_problem_type == "clustering") or _as_bool(run_clustering)
        ) else "skip"
        decisions["clustering_action"] = _ask_choice(
            "Clustering",
            "Should clustering diagnostics be generated?",
            ["run", "skip"],
            clustering_default,
        )

        baseline_default = "run" if (
            target_col is not None
            and target_col in df.columns
            and inferred_problem_type in {"classification", "regression"}
            and (_is_auto(run_baseline_models) or _as_bool(run_baseline_models))
        ) else "skip"
        decisions["baseline_model_action"] = _ask_choice(
            "Baseline Models",
            "Should quick baseline models be trained for feature importance and predictive signal?",
            ["run", "skip"],
            baseline_default,
        )

        return decisions

    def _new_text_page(title, lines, footer=None):
        lines = [str(line) for line in lines]
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.95, title, fontsize=18, weight="bold", va="top")
        y = 0.90
        for raw_line in lines:
            if raw_line == "":
                y -= 0.025
                continue
            wrapped = textwrap.wrap(raw_line, width=105) or [""]
            for line in wrapped:
                if y < 0.07:
                    if footer:
                        fig.text(0.06, 0.035, footer, fontsize=8, color="gray")
                    fig.tight_layout()
                    yield fig
                    fig = plt.figure(figsize=(8.5, 11))
                    fig.patch.set_facecolor("white")
                    y = 0.94
                fig.text(0.06, y, line, fontsize=9.5, va="top")
                y -= 0.020
        if footer:
            fig.text(0.06, 0.035, footer, fontsize=8, color="gray")
        fig.tight_layout()
        yield fig

    def _display_notebook_text(title, lines):
        if not notebook_mode:
            return
        if has_ipython_display:
            _notebook_display(_NotebookMarkdown(f"## {title}"))
        else:
            print(f"\n\n## {title}")
        print("\n".join(str(line) for line in lines))

    def _display_notebook_fig(fig, section_name=None):
        if has_ipython_display:
            _notebook_display(fig)
        else:
            print(f"[plot rendered inline in notebook mode: {section_name or 'figure'}]")

    def _save_text_pages(pdf, title, lines):
        if notebook_mode:
            _display_notebook_text(title, lines)
        if pdf is not None:
            for fig in _new_text_page(title, lines):
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        pdf_sections.append(title)

    def _save_fig(pdf, fig, section_name):
        fig.tight_layout()
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
        if notebook_mode:
            _display_notebook_fig(fig, section_name)
        elif display_plots:
            plt.show()
        plt.close(fig)
        pdf_sections.append(section_name)

    def _plot_missingness(pdf, missing_summary):
        if missing_summary.empty or missing_summary["missing_count"].sum() == 0:
            return
        plot_df = missing_summary[missing_summary["missing_count"] > 0].head(40)
        fig, ax = plt.subplots(figsize=(10, max(4, min(12, 0.28 * len(plot_df)))))
        ax.barh(plot_df["column"].astype(str), plot_df["missing_pct"], color="#4477aa")
        ax.set_title("Missingness by Column")
        ax.set_xlabel("Missing percentage")
        ax.invert_yaxis()
        _save_fig(pdf, fig, "Missingness Plot")

    def _plot_numeric_distributions(pdf, df, numeric_cols):
        cols = [c for c in numeric_cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 1]
        if not cols:
            return
        for start in range(0, min(len(cols), 36), 9):
            chunk = cols[start : start + 9]
            n = len(chunk)
            fig, axes = plt.subplots(math.ceil(n / 3), 3, figsize=(12, 3.2 * math.ceil(n / 3)))
            axes = np.array(axes).reshape(-1)
            for ax, col in zip(axes, chunk):
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if has_seaborn:
                    sns.histplot(values, kde=True, ax=ax, color="#4477aa")
                else:
                    ax.hist(values, bins=30, color="#4477aa", alpha=0.8)
                ax.set_title(str(col), fontsize=10)
                ax.set_xlabel("")
            for ax in axes[len(chunk) :]:
                ax.axis("off")
            fig.suptitle("Numeric Histograms and Density", fontsize=14, y=1.02)
            _save_fig(pdf, fig, "Numeric Distributions")

    def _plot_boxplots(pdf, df, numeric_cols):
        cols = [c for c in numeric_cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 1]
        if not cols:
            return
        for start in range(0, min(len(cols), 36), 9):
            chunk = cols[start : start + 9]
            n = len(chunk)
            fig, axes = plt.subplots(math.ceil(n / 3), 3, figsize=(12, 3.2 * math.ceil(n / 3)))
            axes = np.array(axes).reshape(-1)
            for ax, col in zip(axes, chunk):
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                ax.boxplot(values, vert=False, patch_artist=True, boxprops={"facecolor": "#88ccee"})
                ax.set_title(str(col), fontsize=10)
                ax.set_yticks([])
            for ax in axes[len(chunk) :]:
                ax.axis("off")
            fig.suptitle("Numeric Boxplots", fontsize=14, y=1.02)
            _save_fig(pdf, fig, "Numeric Boxplots")

    def _plot_categoricals(pdf, df, categorical_cols):
        cols = []
        for col in categorical_cols:
            if col in df.columns and 1 < df[col].nunique(dropna=True) <= max_categories_for_countplot:
                cols.append(col)
        if not cols:
            return
        for start in range(0, min(len(cols), 24), 6):
            chunk = cols[start : start + 6]
            n = len(chunk)
            fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(12, 3.6 * math.ceil(n / 2)))
            axes = np.array(axes).reshape(-1)
            for ax, col in zip(axes, chunk):
                counts = df[col].astype("object").fillna("__MISSING__").value_counts().head(max_categories_for_countplot)
                ax.barh(counts.index.astype(str), counts.values, color="#66c2a5")
                ax.set_title(str(col), fontsize=10)
                ax.invert_yaxis()
            for ax in axes[len(chunk) :]:
                ax.axis("off")
            fig.suptitle("Categorical Count Plots", fontsize=14, y=1.02)
            _save_fig(pdf, fig, "Categorical Count Plots")

    def _plot_correlation(pdf, df, numeric_cols):
        cols = [c for c in numeric_cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 2]
        if len(cols) < 2:
            return
        sample = _sample_df(df[cols].apply(pd.to_numeric, errors="coerce"))
        corr = sample.corr(numeric_only=True)
        if corr.empty:
            return
        max_cols = 35
        if corr.shape[0] > max_cols:
            variances = sample.var(numeric_only=True).sort_values(ascending=False).head(max_cols).index
            corr = sample[variances].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(11, 9))
        if has_seaborn:
            sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=False, cbar_kws={"shrink": 0.7})
        else:
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
        ax.set_title("Numeric Correlation Heatmap")
        _save_fig(pdf, fig, "Correlation Heatmap")

    def _plot_pairplot(pdf, df, numeric_cols):
        if not has_seaborn:
            return
        cols = [c for c in numeric_cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 2]
        if len(cols) < 2:
            return
        sample = _sample_df(df[cols], max_rows=min(max_plot_rows, 1500))
        variances = sample.apply(pd.to_numeric, errors="coerce").var(numeric_only=True).sort_values(ascending=False)
        selected = list(variances.head(max_pairplot_features).index)
        if len(selected) < 2:
            return
        plot_df = sample[selected].apply(pd.to_numeric, errors="coerce").dropna()
        if len(plot_df) < 5:
            return
        grid = sns.pairplot(plot_df)
        grid.fig.suptitle("Pairplot of Sampled Numeric Features", y=1.02)
        if pdf is not None:
            pdf.savefig(grid.fig, bbox_inches="tight")
        if notebook_mode:
            _display_notebook_fig(grid.fig, "Pairplot")
        elif display_plots:
            plt.show()
        plt.close(grid.fig)
        pdf_sections.append("Pairplot")

    def _plot_target(pdf, df, target, inferred_problem_type, numeric_cols, categorical_cols):
        if target is None or target not in df.columns:
            return
        y = df[target]
        if inferred_problem_type == "classification":
            counts = y.astype("object").fillna("__MISSING__").value_counts().head(max_categories_for_countplot)
            fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(counts))))
            ax.barh(counts.index.astype(str), counts.values, color="#ddcc77")
            ax.set_title(f"Target Distribution: {target}")
            ax.invert_yaxis()
            _save_fig(pdf, fig, "Target Distribution")
        else:
            values = pd.to_numeric(y, errors="coerce").dropna()
            if len(values) > 1:
                fig, ax = plt.subplots(figsize=(9, 5))
                if has_seaborn:
                    sns.histplot(values, kde=True, ax=ax, color="#ddcc77")
                else:
                    ax.hist(values, bins=30, color="#ddcc77")
                ax.set_title(f"Target Distribution: {target}")
                _save_fig(pdf, fig, "Target Distribution")

        candidate_numeric = [c for c in numeric_cols if c != target and c in df.columns][:6]
        if candidate_numeric:
            n = len(candidate_numeric)
            fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(12, 3.8 * math.ceil(n / 2)))
            axes = np.array(axes).reshape(-1)
            sample = _sample_df(df[[target] + candidate_numeric].dropna(), max_rows=min(max_plot_rows, 3000))
            for ax, col in zip(axes, candidate_numeric):
                if inferred_problem_type == "classification":
                    if has_seaborn:
                        sns.boxplot(data=sample, x=target, y=col, ax=ax)
                        ax.tick_params(axis="x", rotation=45)
                    else:
                        groups = [g[col].dropna().values for _, g in sample.groupby(target)]
                        labels = [str(k) for k, _ in sample.groupby(target)]
                        ax.boxplot(groups, labels=labels)
                        ax.tick_params(axis="x", rotation=45)
                else:
                    ax.scatter(pd.to_numeric(sample[col], errors="coerce"), pd.to_numeric(sample[target], errors="coerce"), s=8, alpha=0.45)
                    ax.set_xlabel(str(col))
                    ax.set_ylabel(str(target))
                ax.set_title(f"{col} vs {target}", fontsize=10)
            for ax in axes[len(candidate_numeric) :]:
                ax.axis("off")
            fig.suptitle("Target vs Numeric Features", fontsize=14, y=1.02)
            _save_fig(pdf, fig, "Target vs Numeric Features")

        candidate_cats = [
            c
            for c in categorical_cols
            if c != target and c in df.columns and 1 < df[c].nunique(dropna=True) <= max_categories_for_countplot
        ][:6]
        if candidate_cats and inferred_problem_type == "regression":
            n = len(candidate_cats)
            fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(12, 3.8 * math.ceil(n / 2)))
            axes = np.array(axes).reshape(-1)
            sample = _sample_df(df[[target] + candidate_cats].dropna(), max_rows=min(max_plot_rows, 3000))
            for ax, col in zip(axes, candidate_cats):
                if has_seaborn:
                    sns.boxplot(data=sample, x=col, y=target, ax=ax)
                    ax.tick_params(axis="x", rotation=45)
                else:
                    groups = [g[target].dropna().values for _, g in sample.groupby(col)]
                    labels = [str(k) for k, _ in sample.groupby(col)]
                    ax.boxplot(groups, labels=labels)
                    ax.tick_params(axis="x", rotation=45)
                ax.set_title(f"{target} by {col}", fontsize=10)
            for ax in axes[len(candidate_cats) :]:
                ax.axis("off")
            fig.suptitle("Target vs Categorical Features", fontsize=14, y=1.02)
            _save_fig(pdf, fig, "Target vs Categorical Features")

    def _choose_numeric_imputer(n_rows, n_cols, missing_fraction, requested):
        if requested in {"median", "mean"}:
            return requested
        if requested == "iterative_or_median":
            if IterativeImputer is not None and n_rows <= 20000 and n_cols <= 40:
                return "iterative"
            return "median"
        if requested == "knn_or_median":
            if n_rows <= 50000 and n_cols <= 60 and missing_fraction <= 0.35:
                return "knn"
            return "median"
        if imputation_strategy == "iterative" and IterativeImputer is not None:
            return "iterative"
        if imputation_strategy == "knn":
            return "knn"
        if imputation_strategy == "simple":
            return "median"
        if imputation_strategy == "none":
            return "none"
        return "median"

    def _make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _prepare_features(X, schema, decisions, fit=True, objects=None, for_modeling=False):
        work = X.copy(deep=True)
        work = work.replace(r"^\s*$", np.nan, regex=True)
        objects = {} if objects is None else objects
        summary = {
            "dropped_columns": [],
            "converted_numeric_columns": [],
            "datetime_features_added": [],
            "text_features_added": [],
            "outlier_rows_removed": 0,
            "outlier_flags_added": [],
            "categorical_encoding": {},
            "imputation": {},
            "scaling": None,
            "skew_transforms": [],
        }
        row_mask = pd.Series(True, index=work.index)

        if fit:
            objects["input_columns"] = list(work.columns)
            objects["schema"] = schema
            objects["decisions"] = decisions

        numeric_convert_cols = [c for c in schema["convertible_numeric_cols"] if c in work.columns]
        if fit:
            objects["numeric_convert_cols"] = numeric_convert_cols
        else:
            numeric_convert_cols = objects.get("numeric_convert_cols", [])
        for col in numeric_convert_cols:
            work[col] = pd.to_numeric(_clean_numeric_text(work[col]), errors="coerce")
        summary["converted_numeric_columns"] = numeric_convert_cols

        if decisions.get("datetime_action") == "extract_parts":
            datetime_cols = [c for c in schema["datetime_cols"] if c in work.columns]
            if fit:
                objects["datetime_cols"] = datetime_cols
            else:
                datetime_cols = objects.get("datetime_cols", [])
            for col in datetime_cols:
                with py_warnings.catch_warnings():
                    py_warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(work[col], errors="coerce")
                new_cols = {
                    f"{col}__year": parsed.dt.year,
                    f"{col}__month": parsed.dt.month,
                    f"{col}__day": parsed.dt.day,
                    f"{col}__weekday": parsed.dt.weekday,
                    f"{col}__quarter": parsed.dt.quarter,
                    f"{col}__is_month_end": parsed.dt.is_month_end.astype("float"),
                    f"{col}__is_weekend": parsed.dt.weekday.isin([5, 6]).astype("float"),
                }
                for new_col, values in new_cols.items():
                    work[new_col] = values
                summary["datetime_features_added"].extend(new_cols.keys())
            work = work.drop(columns=[c for c in datetime_cols if c in work.columns], errors="ignore")
        elif decisions.get("datetime_action") == "drop":
            datetime_cols = [c for c in schema["datetime_cols"] if c in work.columns]
            work = work.drop(columns=datetime_cols, errors="ignore")
            summary["dropped_columns"].extend(datetime_cols)

        text_cols = [c for c in schema["likely_text_cols"] if c in work.columns]
        if decisions.get("text_action") == "length_encode":
            if fit:
                objects["text_cols"] = text_cols
            else:
                text_cols = objects.get("text_cols", [])
            for col in text_cols:
                text = work[col].fillna("").astype(str)
                added = {
                    f"{col}__char_len": text.str.len(),
                    f"{col}__word_count": text.str.split().map(len),
                    f"{col}__has_text": (text.str.len() > 0).astype(int),
                }
                for new_col, values in added.items():
                    work[new_col] = values
                summary["text_features_added"].extend(added.keys())
            work = work.drop(columns=[c for c in text_cols if c in work.columns], errors="ignore")
        elif decisions.get("text_action") == "tfidf_vectorize":
            vectorizers = {} if fit else objects.get("tfidf_vectorizers", {})
            tfidf_frames = []
            for col in text_cols:
                text = work[col].fillna("").astype(str)
                if fit:
                    vec = TfidfVectorizer(max_features=20, min_df=2)
                    try:
                        matrix = vec.fit_transform(text)
                        vectorizers[col] = vec
                    except ValueError:
                        _warn(f"TF-IDF skipped for {col}; not enough usable text.")
                        continue
                else:
                    vec = vectorizers.get(col)
                    if vec is None:
                        continue
                    matrix = vec.transform(text)
                names = [f"{col}__tfidf_{_safe_filename(name)}" for name in vec.get_feature_names_out()]
                tfidf_frames.append(pd.DataFrame(matrix.toarray(), columns=names, index=work.index))
                summary["text_features_added"].extend(names)
            if fit:
                objects["tfidf_vectorizers"] = vectorizers
            if tfidf_frames:
                work = pd.concat([work] + tfidf_frames, axis=1)
            work = work.drop(columns=[c for c in text_cols if c in work.columns], errors="ignore")
        elif decisions.get("text_action") == "drop":
            work = work.drop(columns=text_cols, errors="ignore")
            summary["dropped_columns"].extend(text_cols)

        if decisions.get("id_action") == "drop":
            id_cols = [c for c in schema["likely_id_cols"] if c in work.columns]
            if fit:
                objects["id_cols"] = id_cols
            else:
                id_cols = objects.get("id_cols", [])
            work = work.drop(columns=id_cols, errors="ignore")
            summary["dropped_columns"].extend(id_cols)

        if decisions.get("missing_action") == "drop_columns" and fit:
            drop_missing_cols = [c for c in work.columns if work[c].isna().mean() > 0]
            objects["drop_missing_cols"] = drop_missing_cols
        elif decisions.get("missing_action") == "drop_columns":
            drop_missing_cols = objects.get("drop_missing_cols", [])
        else:
            drop_missing_cols = []
        if drop_missing_cols:
            work = work.drop(columns=[c for c in drop_missing_cols if c in work.columns], errors="ignore")
            summary["dropped_columns"].extend(drop_missing_cols)

        if decisions.get("missing_action") == "drop_rows" and not for_modeling:
            before = len(work)
            work = work.dropna(axis=0)
            row_mask = row_mask.loc[work.index]
            summary["dropped_rows_for_missing"] = int(before - len(work))

        all_missing_cols = [c for c in work.columns if work[c].isna().all()]
        if fit:
            objects["all_missing_cols"] = all_missing_cols
        else:
            all_missing_cols = objects.get("all_missing_cols", [])
        if all_missing_cols:
            work = work.drop(columns=[c for c in all_missing_cols if c in work.columns], errors="ignore")
            summary["dropped_columns"].extend(all_missing_cols)

        for col in work.columns:
            if pd.api.types.is_bool_dtype(work[col]):
                work[col] = work[col].astype("float")

        numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]

        if decisions.get("outlier_action") in {"flag", "cap", "remove_rows"}:
            if fit:
                _, bounds = _detect_outliers(work, numeric_cols)
                objects["outlier_bounds"] = bounds
            else:
                bounds = objects.get("outlier_bounds", {})
            if decisions.get("outlier_action") == "flag":
                for col, (lower, upper) in bounds.items():
                    if col in work.columns:
                        flag_col = f"{col}__outlier_flag"
                        work[flag_col] = ((work[col] < lower) | (work[col] > upper)).astype(int)
                        summary["outlier_flags_added"].append(flag_col)
            elif decisions.get("outlier_action") == "cap":
                for col, (lower, upper) in bounds.items():
                    if col in work.columns:
                        work[col] = pd.to_numeric(work[col], errors="coerce").clip(lower=lower, upper=upper)
            elif decisions.get("outlier_action") == "remove_rows" and not for_modeling:
                remove_mask = pd.Series(False, index=work.index)
                for col, (lower, upper) in bounds.items():
                    if col in work.columns:
                        values = pd.to_numeric(work[col], errors="coerce")
                        remove_mask = remove_mask | (values < lower) | (values > upper)
                summary["outlier_rows_removed"] = int(remove_mask.sum())
                work = work.loc[~remove_mask].copy()
                row_mask = row_mask.loc[work.index]
            elif decisions.get("outlier_action") == "remove_rows" and for_modeling:
                _warn("Outlier row removal is not applied inside baseline model evaluation; model preprocessing uses the learned bounds only for reporting.")

        numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
        categorical_cols = [c for c in work.columns if c not in numeric_cols]

        should_impute = decisions.get("missing_action") == "impute" or for_modeling
        if should_impute:
            if numeric_cols:
                missing_fraction = float(work[numeric_cols].isna().mean().mean())
                if fit:
                    method = _choose_numeric_imputer(
                        n_rows=len(work),
                        n_cols=len(numeric_cols),
                        missing_fraction=missing_fraction,
                        requested=decisions.get("numeric_imputation", "knn_or_median"),
                    )
                    if method == "knn":
                        imputer = KNNImputer(n_neighbors=5, weights="distance")
                    elif method == "iterative" and IterativeImputer is not None:
                        imputer = IterativeImputer(random_state=random_state, max_iter=10, sample_posterior=False)
                    elif method == "mean":
                        imputer = SimpleImputer(strategy="mean")
                    elif method == "none":
                        imputer = None
                    else:
                        imputer = SimpleImputer(strategy="median")
                        method = "median"
                    objects["numeric_imputer"] = imputer
                    objects["numeric_imputer_columns"] = numeric_cols
                    objects["numeric_imputer_method"] = method
                    if imputer is not None:
                        imputer.fit(work[numeric_cols])
                else:
                    imputer = objects.get("numeric_imputer")
                    numeric_cols = [c for c in objects.get("numeric_imputer_columns", []) if c in work.columns]
                    method = objects.get("numeric_imputer_method", "unknown")
                if imputer is not None and numeric_cols:
                    work[numeric_cols] = work[numeric_cols].astype(float)
                    work[numeric_cols] = pd.DataFrame(
                        imputer.transform(work[numeric_cols]),
                        columns=numeric_cols,
                        index=work.index,
                    )
                summary["imputation"]["numeric"] = method

            if categorical_cols:
                if fit:
                    cat_imputer = SimpleImputer(strategy="most_frequent")
                    objects["categorical_imputer"] = cat_imputer
                    objects["categorical_imputer_columns"] = categorical_cols
                    cat_imputer.fit(work[categorical_cols].astype("object"))
                else:
                    cat_imputer = objects.get("categorical_imputer")
                    categorical_cols = [c for c in objects.get("categorical_imputer_columns", []) if c in work.columns]
                if categorical_cols and cat_imputer is not None:
                    work.loc[:, categorical_cols] = cat_imputer.transform(work[categorical_cols].astype("object"))
                summary["imputation"]["categorical"] = "most_frequent"

        if decisions.get("skew_action") == "yeo_johnson":
            numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
            if fit:
                skew_cols = []
                for col in numeric_cols:
                    values = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) >= 8 and values.nunique() > 2 and abs(values.skew()) >= 1.0:
                        skew_cols.append(col)
                objects["skew_cols"] = skew_cols
                if skew_cols:
                    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
                    transformer.fit(work[skew_cols].replace([np.inf, -np.inf], np.nan).fillna(0))
                    objects["skew_transformer"] = transformer
            else:
                skew_cols = [c for c in objects.get("skew_cols", []) if c in work.columns]
                transformer = objects.get("skew_transformer")
            if skew_cols and objects.get("skew_transformer") is not None:
                work[skew_cols] = pd.DataFrame(
                    objects["skew_transformer"].transform(
                        work[skew_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                    ),
                    columns=skew_cols,
                    index=work.index,
                )
                summary["skew_transforms"] = skew_cols

        cleaned_features = work.copy()

        categorical_cols = [c for c in work.columns if not pd.api.types.is_numeric_dtype(work[c])]
        high_card_cols = [c for c in schema["high_cardinality_cols"] if c in categorical_cols]

        if decisions.get("high_cardinality_action") == "frequency_encode":
            freq_maps = {} if fit else objects.get("frequency_maps", {})
            for col in high_card_cols:
                if fit:
                    freq_maps[col] = work[col].astype("object").fillna("__MISSING__").value_counts(normalize=True).to_dict()
                work[f"{col}__frequency"] = work[col].astype("object").fillna("__MISSING__").map(freq_maps.get(col, {})).fillna(0.0)
                work = work.drop(columns=[col])
                summary["categorical_encoding"][col] = "frequency"
            if fit:
                objects["frequency_maps"] = freq_maps
        elif decisions.get("high_cardinality_action") == "group_rare_then_one_hot":
            rare_maps = {} if fit else objects.get("rare_category_maps", {})
            for col in high_card_cols:
                if fit:
                    freqs = work[col].astype("object").fillna("__MISSING__").value_counts(normalize=True)
                    keepers = set(freqs[freqs >= rare_category_min_freq].index.astype(str))
                    if len(keepers) > one_hot_max_categories:
                        keepers = set(freqs.head(one_hot_max_categories).index.astype(str))
                    rare_maps[col] = keepers
                keepers = rare_maps.get(col, set())
                values = work[col].astype("object").fillna("__MISSING__").astype(str)
                work[col] = np.where(values.isin(keepers), values, "__RARE__")
                summary["categorical_encoding"][col] = "rare_grouped"
            if fit:
                objects["rare_category_maps"] = rare_maps
        elif decisions.get("high_cardinality_action") == "drop":
            work = work.drop(columns=high_card_cols, errors="ignore")
            summary["dropped_columns"].extend(high_card_cols)
            for col in high_card_cols:
                summary["categorical_encoding"][col] = "dropped"
        elif decisions.get("high_cardinality_action") == "leave":
            _warn("High-cardinality columns cannot remain as raw strings in transformed data; they will be frequency encoded.")
            freq_maps = {} if fit else objects.get("frequency_maps", {})
            for col in high_card_cols:
                if fit:
                    freq_maps[col] = work[col].astype("object").fillna("__MISSING__").value_counts(normalize=True).to_dict()
                work[f"{col}__frequency"] = work[col].astype("object").fillna("__MISSING__").map(freq_maps.get(col, {})).fillna(0.0)
                work = work.drop(columns=[col])
                summary["categorical_encoding"][col] = "frequency"
            if fit:
                objects["frequency_maps"] = freq_maps

        categorical_cols = [c for c in work.columns if not pd.api.types.is_numeric_dtype(work[c])]
        if fit:
            safe_cat_cols = []
            overflow_cat_cols = []
            for col in categorical_cols:
                n_unique = work[col].astype("object").fillna("__MISSING__").nunique(dropna=False)
                if n_unique <= one_hot_max_categories:
                    safe_cat_cols.append(col)
                else:
                    overflow_cat_cols.append(col)
            objects["safe_one_hot_cols"] = safe_cat_cols
            objects["overflow_cat_cols"] = overflow_cat_cols
        else:
            safe_cat_cols = [c for c in objects.get("safe_one_hot_cols", []) if c in work.columns]
            overflow_cat_cols = [c for c in objects.get("overflow_cat_cols", []) if c in work.columns]

        overflow_maps = {} if fit else objects.get("overflow_frequency_maps", {})
        for col in overflow_cat_cols:
            if fit:
                overflow_maps[col] = work[col].astype("object").fillna("__MISSING__").value_counts(normalize=True).to_dict()
            work[f"{col}__frequency"] = work[col].astype("object").fillna("__MISSING__").map(overflow_maps.get(col, {})).fillna(0.0)
            work = work.drop(columns=[col])
            summary["categorical_encoding"][col] = "frequency_overflow"
        if fit:
            objects["overflow_frequency_maps"] = overflow_maps

        if safe_cat_cols:
            if fit:
                encoder = _make_ohe()
                encoder.fit(work[safe_cat_cols].astype("object").fillna("__MISSING__").astype(str))
                objects["one_hot_encoder"] = encoder
                objects["one_hot_input_cols"] = safe_cat_cols
                encoded_names = list(encoder.get_feature_names_out(safe_cat_cols))
                objects["one_hot_feature_names"] = encoded_names
            else:
                encoder = objects.get("one_hot_encoder")
                safe_cat_cols = [c for c in objects.get("one_hot_input_cols", []) if c in work.columns]
                encoded_names = objects.get("one_hot_feature_names", [])
            if encoder is not None and safe_cat_cols:
                encoded = encoder.transform(work[safe_cat_cols].astype("object").fillna("__MISSING__").astype(str))
                encoded_df = pd.DataFrame(encoded, columns=encoded_names, index=work.index)
                work = pd.concat([work.drop(columns=safe_cat_cols), encoded_df], axis=1)
                for col in safe_cat_cols:
                    summary["categorical_encoding"][col] = "one_hot"
        elif fit:
            objects["one_hot_encoder"] = None
            objects["one_hot_input_cols"] = []
            objects["one_hot_feature_names"] = []

        raw_object_cols = [c for c in work.columns if not pd.api.types.is_numeric_dtype(work[c])]
        if raw_object_cols:
            _warn(f"Dropping non-numeric columns from transformed data: {_format_columns(raw_object_cols)}")
            work = work.drop(columns=raw_object_cols, errors="ignore")
            summary["dropped_columns"].extend(raw_object_cols)

        work = work.replace([np.inf, -np.inf], np.nan)
        numeric_cols = list(work.columns)
        if numeric_cols:
            work[numeric_cols] = work[numeric_cols].astype(float)
            if fit:
                final_imputer = SimpleImputer(strategy="median")
                final_imputer.fit(work[numeric_cols])
                objects["final_numeric_imputer"] = final_imputer
                objects["final_numeric_cols"] = numeric_cols
            else:
                final_imputer = objects.get("final_numeric_imputer")
                numeric_cols = [c for c in objects.get("final_numeric_cols", []) if c in work.columns]
            if final_imputer is not None and numeric_cols:
                work[numeric_cols] = pd.DataFrame(
                    final_imputer.transform(work[numeric_cols]),
                    columns=numeric_cols,
                    index=work.index,
                )

        scale_action = decisions.get("scaling_action")
        if scale_action in {"robust_scale", "standard_scale"}:
            one_hot_names = set(objects.get("one_hot_feature_names", []))
            scale_cols = [
                c
                for c in work.columns
                if c not in one_hot_names
                and not str(c).endswith("__outlier_flag")
                and pd.api.types.is_numeric_dtype(work[c])
                and work[c].nunique(dropna=True) > 2
            ]
            if fit:
                scaler = RobustScaler() if scale_action == "robust_scale" else StandardScaler()
                if scale_cols:
                    scaler.fit(work[scale_cols])
                objects["scaler"] = scaler
                objects["scaled_columns"] = scale_cols
                objects["scaler_type"] = scale_action
            else:
                scaler = objects.get("scaler")
                scale_cols = [c for c in objects.get("scaled_columns", []) if c in work.columns]
            if scale_cols and scaler is not None:
                work[scale_cols] = work[scale_cols].astype(float)
                work[scale_cols] = pd.DataFrame(
                    scaler.transform(work[scale_cols]),
                    columns=scale_cols,
                    index=work.index,
                )
            summary["scaling"] = scale_action

        if fit:
            objects["transformed_columns"] = list(work.columns)
        else:
            expected_cols = objects.get("transformed_columns", [])
            for col in expected_cols:
                if col not in work.columns:
                    work[col] = 0.0
            extra_cols = [c for c in work.columns if c not in expected_cols]
            if extra_cols:
                work = work.drop(columns=extra_cols)
            work = work[expected_cols]

        return cleaned_features, work, objects, row_mask, summary

    def _run_pca_analysis(X_transformed, y=None):
        summary = {"ran": False, "reason": None}
        figs = []
        if decisions.get("pca_action") != "run":
            summary["reason"] = "Skipped by decision."
            return summary, figs
        X_num = X_transformed.select_dtypes(include=[np.number]).copy()
        X_num = X_num.loc[:, X_num.nunique(dropna=True) > 1]
        if X_num.shape[0] < 3 or X_num.shape[1] < 2:
            summary["reason"] = "Not enough numeric features for PCA."
            return summary, figs
        sample = _sample_df(X_num, max_rows=max_plot_rows)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample)
        n_components = int(min(10, X_scaled.shape[0], X_scaled.shape[1]))
        pca_model = PCA(n_components=n_components, random_state=random_state)
        components = pca_model.fit_transform(X_scaled)
        explained = pca_model.explained_variance_ratio_
        summary = {
            "ran": True,
            "sample_rows": int(len(sample)),
            "n_components": n_components,
            "explained_variance_ratio": [float(x) for x in explained],
            "cumulative_explained_variance": [float(x) for x in np.cumsum(explained)],
            "pca_model": pca_model,
            "scaler": scaler,
        }

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(1, n_components + 1), np.cumsum(explained), marker="o", color="#4477aa")
        ax.bar(range(1, n_components + 1), explained, alpha=0.45, color="#88ccee")
        ax.set_title("PCA Explained Variance")
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_ylim(0, min(1.05, max(0.1, np.cumsum(explained)[-1] + 0.05)))
        figs.append(("PCA Explained Variance", fig))

        if n_components >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            if y is not None:
                y_sample = y.loc[sample.index] if hasattr(y, "loc") else None
                labels = pd.Series(y_sample, index=sample.index).astype("object") if y_sample is not None else None
                if labels is not None and labels.nunique(dropna=True) <= 12:
                    for label in labels.dropna().unique():
                        mask = labels == label
                        ax.scatter(components[mask, 0], components[mask, 1], s=14, alpha=0.65, label=str(label))
                    ax.legend(fontsize=8, loc="best")
                else:
                    ax.scatter(components[:, 0], components[:, 1], s=14, alpha=0.65, color="#4477aa")
            else:
                ax.scatter(components[:, 0], components[:, 1], s=14, alpha=0.65, color="#4477aa")
            ax.set_title("PCA 2D Projection")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            figs.append(("PCA 2D Projection", fig))
        return summary, figs

    def _run_clustering_diagnostics(X_transformed):
        summary = {"ran": False, "reason": None}
        figs = []
        if decisions.get("clustering_action") != "run":
            summary["reason"] = "Skipped by decision."
            return summary, figs
        X_num = X_transformed.select_dtypes(include=[np.number]).copy()
        X_num = X_num.loc[:, X_num.nunique(dropna=True) > 1]
        if X_num.shape[0] < 10 or X_num.shape[1] < 2:
            summary["reason"] = "Not enough numeric data for clustering diagnostics."
            return summary, figs

        sample_n = min(max_plot_rows, len(X_num))
        repeats = max(1, sample_count if len(X_num) > 100000 else 1)
        k_values = list(range(2, min(10, sample_n - 1) + 1))
        if not k_values:
            summary["reason"] = "Not enough rows for multiple cluster sizes."
            return summary, figs

        all_inertia = {k: [] for k in k_values}
        all_silhouette = {k: [] for k in k_values}
        for i in range(repeats):
            sample = X_num.sample(n=sample_n, random_state=random_state + i) if len(X_num) > sample_n else X_num.copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(sample)
            for k in k_values:
                try:
                    model = KMeans(n_clusters=k, n_init=10, random_state=random_state + i)
                    labels = model.fit_predict(X_scaled)
                    all_inertia[k].append(float(model.inertia_))
                    if len(set(labels)) > 1 and len(sample) <= 10000:
                        all_silhouette[k].append(float(silhouette_score(X_scaled, labels)))
                except Exception as exc:
                    _warn(f"Clustering diagnostic failed for k={k}: {exc}")

        inertia_mean = {k: float(np.mean(v)) for k, v in all_inertia.items() if v}
        silhouette_mean = {k: float(np.mean(v)) for k, v in all_silhouette.items() if v}
        summary = {
            "ran": True,
            "sample_rows": int(sample_n),
            "sample_repeats": int(repeats),
            "k_values": k_values,
            "inertia_mean": inertia_mean,
            "silhouette_mean": silhouette_mean,
            "best_k_by_silhouette": max(silhouette_mean, key=silhouette_mean.get) if silhouette_mean else None,
        }

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(inertia_mean.keys()), list(inertia_mean.values()), marker="o", color="#4477aa")
        ax.set_title("KMeans Elbow Diagnostic")
        ax.set_xlabel("k")
        ax.set_ylabel("Mean inertia")
        figs.append(("Clustering Elbow", fig))

        if silhouette_mean:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(list(silhouette_mean.keys()), list(silhouette_mean.values()), marker="o", color="#66c2a5")
            ax.set_title("KMeans Silhouette Diagnostic")
            ax.set_xlabel("k")
            ax.set_ylabel("Mean silhouette score")
            figs.append(("Clustering Silhouette", fig))
        return summary, figs

    def _compute_vif(X_transformed):
        summary = {"ran": False, "reason": None, "table": None}
        if not run_vif:
            summary["reason"] = "Skipped by parameter."
            return summary
        if not has_statsmodels:
            summary["reason"] = "statsmodels is not installed."
            return summary
        X_num = X_transformed.select_dtypes(include=[np.number]).copy()
        X_num = X_num.loc[:, X_num.nunique(dropna=True) > 1]
        if X_num.shape[1] < 2:
            summary["reason"] = "Not enough numeric features."
            return summary
        if X_num.shape[1] > 50:
            selected = X_num.var(numeric_only=True).sort_values(ascending=False).head(50).index
            X_num = X_num[selected]
            _warn("VIF limited to the 50 highest-variance transformed features.")
        X_sample = _sample_df(X_num, max_rows=min(max_plot_rows, 5000)).replace([np.inf, -np.inf], np.nan).fillna(0)
        try:
            rows = []
            values = X_sample.to_numpy(dtype=float)
            for i, col in enumerate(X_sample.columns):
                vif_value = float(variance_inflation_factor(values, i))
                rows.append({"feature": col, "vif": vif_value})
            table = pd.DataFrame(rows).sort_values("vif", ascending=False)
            return {"ran": True, "reason": None, "table": table}
        except Exception as exc:
            summary["reason"] = str(exc)
            return summary

    def _normality_summary(df, numeric_cols):
        rows = []
        for col in numeric_cols[:100]:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 8 or values.nunique() <= 2:
                continue
            sample = values.sample(n=min(len(values), 5000), random_state=random_state)
            row = {
                "column": col,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "skew": float(values.skew()),
                "kurtosis": float(values.kurtosis()),
            }
            if has_scipy and len(sample) >= 20:
                try:
                    stat, pvalue = stats.normaltest(sample)
                    row["normaltest_pvalue"] = float(pvalue)
                except Exception:
                    row["normaltest_pvalue"] = np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _run_baseline_modeling(raw_df, feature_columns, target, inferred_problem_type, schema, decisions):
        summary = {"ran": False, "reason": None}
        model_objects = {}
        if decisions.get("baseline_model_action") != "run":
            summary["reason"] = "Skipped by decision."
            return summary, model_objects
        if target is None or target not in raw_df.columns:
            summary["reason"] = "No target column supplied."
            return summary, model_objects
        model_df = raw_df[feature_columns + [target]].copy()
        model_df = model_df.dropna(subset=[target])
        if len(model_df) < 30:
            summary["reason"] = "Not enough rows with non-missing target for baseline modeling."
            return summary, model_objects
        X_raw = model_df[feature_columns]
        y = model_df[target]
        if inferred_problem_type == "classification" and y.nunique(dropna=True) < 2:
            summary["reason"] = "Classification target has fewer than two classes."
            return summary, model_objects

        stratify = None
        if inferred_problem_type == "classification":
            counts = y.value_counts()
            if len(counts) > 1 and counts.min() >= 2 and len(counts) <= max(20, len(model_df) // 5):
                stratify = y
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw,
                y,
                test_size=0.25,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw,
                y,
                test_size=0.25,
                random_state=random_state,
            )

        _, X_train_t, train_preprocessor, _, train_prep_summary = _prepare_features(
            X_train,
            schema=schema,
            decisions=decisions,
            fit=True,
            objects=None,
            for_modeling=True,
        )
        _, X_test_t, _, _, _ = _prepare_features(
            X_test,
            schema=schema,
            decisions=decisions,
            fit=False,
            objects=train_preprocessor,
            for_modeling=True,
        )

        if X_train_t.shape[1] == 0:
            summary["reason"] = "No usable model features after preprocessing."
            return summary, model_objects

        if inferred_problem_type == "classification":
            model = RandomForestClassifier(
                n_estimators=250,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            model.fit(X_train_t, y_train)
            pred = model.predict(X_test_t)
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
                "classification_report": classification_report(y_test, pred, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            }
        else:
            model = RandomForestRegressor(
                n_estimators=250,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train_t, y_train)
            pred = model.predict(X_test_t)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            metrics = {
                "rmse": rmse,
                "mae": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
            }

        importances = pd.DataFrame(
            {
                "feature": X_train_t.columns,
                "importance": getattr(model, "feature_importances_", np.zeros(X_train_t.shape[1])),
            }
        ).sort_values("importance", ascending=False)

        summary = {
            "ran": True,
            "problem_type": inferred_problem_type,
            "train_rows": int(len(X_train_t)),
            "test_rows": int(len(X_test_t)),
            "n_features": int(X_train_t.shape[1]),
            "metrics": metrics,
            "feature_importance": importances.head(50),
            "preprocessing_summary": train_prep_summary,
        }
        model_objects = {
            "model": model,
            "train_preprocessor": train_preprocessor,
            "feature_columns": list(X_train_t.columns),
        }
        return summary, model_objects

    def _plot_feature_importance(pdf, modeling_summary):
        if not modeling_summary.get("ran"):
            return
        importances = modeling_summary.get("feature_importance")
        if importances is None or len(importances) == 0:
            return
        plot_df = importances.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(plot_df))))
        ax.barh(plot_df["feature"].astype(str), plot_df["importance"], color="#4477aa")
        ax.set_title("Baseline Model Feature Importance")
        ax.set_xlabel("Importance")
        _save_fig(pdf, fig, "Feature Importance")

    def _lines_from_df(df, max_rows=20):
        if df is None or len(df) == 0:
            return ["None"]
        with pd.option_context("display.max_rows", max_rows, "display.max_columns", 8, "display.width", 120):
            return str(df.head(max_rows)).splitlines()

    def _print_basic_summary():
        if not verbose:
            return
        print("\nBASIC EDA SUMMARY")
        print(f"Rows: {eda_summary['n_rows']:,} | Columns: {eda_summary['n_columns']:,}")
        print(f"Duplicate rows: {eda_summary['duplicate_rows']:,}")
        print(f"Inferred problem type: {inferred_problem_type}")
        print(f"Numeric columns: {_format_columns(schema['numeric_cols'])}")
        print(f"Categorical columns: {_format_columns(schema['categorical_cols'])}")
        print(f"Numeric-like text columns: {_format_columns(schema['convertible_numeric_cols'])}")
        print(f"Likely datetime columns: {_format_columns(schema['datetime_cols'])}")
        print(f"Likely text columns: {_format_columns(schema['likely_text_cols'])}")
        print(f"Likely ID columns: {_format_columns(schema['likely_id_cols'])}")
        print(f"High-cardinality categoricals: {_format_columns(schema['high_cardinality_cols'])}")
        if not missingness_summary.empty and missingness_summary["missing_count"].sum() > 0:
            print("\nTop missingness:")
            print(missingness_summary[missingness_summary["missing_count"] > 0].head(10).to_string(index=False))
        if not outlier_summary.empty:
            print("\nTop outlier findings:")
            print(outlier_summary.head(10).to_string(index=False))

    def _render_report(pdf):
            intro_lines = [
                "This report was generated by universal_eda_pipeline.",
                f"Source: {source_description}",
                f"Original shape: {original_shape[0]:,} rows x {original_shape[1]:,} columns.",
                f"Inferred problem type: {inferred_problem_type}.",
                f"Target column: {target_col if target_col else 'None supplied'}.",
                "",
                "Executive summary:",
                f"- Missing columns: {int((missingness_summary['missing_count'] > 0).sum()) if not missingness_summary.empty else 0}.",
                f"- Duplicate rows: {eda_summary['duplicate_rows']:,}.",
                f"- Numeric columns: {len(schema['numeric_cols'])}.",
                f"- Categorical columns: {len(schema['categorical_cols'])}.",
                f"- Likely ID columns: {len(schema['likely_id_cols'])}.",
                f"- Likely datetime columns: {len(schema['datetime_cols'])}.",
                f"- Likely free-text columns: {len(schema['likely_text_cols'])}.",
                f"- High-cardinality categorical columns: {len(schema['high_cardinality_cols'])}.",
                f"- Numeric columns with detected outliers: {len(outlier_summary) if not outlier_summary.empty else 0}.",
                "",
                "Important caution:",
                "The transformed_df output is fitted on the provided dataset for exploratory use. Baseline model metrics use a separate train/test split with preprocessing fitted on the training split only.",
            ]
            _save_text_pages(pdf, "Executive Summary", intro_lines)

            overview_lines = [
                "Dataset overview:",
                f"Rows: {eda_summary['n_rows']:,}",
                f"Columns: {eda_summary['n_columns']:,}",
                f"Duplicate rows: {eda_summary['duplicate_rows']:,}",
                f"Memory usage: {eda_summary['memory_mb']:.2f} MB",
                "",
                "Column type findings:",
                f"Numeric: {_format_columns(schema['numeric_cols'], 30)}",
                f"Categorical/object: {_format_columns(schema['categorical_cols'], 30)}",
                f"Boolean: {_format_columns(schema['boolean_cols'], 30)}",
                f"Numeric-like text: {_format_columns(schema['convertible_numeric_cols'], 30)}",
                f"Datetime: {_format_columns(schema['datetime_cols'], 30)}",
                f"Free-text: {_format_columns(schema['likely_text_cols'], 30)}",
                f"Likely IDs: {_format_columns(schema['likely_id_cols'], 30)}",
                f"High-cardinality categoricals: {_format_columns(schema['high_cardinality_cols'], 30)}",
            ]
            _save_text_pages(pdf, "Dataset Overview", overview_lines)

            missing_lines = [
                "Missingness analysis:",
                "Columns are sorted by missing percentage. Missingness should be reviewed for business meaning before interpreting model results.",
                "",
            ] + _lines_from_df(missingness_summary, max_rows=35)
            _save_text_pages(pdf, "Missingness Analysis", missing_lines)
            _plot_missingness(pdf, missingness_summary)

            if numeric_summary is not None and not numeric_summary.empty:
                _save_text_pages(
                    pdf,
                    "Numeric Descriptive Statistics",
                    ["Numeric descriptive statistics:"] + _lines_from_df(numeric_summary, max_rows=35),
                )
            if categorical_summary is not None and not categorical_summary.empty:
                _save_text_pages(
                    pdf,
                    "Categorical Descriptive Statistics",
                    ["Categorical descriptive statistics:"] + _lines_from_df(categorical_summary, max_rows=35),
                )

            _plot_numeric_distributions(pdf, raw_df, schema["numeric_cols"] + schema["convertible_numeric_cols"])
            _plot_boxplots(pdf, raw_df, schema["numeric_cols"])
            _plot_categoricals(pdf, raw_df, schema["categorical_cols"])
            _plot_correlation(pdf, raw_df, schema["numeric_cols"])
            _plot_pairplot(pdf, raw_df, schema["numeric_cols"])
            _plot_target(pdf, raw_df, target_col, inferred_problem_type, schema["numeric_cols"], schema["categorical_cols"])

            outlier_lines = [
                "Outlier analysis:",
                f"Detection method: {outlier_method}.",
                "Outliers are statistical flags, not automatic errors. Review the business context before removing or capping observations.",
                "",
            ] + _lines_from_df(outlier_summary, max_rows=35)
            _save_text_pages(pdf, "Outlier Analysis", outlier_lines)

            decision_lines = ["Preprocessing decisions:"] + [f"- {k}: {v}" for k, v in decisions.items()]
            decision_lines += [
                "",
                "Preprocessing summary:",
                f"Dropped columns: {_format_columns(prep_summary.get('dropped_columns', []), 35)}",
                f"Converted numeric-like columns: {_format_columns(prep_summary.get('converted_numeric_columns', []), 35)}",
                f"Datetime features added: {_format_columns(prep_summary.get('datetime_features_added', []), 35)}",
                f"Text features added: {_format_columns(prep_summary.get('text_features_added', []), 35)}",
                f"Outlier flags added: {_format_columns(prep_summary.get('outlier_flags_added', []), 35)}",
                f"Outlier rows removed: {prep_summary.get('outlier_rows_removed', 0)}",
                f"Scaling: {prep_summary.get('scaling')}",
                f"Skew transforms: {_format_columns(prep_summary.get('skew_transforms', []), 35)}",
            ]
            _save_text_pages(pdf, "Preprocessing Decisions", decision_lines)

            normality_lines = [
                "Distribution and normality checks:",
                "Normality tests are sensitive to sample size. Use them as diagnostics, not as absolute pass/fail rules.",
                "",
            ] + _lines_from_df(normality_results, max_rows=35)
            _save_text_pages(pdf, "Normality and Skewness", normality_lines)

            if vif_summary.get("ran"):
                vif_lines = [
                    "Multicollinearity check:",
                    "VIF values above 5 to 10 often deserve review, depending on the modeling goal.",
                    "",
                ] + _lines_from_df(vif_summary.get("table"), max_rows=35)
            else:
                vif_lines = ["Multicollinearity check skipped.", f"Reason: {vif_summary.get('reason')}"]
            _save_text_pages(pdf, "Multicollinearity", vif_lines)

            if pca_summary.get("ran"):
                pca_lines = [
                    "PCA analysis:",
                    f"Sample rows: {pca_summary['sample_rows']:,}",
                    f"Components: {pca_summary['n_components']}",
                    f"Cumulative explained variance: {[round(x, 4) for x in pca_summary['cumulative_explained_variance']]}",
                ]
            else:
                pca_lines = ["PCA skipped.", f"Reason: {pca_summary.get('reason')}"]
            _save_text_pages(pdf, "PCA Analysis", pca_lines)
            for section, fig in pca_figs:
                _save_fig(pdf, fig, section)

            if clustering_summary.get("ran"):
                clustering_lines = [
                    "Clustering diagnostics:",
                    f"Sample rows: {clustering_summary['sample_rows']:,}",
                    f"Sample repeats: {clustering_summary['sample_repeats']}",
                    f"k values: {clustering_summary['k_values']}",
                    f"Best k by silhouette: {clustering_summary['best_k_by_silhouette']}",
                    f"Mean inertia: {clustering_summary['inertia_mean']}",
                    f"Mean silhouette: {clustering_summary['silhouette_mean']}",
                ]
            else:
                clustering_lines = ["Clustering diagnostics skipped.", f"Reason: {clustering_summary.get('reason')}"]
            _save_text_pages(pdf, "Clustering Diagnostics", clustering_lines)
            for section, fig in clustering_figs:
                _save_fig(pdf, fig, section)

            if modeling_summary.get("ran"):
                modeling_lines = [
                    "Baseline modeling:",
                    f"Problem type: {modeling_summary['problem_type']}",
                    f"Train rows: {modeling_summary['train_rows']:,}",
                    f"Test rows: {modeling_summary['test_rows']:,}",
                    f"Transformed features: {modeling_summary['n_features']:,}",
                    "",
                    "Metrics:",
                ]
                for key, value in modeling_summary["metrics"].items():
                    if key in {"classification_report", "confusion_matrix"}:
                        modeling_lines.append(f"{key}:")
                        modeling_lines.extend(str(value).splitlines())
                    else:
                        modeling_lines.append(f"- {key}: {value:.4f}")
                modeling_lines += ["", "Top feature importances:"]
                modeling_lines += _lines_from_df(modeling_summary.get("feature_importance"), max_rows=35)
            else:
                modeling_lines = ["Baseline modeling skipped.", f"Reason: {modeling_summary.get('reason')}"]
            _save_text_pages(pdf, "Baseline Modeling", modeling_lines)
            _plot_feature_importance(pdf, modeling_summary)

            limitations = [
                "Limitations and cautions:",
                "- Automated EDA can find patterns and risks, but it cannot validate business meaning.",
                "- ID, datetime, and text detection are heuristic and should be reviewed.",
                "- Outlier handling should be justified by domain context.",
                "- One-hot encoding is limited to safe-cardinality columns; high-cardinality columns are frequency encoded or handled according to the recorded decisions.",
                "- Baseline models are quick diagnostics, not tuned production models.",
                "- If this dataset includes train and holdout/future rows together, rerun preprocessing on the train split only before final modeling.",
                "",
                "Recommended next steps:",
                "- Review the missingness and outlier sections.",
                "- Confirm which detected ID, text, and datetime columns should be included in modeling.",
                "- Use the returned preprocessing objects to reproduce transformations.",
                "- Build a task-specific validation strategy before final model selection.",
            ]
            if warning_messages:
                limitations += ["", "Warnings:"] + [f"- {w}" for w in warning_messages]
            _save_text_pages(pdf, "Limitations and Next Steps", limitations)

    def _build_pdf_report():
        output_path.mkdir(parents=True, exist_ok=True)
        with PdfPages(report_path) as pdf:
            _render_report(pdf)

    output_path = Path(output_dir)
    if create_pdf or save_cleaned_csv:
        output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / report_name if create_pdf else None

    raw_df, source_description = _load_data(data)
    original_shape = raw_df.shape
    if target_col is not None and target_col not in raw_df.columns:
        _warn(f"Target column '{target_col}' was not found. Continuing without a target.")
        target_col = None

    schema = _infer_schema(raw_df)
    inferred_problem_type = _likely_problem_type(raw_df, target_col)

    missingness_summary = pd.DataFrame(
        {
            "column": raw_df.columns,
            "missing_count": [int(raw_df[c].isna().sum()) for c in raw_df.columns],
            "missing_pct": [float(raw_df[c].isna().mean() * 100) for c in raw_df.columns],
            "dtype": [str(raw_df[c].dtype) for c in raw_df.columns],
            "unique_count": [int(raw_df[c].nunique(dropna=True)) for c in raw_df.columns],
        }
    ).sort_values(["missing_pct", "missing_count"], ascending=False)

    numeric_for_stats = [c for c in schema["numeric_cols"] if c in raw_df.columns]
    categorical_for_stats = [c for c in raw_df.columns if c not in numeric_for_stats]
    numeric_summary = raw_df[numeric_for_stats].describe().T if numeric_for_stats else pd.DataFrame()
    if categorical_for_stats:
        categorical_summary = raw_df[categorical_for_stats].astype("object").describe().T
        categorical_summary["unique_count"] = [raw_df[c].nunique(dropna=True) for c in categorical_for_stats]
    else:
        categorical_summary = pd.DataFrame()

    outlier_summary, initial_outlier_bounds = _detect_outliers(raw_df, schema["numeric_cols"])

    eda_summary = {
        "source": source_description,
        "n_rows": int(raw_df.shape[0]),
        "n_columns": int(raw_df.shape[1]),
        "column_names": list(raw_df.columns),
        "dtypes": {c: str(raw_df[c].dtype) for c in raw_df.columns},
        "duplicate_rows": int(raw_df.duplicated().sum()),
        "memory_mb": float(raw_df.memory_usage(deep=True).sum() / (1024**2)),
        "dataset_size_bucket": "small"
        if len(raw_df) < 10000
        else "medium"
        if len(raw_df) <= 100000
        else "large",
        "sampling_used_for_expensive_steps": bool(len(raw_df) > max_plot_rows),
        "inferred_problem_type": inferred_problem_type,
    }

    _print_basic_summary()
    decisions = _make_decisions(raw_df, schema, outlier_summary, inferred_problem_type)
    if verbose:
        print("\nPREPROCESSING DECISIONS")
        for key, value in decisions.items():
            print(f"{key}: {value}")

    feature_columns = [c for c in raw_df.columns if c != target_col]
    cleaned_features, X_transformed, exploratory_objects, row_mask, prep_summary = _prepare_features(
        raw_df[feature_columns],
        schema=schema,
        decisions=decisions,
        fit=True,
        objects=None,
        for_modeling=False,
    )

    if len(cleaned_features) != len(raw_df):
        aligned_target = raw_df.loc[cleaned_features.index, target_col] if target_col else None
    else:
        aligned_target = raw_df[target_col] if target_col else None

    cleaned_df = cleaned_features.copy()
    transformed_df = X_transformed.copy()
    if target_col is not None and aligned_target is not None:
        cleaned_df[target_col] = aligned_target
        transformed_df[target_col] = aligned_target.loc[transformed_df.index]

    normality_results = _normality_summary(cleaned_df, [c for c in cleaned_df.columns if c != target_col])
    X_for_advanced = transformed_df.drop(columns=[target_col], errors="ignore")
    y_for_advanced = transformed_df[target_col] if target_col in transformed_df.columns else None
    pca_summary, pca_figs = _run_pca_analysis(X_for_advanced, y=y_for_advanced)
    clustering_summary, clustering_figs = _run_clustering_diagnostics(X_for_advanced)
    vif_summary = _compute_vif(X_for_advanced)
    modeling_summary, modeling_objects = _run_baseline_modeling(
        raw_df=raw_df,
        feature_columns=feature_columns,
        target=target_col,
        inferred_problem_type=inferred_problem_type,
        schema=schema,
        decisions=decisions,
    )

    preprocessing_objects = {
        "exploratory": exploratory_objects,
        "modeling": modeling_objects,
        "initial_outlier_bounds": initial_outlier_bounds,
    }

    encoding_summary = prep_summary.get("categorical_encoding", {})
    if save_cleaned_csv:
        cleaned_path = output_path / "cleaned_data.csv"
        transformed_path = output_path / "transformed_data.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        transformed_df.to_csv(transformed_path, index=False)
    else:
        cleaned_path = None
        transformed_path = None

    if create_pdf:
        _build_pdf_report()
    elif notebook_mode:
        _render_report(None)
    else:
        for _, fig in pca_figs + clustering_figs:
            plt.close(fig)

    if verbose:
        if create_pdf:
            print(f"\nEDA report saved to: {report_path}")
        elif notebook_mode:
            print("\nNotebook EDA report rendered inline. No PDF was created.")
        else:
            print("\nEDA summaries created. No PDF or inline notebook report was requested.")
        print(f"Cleaned shape: {cleaned_df.shape[0]:,} rows x {cleaned_df.shape[1]:,} columns")
        print(f"Transformed shape: {transformed_df.shape[0]:,} rows x {transformed_df.shape[1]:,} columns")

    return {
        "original_shape": original_shape,
        "cleaned_df": cleaned_df,
        "transformed_df": transformed_df,
        "preprocessing_objects": preprocessing_objects,
        "eda_summary": eda_summary,
        "missingness_summary": missingness_summary,
        "outlier_summary": outlier_summary,
        "encoding_summary": encoding_summary,
        "modeling_summary": modeling_summary,
        "pca_summary": pca_summary,
        "clustering_summary": clustering_summary,
        "vif_summary": vif_summary,
        "normality_summary": normality_results,
        "generated_plot_paths": [],
        "pdf_sections": pdf_sections,
        "report_path": str(report_path) if report_path else None,
        "notebook_mode": bool(notebook_mode),
        "create_pdf": bool(create_pdf),
        "cleaned_csv_path": str(cleaned_path) if cleaned_path else None,
        "transformed_csv_path": str(transformed_path) if transformed_path else None,
        "user_decisions": decisions,
        "warnings": warning_messages,
    }
