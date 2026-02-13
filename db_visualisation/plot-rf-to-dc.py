import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math
from typing import Dict, Any, List

try:
    from scipy.optimize import curve_fit
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

db_path = "../data.db"
conn = sqlite3.connect(db_path)

f_min = 910
f_max = 920

# Optional target voltage filter (mV). Set both to None to disable.
v_min = 1450
v_max = 1550

# Haal alle tabelnamen op
tables = pd.read_sql("""
    SELECT name FROM sqlite_master
    WHERE type='table'
""", conn)["name"].tolist()

plt.figure(figsize=(10, 6))

marker_size = 3

# Linear power units for both axes
power_unit = "uW"
rf_scale = 1e6  # W -> uW
dc_scale = 1e6  # W -> uW

# Select which models to include
ENABLED_MODELS = [
    # "linear_eta",
    "piecewise",
    # "logistic",
    # "shifted_sigmoid",
    # "polynomial3",
    # "rational",
    # "log_surrogate",
    # "power_law",
]

MODEL_LABELS = {
    "linear_eta": r"$y = a x + b$",
    "piecewise": r"$y=0;\ y=\eta(x-x_s);\ y=y_{\max}$",
    "logistic": r"$y=\frac{y_{\max}}{1+\exp(-a(x-b))}$",
    "shifted_sigmoid": r"$y=y_{\max}\left(\frac{1}{1+\exp(-a(x-b))}-c\right)$",
    "polynomial3": r"$y=a_1 x + a_2 x^2 + a_3 x^3$",
    "rational": r"$y=\frac{a x^2}{1 + b x}$",
    "log_surrogate": r"$y=a\log(1 + b x)$",
    "power_law": r"$y=a x^k$",
}

color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
linestyles = ["-", "--", "-.", ":"]
table_colors = {
    table: (color_cycle[i % len(color_cycle)] if color_cycle else None)
    for i, table in enumerate(tables)
}
freq_linestyles = {}
freq_to_tuning = {}
legend_labels_used = set()
max_p_dc = None

fit_rows: List[Dict[str, Any]] = []
demo_samples = []
tikz_blocks = []


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot


def linear_efficiency_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    y_hat = a * x + b
    return {"a": float(a), "b": float(b), "y_hat": y_hat, "r2": r2_score(y, y_hat)}


def piecewise_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_min == x_max:
        return {"x_sens": x_min, "x_sat": x_max, "eta": 0.0, "y_max": 0.0, "y_hat": np.zeros_like(y), "r2": float("nan")}

    xs_grid = np.linspace(x_min, x_max * 0.8, 10)
    xsat_grid = np.linspace(x_min * 1.2, x_max, 10)
    best = None

    for x_sens in xs_grid:
        for x_sat in xsat_grid:
            if x_sat <= x_sens:
                continue
            in_lin = (x >= x_sens) & (x <= x_sat)
            if not np.any(in_lin):
                continue
            x_lin = x[in_lin] - x_sens
            y_lin = y[in_lin]
            denom = np.sum(x_lin ** 2)
            if denom == 0:
                continue
            eta = np.sum(x_lin * y_lin) / denom
            if not (0 < eta < 1.0):
                continue
            y_max = eta * (x_sat - x_sens)
            y_hat = np.where(
                x < x_sens,
                0.0,
                np.where(x <= x_sat, eta * (x - x_sens), y_max),
            )
            r2 = r2_score(y, y_hat)
            if best is None or (not math.isnan(r2) and r2 > best["r2"]):
                best = {
                    "x_sens": float(x_sens),
                    "x_sat": float(x_sat),
                    "eta": float(eta),
                    "y_max": float(y_max),
                    "y_hat": y_hat,
                    "r2": float(r2),
                }

    if best is None:
        return {"x_sens": x_min, "x_sat": x_max, "eta": 0.0, "y_max": 0.0, "y_hat": np.zeros_like(y), "r2": float("nan")}
    return best


def logistic_model(x: np.ndarray, y_max: float, a: float, b: float) -> np.ndarray:
    return y_max / (1.0 + np.exp(-a * (x - b)))


def shifted_sigmoid_model(x: np.ndarray, y_max: float, a: float, b: float) -> np.ndarray:
    c = 1.0 / (1.0 + np.exp(a * b))
    return y_max * (1.0 / (1.0 + np.exp(-a * (x - b))) - c)


def logistic_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_max0 = float(np.max(y)) if np.max(y) > 0 else 1.0
    a0 = 0.01 if np.max(x) > 0 else 1.0
    b0 = float(np.median(x))

    if SCIPY_OK:
        try:
            params, _ = curve_fit(
                logistic_model,
                x,
                y,
                p0=[y_max0, a0, b0],
                bounds=([0.0, 0.0, np.min(x)], [np.inf, np.inf, np.max(x)]),
                maxfev=5000,
            )
            y_hat = logistic_model(x, *params)
            return {"y_max": float(params[0]), "a": float(params[1]), "b": float(params[2]), "y_hat": y_hat, "r2": r2_score(y, y_hat)}
        except Exception:
            pass

    a_grid = np.linspace(0.001, 0.2, 10)
    b_grid = np.linspace(np.min(x), np.max(x), 10)
    y_max_grid = np.linspace(np.max(y) * 0.8, np.max(y) * 1.2, 6)
    best = None
    for a in a_grid:
        for b in b_grid:
            for y_max in y_max_grid:
                y_hat = logistic_model(x, y_max, a, b)
                r2 = r2_score(y, y_hat)
                if best is None or (not math.isnan(r2) and r2 > best["r2"]):
                    best = {"y_max": float(y_max), "a": float(a), "b": float(b), "y_hat": y_hat, "r2": float(r2)}
    return best if best is not None else {"y_max": y_max0, "a": a0, "b": b0, "y_hat": logistic_model(x, y_max0, a0, b0), "r2": float("nan")}


def shifted_sigmoid_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_max0 = float(np.max(y)) if np.max(y) > 0 else 1.0
    a0 = 0.01 if np.max(x) > 0 else 1.0
    b0 = float(np.median(x))

    if SCIPY_OK:
        try:
            params, _ = curve_fit(
                shifted_sigmoid_model,
                x,
                y,
                p0=[y_max0, a0, b0],
                bounds=([0.0, 0.0, np.min(x)], [np.inf, np.inf, np.max(x)]),
                maxfev=5000,
            )
            y_hat = shifted_sigmoid_model(x, *params)
            return {"y_max": float(params[0]), "a": float(params[1]), "b": float(params[2]), "y_hat": y_hat, "r2": r2_score(y, y_hat)}
        except Exception:
            pass

    a_grid = np.linspace(0.001, 0.2, 10)
    b_grid = np.linspace(np.min(x), np.max(x), 10)
    y_max_grid = np.linspace(np.max(y) * 0.8, np.max(y) * 1.2, 6)
    best = None
    for a in a_grid:
        for b in b_grid:
            for y_max in y_max_grid:
                y_hat = shifted_sigmoid_model(x, y_max, a, b)
                r2 = r2_score(y, y_hat)
                if best is None or (not math.isnan(r2) and r2 > best["r2"]):
                    best = {"y_max": float(y_max), "a": float(a), "b": float(b), "y_hat": y_hat, "r2": float(r2)}
    return best if best is not None else {"y_max": y_max0, "a": a0, "b": b0, "y_hat": shifted_sigmoid_model(x, y_max0, a0, b0), "r2": float("nan")}


def polynomial_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    X = np.vstack([x, x ** 2, x ** 3]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    return {"a1": float(coeffs[0]), "a2": float(coeffs[1]), "a3": float(coeffs[2]), "y_hat": y_hat, "r2": r2_score(y, y_hat)}


def rational_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    # y + b x y = a x^2  ->  [x^2, -x*y] [a, b]^T = y
    A = np.vstack([x ** 2, -x * y]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    y_hat = (a * x ** 2) / (1 + b * x)
    return {"a": float(a), "b": float(b), "y_hat": y_hat, "r2": r2_score(y, y_hat)}


def log_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    b_grid = np.linspace(1e-6, 0.02, 12)
    best = None
    for b in b_grid:
        lx = np.log1p(b * x)
        denom = np.sum(lx ** 2)
        if denom == 0:
            continue
        a = np.sum(y * lx) / denom
        y_hat = a * lx
        r2 = r2_score(y, y_hat)
        if best is None or (not math.isnan(r2) and r2 > best["r2"]):
            best = {"a": float(a), "b": float(b), "y_hat": y_hat, "r2": float(r2)}
    return best if best is not None else {"a": 0.0, "b": 0.0, "y_hat": np.zeros_like(y), "r2": float("nan")}


def power_law_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    mask = (x > 0) & (y > 0)
    if np.count_nonzero(mask) < 2:
        return {"a": 0.0, "k": 0.0, "y_hat": np.zeros_like(y), "r2": float("nan")}
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    A = np.vstack([np.ones_like(lx), lx]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    log_a, k = coeffs
    k = float(np.clip(k, 0.0, 1.0))
    a = float(np.exp(log_a))
    y_hat = a * (x ** k)
    return {"a": a, "k": k, "y_hat": y_hat, "r2": r2_score(y, y_hat)}


def append_fit_rows(group_meta: Dict[str, Any], x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 3:
        return

    fits = [
        ("linear_eta", linear_efficiency_fit(x, y)),
        ("piecewise", piecewise_fit(x, y)),
        ("logistic", logistic_fit(x, y)),
        ("shifted_sigmoid", shifted_sigmoid_fit(x, y)),
        ("polynomial3", polynomial_fit(x, y)),
        ("rational", rational_fit(x, y)),
        ("log_surrogate", log_fit(x, y)),
        ("power_law", power_law_fit(x, y)),
    ]

    for model_name, result in fits:
        row = {
            **group_meta,
            "model": model_name,
            "r2": result.get("r2"),
        }
        for k, v in result.items():
            if k in {"y_hat", "r2"}:
                continue
            row[k] = v
        fit_rows.append(row)


def eval_model(model_name: str, params: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    if model_name == "linear_eta":
        return params["a"] * x + params["b"]
    if model_name == "piecewise":
        x_sens = params["x_sens"]
        x_sat = params["x_sat"]
        eta = params["eta"]
        y_max = params["y_max"]
        return np.where(
            x < x_sens,
            0.0,
            np.where(x <= x_sat, eta * (x - x_sens), y_max),
        )
    if model_name == "logistic":
        return logistic_model(x, params["y_max"], params["a"], params["b"])
    if model_name == "shifted_sigmoid":
        return shifted_sigmoid_model(x, params["y_max"], params["a"], params["b"])
    if model_name == "polynomial3":
        return params["a1"] * x + params["a2"] * (x ** 2) + params["a3"] * (x ** 3)
    if model_name == "rational":
        return (params["a"] * x ** 2) / (1 + params["b"] * x)
    if model_name == "log_surrogate":
        return params["a"] * np.log1p(params["b"] * x)
    if model_name == "power_law":
        return params["a"] * (x ** params["k"])
    return np.zeros_like(x)


def best_fit_and_log(group_meta: Dict[str, Any], x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if len(x) < 3:
        return {}

    fits = [
        ("linear_eta", linear_efficiency_fit(x, y)),
        ("piecewise", piecewise_fit(x, y)),
        ("logistic", logistic_fit(x, y)),
        ("shifted_sigmoid", shifted_sigmoid_fit(x, y)),
        ("polynomial3", polynomial_fit(x, y)),
        ("rational", rational_fit(x, y)),
        ("log_surrogate", log_fit(x, y)),
        ("power_law", power_law_fit(x, y)),
    ]

    best = None
    for model_name, result in fits:
        if model_name not in ENABLED_MODELS:
            continue
        row = {
            **group_meta,
            "model": model_name,
            "r2": result.get("r2"),
        }
        for k, v in result.items():
            if k in {"y_hat", "r2"}:
                continue
            row[k] = v
        fit_rows.append(row)
        if best is None or (
            result.get("r2") is not None
            and not math.isnan(result["r2"])
            and result["r2"] > best["r2"]
        ):
            best = {
                "model": model_name,
                "r2": float(result.get("r2")),
                "params": {k: v for k, v in result.items() if k not in {"y_hat", "r2"}},
            }

    if best is None:
        return {}
    print(f"Best fit | {group_meta} -> {best['model']} (R2={best['r2']:.4f}) params={best['params']}")
    return best

for table in tables:

    print(table)
    try:
        df = pd.read_sql(f"""
            SELECT frequency_mhz, level_dbm, pwr_pw, source, tuning_frequency_mhz, target_voltage_mv
            FROM "{table}"
            WHERE frequency_mhz BETWEEN ? AND ?
        """, conn, params=(f_min, f_max))

        if df.empty:
            continue

        if v_min is not None and v_max is not None:
            df = df.loc[
                (df["target_voltage_mv"] >= v_min) &
                (df["target_voltage_mv"] <= v_max)
            ]
            if df.empty:
                continue

        unique_frequencies = sorted(df["frequency_mhz"].unique())
        unique_targets = sorted(df["target_voltage_mv"].dropna().unique())
        table_color = table_colors.get(table)

        for freq in unique_frequencies:
            df_f = df[df["frequency_mhz"] == freq]
            if df_f.empty:
                continue

            tf_vals = sorted(df_f["tuning_frequency_mhz"].dropna().unique())
            if tf_vals:
                freq_to_tuning.setdefault(freq, set()).update(tf_vals)

            if freq not in freq_linestyles:
                freq_linestyles[freq] = linestyles[len(freq_linestyles) % len(linestyles)]
            table_linestyle = freq_linestyles[freq]

            if len(unique_targets) == 0:
                source = df_f["source"].iloc[0]
                tf_mhz = df_f["tuning_frequency_mhz"].iloc[0]
                df_plot = df_f.sort_values(by="level_dbm").copy()
                df_plot["p_rf_w"] = (10 ** (df_plot["level_dbm"] / 10)) / 1e3
                plot_label = table if table not in legend_labels_used else None
                legend_labels_used.add(table)
                y_vals = df_plot["pwr_pw"] / dc_scale
                local_max = y_vals.max()
                max_p_dc = local_max if max_p_dc is None else max(max_p_dc, local_max)
                x_vals = df_plot["p_rf_w"] * rf_scale
                demo_samples.append(
                    {
                        "meta": {
                            "table": table,
                            "frequency_mhz": float(freq),
                            "tuning_frequency_mhz": float(tf_mhz),
                            "target_voltage_mv": None,
                            "source": source,
                            "n_points": int(len(df_plot)),
                        },
                        "x": x_vals.to_numpy(),
                        "y": y_vals.to_numpy(),
                    }
                )

                best = best_fit_and_log(
                    {
                        "table": table,
                        "frequency_mhz": float(freq),
                        "tuning_frequency_mhz": float(tf_mhz),
                        "target_voltage_mv": None,
                        "source": source,
                        "n_points": int(len(df_plot)),
                    },
                    x_vals.to_numpy(),
                    y_vals.to_numpy(),
                )
                plt.plot(
                    x_vals,
                    y_vals,
                    marker="o",
                    markersize=marker_size,
                    linestyle="None",
                    color=table_color,
                    label=plot_label
                )
                if best:
                    x_sorted = np.sort(x_vals.to_numpy())
                    y_fit = eval_model(best["model"], best["params"], x_sorted)
                    plt.plot(
                        x_sorted,
                        y_fit,
                        linestyle=table_linestyle,
                        color=table_color,
                        linewidth=1.5,
                        alpha=0.9,
                    )
                    tikz_blocks.append(
                        {
                            "meta": {
                                "table": table,
                                "frequency_mhz": float(freq),
                                "tuning_frequency_mhz": float(tf_mhz),
                                "target_voltage_mv": None,
                                "source": source,
                            },
                            "data": list(zip(x_vals.to_numpy(), y_vals.to_numpy())),
                            "fit": list(zip(x_sorted, y_fit)),
                            "fit_model": best["model"],
                            "fit_params": best["params"],
                        }
                    )
                continue

            for target in unique_targets:
                df_ft = df_f[df_f["target_voltage_mv"] == target]
                if df_ft.empty:
                    continue

                source = df_ft["source"].iloc[0]
                tf_mhz = df_ft["tuning_frequency_mhz"].iloc[0]
                df_plot = df_ft.sort_values(by="level_dbm").copy()
                df_plot["p_rf_w"] = (10 ** (df_plot["level_dbm"] / 10)) / 1e3

                plot_label = table if table not in legend_labels_used else None
                legend_labels_used.add(table)
                y_vals = df_plot["pwr_pw"] / dc_scale
                local_max = y_vals.max()
                max_p_dc = local_max if max_p_dc is None else max(max_p_dc, local_max)
                x_vals = df_plot["p_rf_w"] * rf_scale
                demo_samples.append(
                    {
                        "meta": {
                            "table": table,
                            "frequency_mhz": float(freq),
                            "tuning_frequency_mhz": float(tf_mhz),
                            "target_voltage_mv": float(target),
                            "source": source,
                            "n_points": int(len(df_plot)),
                        },
                        "x": x_vals.to_numpy(),
                        "y": y_vals.to_numpy(),
                    }
                )

                best = best_fit_and_log(
                    {
                        "table": table,
                        "frequency_mhz": float(freq),
                        "tuning_frequency_mhz": float(tf_mhz),
                        "target_voltage_mv": float(target),
                        "source": source,
                        "n_points": int(len(df_plot)),
                    },
                    x_vals.to_numpy(),
                    y_vals.to_numpy(),
                )
                plt.plot(
                    x_vals,
                    y_vals,
                    marker="o",
                    markersize=marker_size,
                    linestyle="None",
                    color=table_color,
                    label=plot_label
                )
                if best:
                    x_sorted = np.sort(x_vals.to_numpy())
                    y_fit = eval_model(best["model"], best["params"], x_sorted)
                    plt.plot(
                        x_sorted,
                        y_fit,
                        linestyle=table_linestyle,
                        color=table_color,
                        linewidth=1.5,
                        alpha=0.9,
                    )
                    tikz_blocks.append(
                        {
                            "meta": {
                                "table": table,
                                "frequency_mhz": float(freq),
                                "tuning_frequency_mhz": float(tf_mhz),
                                "target_voltage_mv": float(target),
                                "source": source,
                            },
                            "data": list(zip(x_vals.to_numpy(), y_vals.to_numpy())),
                            "fit": list(zip(x_sorted, y_fit)),
                            "fit_model": best["model"],
                            "fit_params": best["params"],
                        }
                    )

    except Exception as e:
        print(f"⚠️ Skipping table {table}: {e}")

conn.close()

plt.xlabel(f"P_rf ({power_unit})")
plt.ylabel(f"P_dc ({power_unit})")
plt.grid(True)
plt.xlim(0, 1000)
if max_p_dc is not None:
    plt.ylim(0, max_p_dc * 1.05)
ax = plt.gca()

if freq_linestyles:
    freq_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls,
            marker="o",
            markersize=marker_size,
            label=(
                f"{round(freq)} MHz"
                + (
                    f" ({', '.join(str(int(t)) for t in sorted(freq_to_tuning.get(freq, [])))} MHz)"
                    if freq in freq_to_tuning
                    else ""
                )
            )
        )
        for freq, ls in sorted(freq_linestyles.items())
    ]
    freq_legend = ax.legend(handles=freq_handles, title="Frequency (tune freq.)", loc="upper left")
    ax.add_artist(freq_legend)

table_handles = [
    Line2D(
        [0],
        [0],
        color=table_colors[table],
        linestyle="-",
        marker="o",
        markersize=marker_size,
        label=table
    )
    for table in tables
    if table in table_colors and table_colors[table] is not None
]
if table_handles:
    ax.legend(handles=table_handles, title="Harvester", loc="lower right")

if fit_rows:
    fit_df = pd.DataFrame(fit_rows)
    fit_df = fit_df.sort_values(
        by=["table", "frequency_mhz", "target_voltage_mv", "model"],
        na_position="last"
    )
    print("\nFit results:")
    print(fit_df)
    fit_df.to_csv("fit_results.csv", index=False)

    group_cols = ["table", "frequency_mhz", "tuning_frequency_mhz", "target_voltage_mv", "source", "n_points"]
    best_df = (
        fit_df.dropna(subset=["r2"])
        .sort_values(by=["r2"], ascending=False)
        .groupby(group_cols, dropna=False, as_index=False)
        .head(1)
    )
    if not best_df.empty:
        best_df = best_df.sort_values(
            by=["table", "frequency_mhz", "target_voltage_mv", "model"],
            na_position="last"
        )
        print("\nBest fit per experiment:")
        print(best_df)
        best_df.to_csv("fit_results_best.csv", index=False)

        summary_path = "fit_results_best.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            for _, row in best_df.iterrows():
                meta = (
                    f"table={row.get('table')}, "
                    f"frequency_mhz={row.get('frequency_mhz')}, "
                    f"tuning_frequency_mhz={row.get('tuning_frequency_mhz')}, "
                    f"target_voltage_mv={row.get('target_voltage_mv')}, "
                    f"source={row.get('source')}, "
                    f"n_points={row.get('n_points')}"
                )
                model = row.get("model")
                r2 = row.get("r2")
                params = []
                params_dict = {}
                for col, val in row.items():
                    if col in {"table", "frequency_mhz", "tuning_frequency_mhz", "target_voltage_mv", "source", "n_points", "model", "r2"}:
                        continue
                    if pd.isna(val):
                        continue
                    params.append(f"{col}={val}")
                    params_dict[col] = val
                params_str = ", ".join(params) if params else "none"
                f.write(f"{meta}\n")
                f.write(f"  best_fit={model}, r2={r2}, function={MODEL_LABELS.get(model, model)}\n")
                f.write(f"  params: {params_str}\n\n")

                f.write("  python:\n")
                f.write("  ```python\n")
                f.write("  import numpy as np\n")
                if model == "linear_eta":
                    f.write(f"  a = {params_dict.get('a', 0.0)}\n")
                    f.write(f"  b = {params_dict.get('b', 0.0)}\n")
                    f.write("  def f(x):\n")
                    f.write("      return a * x + b\n")
                elif model == "piecewise":
                    f.write(f"  x_sens = {params_dict.get('x_sens', 0.0)}\n")
                    f.write(f"  x_sat = {params_dict.get('x_sat', 0.0)}\n")
                    f.write(f"  eta = {params_dict.get('eta', 0.0)}\n")
                    f.write(f"  y_max = {params_dict.get('y_max', 0.0)}\n")
                    f.write("  def f(x):\n")
                    f.write("      return np.where(\n")
                    f.write("          x < x_sens,\n")
                    f.write("          0.0,\n")
                    f.write("          np.where(x <= x_sat, eta * (x - x_sens), y_max),\n")
                    f.write("      )\n")
                elif model == "logistic":
                    f.write(f"  y_max = {params_dict.get('y_max', 0.0)}\n")
                    f.write(f"  a = {params_dict.get('a', 0.0)}\n")
                    f.write(f"  b = {params_dict.get('b', 0.0)}\n")
                    f.write("  def f(x):\n")
                    f.write("      return y_max / (1.0 + np.exp(-a * (x - b)))\n")
                elif model == "shifted_sigmoid":
                    f.write(f"  y_max = {params_dict.get('y_max', 0.0)}\n")
                    f.write(f"  a = {params_dict.get('a', 0.0)}\n")
                    f.write(f"  b = {params_dict.get('b', 0.0)}\n")
                    f.write("  c = 1.0 / (1.0 + np.exp(a * b))\n")
                    f.write("  def f(x):\n")
                    f.write("      return y_max * (1.0 / (1.0 + np.exp(-a * (x - b))) - c)\n")
                elif model == "polynomial3":
                    f.write(f"  a1 = {params_dict.get('a1', 0.0)}\n")
                    f.write(f"  a2 = {params_dict.get('a2', 0.0)}\n")
                    f.write(f"  a3 = {params_dict.get('a3', 0.0)}\n")
                    f.write("  def f(x):\n")
                    f.write("      return a1 * x + a2 * (x ** 2) + a3 * (x ** 3)\n")
                elif model == "rational":
                    f.write(f"  a = {params_dict.get('a', 0.0)}\n")
                    f.write(f"  b = {params_dict.get('b', 0.0)}\n")
                    f.write("  def f(x):\n")
                    f.write("      return (a * x ** 2) / (1.0 + b * x)\n")
                elif model == "log_surrogate":
                    f.write(f"  a = {params_dict.get('a', 0.0)}\n")
                    f.write(f"  b = {params_dict.get('b', 0.0)}\n")
                    f.write("  def f(x):\n")
                    f.write("      return a * np.log1p(b * x)\n")
                elif model == "power_law":
                    f.write(f"  a = {params_dict.get('a', 0.0)}\n")
                    f.write(f"  k = {params_dict.get('k', 0.0)}\n")
                    f.write("  def f(x):\n")
                    f.write("      return a * (x ** k)\n")
                else:
                    f.write("  def f(x):\n")
                    f.write("      return np.zeros_like(x)\n")
                f.write("  ```\n\n")

if tikz_blocks:
    with open("tikz_addplot.txt", "w", encoding="utf-8") as f:
        for block in tikz_blocks:
            meta = block["meta"]
            f.write(
                f"% table={meta['table']}, f={meta['frequency_mhz']} MHz, "
                f"tune={meta['tuning_frequency_mhz']} MHz, "
                f"t={meta['target_voltage_mv']} mV, source={meta['source']}, "
                f"fit={block['fit_model']}\n"
            )
            f.write("\\addplot coordinates {\n")
            for x, y in block["data"]:
                f.write(f"  ({x:.6g},{y:.6g})\n")
            f.write("};\n")
            x_min = min(x for x, _ in block["fit"])
            x_max = max(x for x, _ in block["fit"])
            f.write(f"\\addplot[domain={x_min:.6g}:{x_max:.6g}] ")
            model = block["fit_model"]
            if model == "linear_eta":
                a = block["fit_params"].get("a", 0.0)
                b = block["fit_params"].get("b", 0.0)
                expr = f"{a:.6g}*x + {b:.6g}"
            elif model == "piecewise":
                x_sens = block["fit_params"].get("x_sens", 0.0)
                x_sat = block["fit_params"].get("x_sat", 0.0)
                eta = block["fit_params"].get("eta", 0.0)
                y_max = block["fit_params"].get("y_max", 0.0)
                expr = (
                    f"(x < {x_sens:.6g} ? 0 : "
                    f"(x <= {x_sat:.6g} ? {eta:.6g}*(x-{x_sens:.6g}) : {y_max:.6g}))"
                )
            elif model == "logistic":
                y_max = block["fit_params"].get("y_max", 0.0)
                a = block["fit_params"].get("a", 0.0)
                b = block["fit_params"].get("b", 0.0)
                expr = f"{y_max:.6g}/(1+exp(-{a:.6g}*(x-{b:.6g})))"
            elif model == "shifted_sigmoid":
                y_max = block["fit_params"].get("y_max", 0.0)
                a = block["fit_params"].get("a", 0.0)
                b = block["fit_params"].get("b", 0.0)
                expr = (
                    f"{y_max:.6g}*(1/(1+exp(-{a:.6g}*(x-{b:.6g})))"
                    f" - (1/(1+exp({a:.6g}*{b:.6g}))))"
                )
            elif model == "polynomial3":
                a1 = block["fit_params"].get("a1", 0.0)
                a2 = block["fit_params"].get("a2", 0.0)
                a3 = block["fit_params"].get("a3", 0.0)
                expr = f"{a1:.6g}*x + {a2:.6g}*x^2 + {a3:.6g}*x^3"
            elif model == "rational":
                a = block["fit_params"].get("a", 0.0)
                b = block["fit_params"].get("b", 0.0)
                expr = f"({a:.6g}*x^2)/(1+{b:.6g}*x)"
            elif model == "log_surrogate":
                a = block["fit_params"].get("a", 0.0)
                b = block["fit_params"].get("b", 0.0)
                expr = f"{a:.6g}*ln(1+{b:.6g}*x)"
            elif model == "power_law":
                a = block["fit_params"].get("a", 0.0)
                k = block["fit_params"].get("k", 0.0)
                expr = f"{a:.6g}*x^{k:.6g}"
            else:
                expr = "0"
            f.write(f"{{{expr}}};\n\n")

    with open("tikz_full.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")
        f.write(f"  xlabel={{P_rf ({power_unit})}},\n")
        f.write(f"  ylabel={{P_dc ({power_unit})}},\n")
        f.write("  grid=both,\n")
        f.write("  xmin=0,\n")
        f.write("  xmax=1000,\n")
        if max_p_dc is not None:
            f.write(f"  ymin=0,\n  ymax={max_p_dc * 1.05:.6g},\n")
        f.write("]\n")
        for block in tikz_blocks:
            meta = block["meta"]
            f.write(
                f"% table={meta['table']}, f={meta['frequency_mhz']} MHz, "
                f"tune={meta['tuning_frequency_mhz']} MHz, "
                f"t={meta['target_voltage_mv']} mV, source={meta['source']}, "
                f"fit={block['fit_model']}\n"
            )
            x_min = min(x for x, _ in block["fit"])
            x_max = max(x for x, _ in block["fit"])
            f.write(f"\\addplot[domain={x_min:.6g}:{x_max:.6g}] ")
            model = block["fit_model"]
            params = block["fit_params"]
            if model == "linear_eta":
                a = params.get("a", 0.0)
                b = params.get("b", 0.0)
                expr = f"{a:.6g}*x + {b:.6g}"
            elif model == "piecewise":
                x_sens = params.get("x_sens", 0.0)
                x_sat = params.get("x_sat", 0.0)
                eta = params.get("eta", 0.0)
                y_max = params.get("y_max", 0.0)
                expr = (
                    f"(x < {x_sens:.6g} ? 0 : "
                    f"(x <= {x_sat:.6g} ? {eta:.6g}*(x-{x_sens:.6g}) : {y_max:.6g}))"
                )
            elif model == "logistic":
                y_max = params.get("y_max", 0.0)
                a = params.get("a", 0.0)
                b = params.get("b", 0.0)
                expr = f"{y_max:.6g}/(1+exp(-{a:.6g}*(x-{b:.6g})))"
            elif model == "shifted_sigmoid":
                y_max = params.get("y_max", 0.0)
                a = params.get("a", 0.0)
                b = params.get("b", 0.0)
                expr = (
                    f"{y_max:.6g}*(1/(1+exp(-{a:.6g}*(x-{b:.6g})))"
                    f" - (1/(1+exp({a:.6g}*{b:.6g}))))"
                )
            elif model == "polynomial3":
                a1 = params.get("a1", 0.0)
                a2 = params.get("a2", 0.0)
                a3 = params.get("a3", 0.0)
                expr = f"{a1:.6g}*x + {a2:.6g}*x^2 + {a3:.6g}*x^3"
            elif model == "rational":
                a = params.get("a", 0.0)
                b = params.get("b", 0.0)
                expr = f"({a:.6g}*x^2)/(1+{b:.6g}*x)"
            elif model == "log_surrogate":
                a = params.get("a", 0.0)
                b = params.get("b", 0.0)
                expr = f"{a:.6g}*ln(1+{b:.6g}*x)"
            elif model == "power_law":
                a = params.get("a", 0.0)
                k = params.get("k", 0.0)
                expr = f"{a:.6g}*x^{k:.6g}"
            else:
                expr = "0"
            f.write(f"{{{expr}}};\n\n")
        f.write("\\end{axis}\n")
        f.write("\\end{tikzpicture}\n")

plt.tight_layout()
plt.show()

if demo_samples:
    model_funcs = {
        "linear_eta": linear_efficiency_fit,
        "piecewise": piecewise_fit,
        "logistic": logistic_fit,
        "shifted_sigmoid": shifted_sigmoid_fit,
        "polynomial3": polynomial_fit,
        "rational": rational_fit,
        "log_surrogate": log_fit,
        "power_law": power_law_fit,
    }

    for sample in demo_samples:
        demo_x = sample["x"]
        demo_y = sample["y"]
        meta = sample["meta"]

        plt.figure(figsize=(10, 6))
        plt.scatter(demo_x, demo_y, s=marker_size * 6, color="black", label="data")

        x_sorted = np.sort(demo_x)
        demo_results = []
        for model_name in ENABLED_MODELS:
            fit_fn = model_funcs.get(model_name)
            if fit_fn is None:
                continue
            result = fit_fn(demo_x, demo_y)
            params = {k: v for k, v in result.items() if k not in {"y_hat", "r2"}}
            r2 = result.get("r2")
            demo_results.append((model_name, params, r2))

        best_name = None
        best_r2 = None
        for model_name, _, r2 in demo_results:
            if r2 is None or math.isnan(r2):
                continue
            if best_r2 is None or r2 > best_r2:
                best_r2 = r2
                best_name = model_name

        ordered = [item for item in demo_results if item[0] != best_name]
        if best_name is not None:
            ordered.append(next(item for item in demo_results if item[0] == best_name))

        for model_name, params, r2 in ordered:
            y_fit = eval_model(model_name, params, x_sorted)
            r2_text = f"{r2:.3f}" if r2 is not None and not math.isnan(r2) else "nan"
            label = f"{MODEL_LABELS.get(model_name, model_name)} ({model_name}), $R^2$={r2_text}"
            if model_name == best_name:
                label = label + r" $\mathbf{best}$"
            plt.plot(x_sorted, y_fit, label=label)

        plt.xlabel(f"P_rf ({power_unit})")
        plt.xlim(0, None)
        plt.ylim(0, None)
        plt.ylabel(f"P_dc ({power_unit})")
        plt.grid(True)
        title = (
            f"Model fits | {meta['table']} | f={meta['frequency_mhz']} MHz | "
            f"tune={meta['tuning_frequency_mhz']} MHz | "
            f"t={meta['target_voltage_mv']} mV"
        )
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
