#!/usr/bin/env python3
"""Compare LimiX evaluation folders from two result roots.

Typical usage:

    python script/compare_eval_results.py /path/to/result_a /path/to/result_b
    python script/compare_eval_results.py /path/to/result_a /path/to/result_b -o compare.csv

Each root should contain folders named ``LimiX*`` (e.g. ``LimiX-16M_cls``).
Datasets are inner-joined per folder. Diff = path2 - path1, always signed.
If ``all_rst.csv`` is missing, metrics are computed from ``*_pred_LimiX.csv``
on the datasets present in both folders.
"""

from __future__ import annotations

import argparse
import csv
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, r2_score, roc_auc_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    root_mean_squared_error = partial(mean_squared_error, squared=False)


RESULT_CSV = "all_rst.csv"
PRED_SUFFIX = "_pred_LimiX.csv"
FOLDER_PREFIX = "LimiX"
DATASET_COL = "dataset name"

CLS_METRICS = (("auc", "AUC"), ("acc", "ACC"), ("logloss", "Logloss"))
REG_METRICS = (("r2", "R2"), ("rmse", "rmse"))
ALL_METRICS = CLS_METRICS + REG_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LimiX* eval folders under two result paths.",
    )
    parser.add_argument("path1", nargs="?", help="First result directory")
    parser.add_argument("path2", nargs="?", help="Second result directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: compare_<path1>_vs_<path2>.csv",
    )
    args = parser.parse_args()
    if not args.path1:
        args.path1 = input("请输入第一个结果路径: ").strip()
    if not args.path2:
        args.path2 = input("请输入第二个结果路径: ").strip()
    if not args.path1 or not args.path2:
        parser.error("需要两个结果路径")
    args.path1 = Path(args.path1).expanduser().resolve()
    args.path2 = Path(args.path2).expanduser().resolve()
    return args


def path_label(path: Path, other: Path) -> str:
    if path.name != other.name:
        return path.name
    if path.parent.name:
        return path.parent.name
    return str(path)


def list_limix_folders(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"路径不存在或不是目录: {root}")
    folders = {}
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(FOLDER_PREFIX):
            folders[child.name] = child
    return folders


def sort_folder_names(names: list[str]) -> list[str]:
    def key(name: str) -> tuple[str, int]:
        if name.endswith("_cls"):
            return (name[: -len("_cls")], 0)
        if name.endswith("_reg"):
            return (name[: -len("_reg")], 1)
        return (name, 2)

    return sorted(names, key=key)


def infer_task(folder_name: str, columns: list[str]) -> str:
    lowered = folder_name.lower()
    if lowered.endswith("_cls"):
        return "cls"
    if lowered.endswith("_reg"):
        return "reg"
    colset = {c.strip().lower() for c in columns}
    if {"auc", "acc", "logloss"} & colset:
        return "cls"
    if {"r2", "rmse"} & colset:
        return "reg"
    raise ValueError(f"无法判断任务类型: {folder_name}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(columns: list[str], name: str) -> str | None:
    want = name.strip().lower()
    for col in columns:
        if col.strip().lower() == want:
            return col
    return None


def list_pred_datasets(folder: Path) -> dict[str, Path]:
    datasets = {}
    for path in folder.glob(f"*{PRED_SUFFIX}"):
        if path.is_file():
            datasets[path.name[: -len(PRED_SUFFIX)]] = path
    return datasets


def pred_class_columns(columns: list[str]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for col in columns:
        lowered = col.strip().lower()
        if not lowered.startswith("pred_"):
            continue
        suffix = lowered[5:]
        if suffix.isdigit():
            indexed.append((int(suffix), col))
    indexed.sort(key=lambda item: item[0])
    return [col for _, col in indexed]


def auc_from_pred(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) > 2:
            return float(roc_auc_score(y_true, proba, multi_class="ovo"))
        scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.nan_to_num(proba, nan=0.0), 0.0, None)
    row_sum = proba.sum(axis=1, keepdims=True)
    ok = row_sum.squeeze(-1) > 0
    proba[ok] = proba[ok] / row_sum[ok]
    return proba


def metrics_from_pred(path: Path) -> dict[str, float]:
    df = normalize_columns(pd.read_csv(path))
    label_col = find_column(list(df.columns), "label")
    if label_col is None:
        raise ValueError(f"{path} 缺少 label 列")
    pred_cols = pred_class_columns(list(df.columns))
    pred_col = find_column(list(df.columns), "pred")
    if pred_cols:
        y_true = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
        proba = normalize_proba(
            df[pred_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        )
        y_hat = np.argmax(proba, axis=1)
        try:
            ce = float(log_loss(y_true, proba, labels=np.arange(proba.shape[1])))
        except ValueError:
            ce = float("nan")
        return {
            "auc": auc_from_pred(y_true, proba),
            "acc": float(accuracy_score(y_true, y_hat)),
            "logloss": ce,
        }
    if pred_col is None:
        raise ValueError(f"{path} 没有 pred / pred_* 列")
    y_true = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def load_metrics_from_preds(folder: Path, datasets: list[str] | None = None) -> pd.DataFrame:
    pred_map = list_pred_datasets(folder)
    names = datasets if datasets is not None else sorted(pred_map)
    missing = [name for name in names if name not in pred_map]
    if missing:
        raise FileNotFoundError(f"{folder} 缺少 pred 文件: {', '.join(missing[:5])}")
    if not names:
        raise FileNotFoundError(f"缺少 {RESULT_CSV}，且没有 pred 文件: {folder}")
    rows = []
    for name in names:
        row = metrics_from_pred(pred_map[name])
        row[DATASET_COL] = name
        rows.append(row)
    return pd.DataFrame(rows)


def finalize_metrics(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    df = normalize_columns(df)
    dataset_col = find_column(list(df.columns), DATASET_COL)
    if dataset_col is None:
        raise ValueError(f"{source} 缺少列 {DATASET_COL!r}")
    df[dataset_col] = df[dataset_col].astype(str).str.strip()
    metric_cols = []
    for raw_name, _ in ALL_METRICS:
        col = find_column(list(df.columns), raw_name)
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            metric_cols.append(col)
    if not metric_cols:
        raise ValueError(f"{source} 没有可对比指标列")
    return (
        df.groupby(dataset_col, dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={dataset_col: DATASET_COL})
    )


def load_metrics(folder: Path, datasets: list[str] | None = None) -> pd.DataFrame:
    csv_path = folder / RESULT_CSV
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        if datasets is not None:
            df = normalize_columns(df)
            dataset_col = find_column(list(df.columns), DATASET_COL)
            if dataset_col is None:
                raise ValueError(f"{csv_path} 缺少列 {DATASET_COL!r}")
            wanted = {name.strip() for name in datasets}
            df = df[df[dataset_col].astype(str).str.strip().isin(wanted)]
        return finalize_metrics(df, csv_path)
    return finalize_metrics(load_metrics_from_preds(folder, datasets), folder)


def folder_datasets(folder: Path) -> set[str]:
    csv_path = folder / RESULT_CSV
    if csv_path.is_file():
        df = normalize_columns(pd.read_csv(csv_path))
        dataset_col = find_column(list(df.columns), DATASET_COL)
        if dataset_col is None:
            raise ValueError(f"{csv_path} 缺少列 {DATASET_COL!r}")
        return set(df[dataset_col].astype(str).str.strip())
    return set(list_pred_datasets(folder))


def fmt_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def fmt_diff(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):+.6f}"


def metric_pair(
    row1: pd.Series | None,
    row2: pd.Series | None,
    raw_name: str,
) -> tuple[str, str, str]:
    col1 = find_column(list(row1.index), raw_name) if row1 is not None else None
    col2 = find_column(list(row2.index), raw_name) if row2 is not None else None
    v1 = float(row1[col1]) if row1 is not None and col1 and pd.notna(row1[col1]) else None
    v2 = float(row2[col2]) if row2 is not None and col2 and pd.notna(row2[col2]) else None
    diff = None if v1 is None or v2 is None else v2 - v1
    return fmt_value(v1), fmt_value(v2), fmt_diff(diff)


def empty_metric_fields(label1: str, label2: str) -> dict[str, str]:
    fields = {}
    for _, display in ALL_METRICS:
        fields[f"{display}_{label1}"] = ""
        fields[f"{display}_{label2}"] = ""
        fields[f"{display}_diff"] = ""
    return fields


def fill_metrics(
    fields: dict[str, str],
    task: str,
    row1: pd.Series | None,
    row2: pd.Series | None,
    label1: str,
    label2: str,
) -> None:
    metrics = CLS_METRICS if task == "cls" else REG_METRICS
    for raw_name, display in metrics:
        v1, v2, diff = metric_pair(row1, row2, raw_name)
        fields[f"{display}_{label1}"] = v1
        fields[f"{display}_{label2}"] = v2
        fields[f"{display}_diff"] = diff


def compare_folder(
    name: str,
    folder1: Path,
    folder2: Path,
    label1: str,
    label2: str,
) -> list[dict[str, str]]:
    aligned = sorted(folder_datasets(folder1) & folder_datasets(folder2))
    if not aligned:
        print(f"[skip] {name}: 没有对齐的 dataset", file=sys.stderr)
        return []
    missing = [
        label
        for label, folder in ((label1, folder1), (label2, folder2))
        if not (folder / RESULT_CSV).is_file()
    ]
    if missing:
        print(
            f"[info] {name}: 缺少 {RESULT_CSV} ({', '.join(missing)})，"
            f"对 {len(aligned)} 个共有 dataset 从 pred 计算指标",
            file=sys.stderr,
        )
    df1 = load_metrics(folder1, aligned)
    df2 = load_metrics(folder2, aligned)
    task = infer_task(name, list(df1.columns) + list(df2.columns))

    left = df1.set_index(DATASET_COL).loc[aligned]
    right = df2.set_index(DATASET_COL).loc[aligned]
    mean1 = left.mean(numeric_only=True)
    mean2 = right.mean(numeric_only=True)

    rows: list[dict[str, str]] = []
    mean_row = {
        "model": name,
        "task": task,
        "dataset": "MEAN",
        "n_aligned": str(len(aligned)),
        **empty_metric_fields(label1, label2),
    }
    fill_metrics(mean_row, task, mean1, mean2, label1, label2)
    rows.append(mean_row)

    for dataset in aligned:
        detail = {
            "model": name,
            "task": task,
            "dataset": dataset,
            "n_aligned": "",
            **empty_metric_fields(label1, label2),
        }
        fill_metrics(detail, task, left.loc[dataset], right.loc[dataset], label1, label2)
        rows.append(detail)
    return rows


def output_columns(label1: str, label2: str) -> list[str]:
    cols = ["model", "task", "dataset", "n_aligned"]
    for _, display in ALL_METRICS:
        cols.extend(
            [f"{display}_{label1}", f"{display}_{label2}", f"{display}_diff"]
        )
    return cols


def print_mean_preview(rows: list[dict[str, str]], label1: str, label2: str) -> None:
    means = [row for row in rows if row["dataset"] == "MEAN"]
    if not means:
        return
    print(f"对齐均值  diff = {label2} - {label1}")
    for row in means:
        bits = [f"{row['model']} [{row['task']}] n={row['n_aligned']}"]
        metrics = CLS_METRICS if row["task"] == "cls" else REG_METRICS
        for _, display in metrics:
            bits.append(
                f"{display} {row[f'{display}_{label1}']} vs {row[f'{display}_{label2}']} "
                f"({row[f'{display}_diff']})"
            )
        print("  " + " | ".join(bits))


def main() -> int:
    args = parse_args()
    folders1 = list_limix_folders(args.path1)
    folders2 = list_limix_folders(args.path2)
    common = sort_folder_names(sorted(set(folders1) & set(folders2)))
    only1 = sorted(set(folders1) - set(folders2))
    only2 = sorted(set(folders2) - set(folders1))

    if only1:
        print(f"仅在 path1 中: {', '.join(only1)}", file=sys.stderr)
    if only2:
        print(f"仅在 path2 中: {', '.join(only2)}", file=sys.stderr)
    if not common:
        print("两个路径下没有可对比的 LimiX 文件夹", file=sys.stderr)
        return 1

    label1 = path_label(args.path1, args.path2)
    label2 = path_label(args.path2, args.path1)
    if label1 == label2:
        label1, label2 = "path1", "path2"

    rows: list[dict[str, str]] = []
    for name in common:
        rows.extend(
            compare_folder(name, folders1[name], folders2[name], label1, label2)
        )
    if not rows:
        print("没有可写入的对比行", file=sys.stderr)
        return 1

    output = args.output
    if output is None:
        output = Path(f"compare_{label1}_vs_{label2}.csv")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = output_columns(label1, label2)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_mean_preview(rows, label1, label2)
    print(f"已写入 {output}  ({len(rows)} 行, {len(common)} 个模型文件夹)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
