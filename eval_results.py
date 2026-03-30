import argparse
from pathlib import Path
import string
import re
from dataclasses import dataclass
from typing import Callable
import json

import numpy as np
import Levenshtein
from datasets import Dataset, load_dataset
from quran_muaalem.modeling.multi_level_tokenizer import MultiLevelTokenizer
from quran_transcript.alphabet import phonetics as ph
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    r2_score,
    mean_squared_error,
)

from qdat_bench.data_models import (
    QdataBenchItem,
    NoonMoshaddahLen,
    NoonMokhfahLen,
    Qalqalah,
)


def extract_qalo_alif_len(
    phonetic_transcript: str, sifat: list[dict] | None = None
) -> int:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(f"{ph.qaf}{ph.fatha}({ph.alif}{{1,8}}){ph.lam}", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def extract_qalo_waw_len(
    phonetic_transcript: str, sifat: list[dict] | None = None
) -> int:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(f"{ph.lam}{ph.dama}({ph.waw_madd}{{1,8}})", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def extract_laa_alif_len(
    phonetic_transcript: str, sifat: list[dict] | None = None
) -> int:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(f"{ph.lam}{ph.fatha}({ph.alif}{{1,8}}){ph.ayn}", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def extract_separate_madd(
    phonetic_transcript: str, sifat: list[dict] | None = None
) -> int:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(
        f"{ph.lam}{ph.fatha}{ph.noon}{ph.fatha}({ph.alif}{{1,8}})", ph_trans
    )
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def extract_noon_moshaddadah_len(
    phonetic_transcript: str, sifat: list | None = None
) -> NoonMoshaddahLen:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(
        f"{ph.hamza}{ph.kasra}({ph.noon}{{1,8}}){ph.fatha}{ph.kaf}", ph_trans
    )
    if match:
        noon_len = len(match.group(1))
        if noon_len < 4:
            return NoonMoshaddahLen.PARTIAL
        else:
            return NoonMoshaddahLen.COMPLETE
    else:
        return NoonMoshaddahLen.PARTIAL


def extract_noon_mokhfah_len(
    phonetic_transcript: str, sifat: list | None = None
) -> NoonMokhfahLen:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(
        f"({ph.noon_mokhfah}{{1,8}})",
        ph_trans,
    )
    if match:
        if len(match.group(1)) < 3:
            return NoonMokhfahLen.PARTIAL
        else:
            return NoonMokhfahLen.COMPLETE
    else:
        return NoonMokhfahLen.NOON


def extract_allam_alif_len(
    phonetic_transcript: str, sifat: list[dict] | None = None
) -> int:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(f"{ph.lam}{ph.fatha}({ph.alif}{{1,8}}){ph.meem}", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def extract_madd_aared_len(
    phonetic_transcript: str, sifat: list[dict] | None = None
) -> int:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(f"({ph.waw_madd}{{1,8}}).?.?{ph.qlqla}?$", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


# TODO: parse sifat
def extract_qalqalah(phonetic_transcript: str, sifat: list | None = None) -> Qalqalah:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(
        f"{ph.baa}{ph.qlqla}",
        ph_trans,
    )
    if match:
        return Qalqalah.HAS_QALQALAH
    else:
        return Qalqalah.NO_QALQALH
    # table_qalqalah = item.sifat[-1].qalqla
    # if match and phonetic_transcript_only:
    #     return Qalqalah.HAS_QALQALAH
    # elif (
    #     match and table_qalqalah == "moqalqal" and item.sifat[-1].phonemes[0] == ph.baa
    # ):
    #     return Qalqalah.HAS_QALQALAH
    # else:
    #     return Qalqalah.NO_QALQALH


@dataclass
class QdataBenchMetric:
    name: str

    def compute(
        self,
        pred: list[int],
        ref: list[int],
        col_name: str,
        idx2label: list[str] | dict[int, str] | None = None,
    ) -> dict:
        raise NotImplementedError


@dataclass
class QdataBenchRecall(QdataBenchMetric):
    name: str = "recall"

    def compute(
        self,
        pred: list[int],
        ref: list[int],
        col_name: str,
        idx2label: list[str] | dict[int, str] | None = None,
    ) -> dict:
        assert isinstance(idx2label, list) or isinstance(idx2label, dict), (
            "You have to supply `idx2label for recall"
        )

        # Compute recall for each class
        recalls = recall_score(ref, pred, average=None, zero_division=0)

        # Create result dictionary
        result = {}
        for i, recall in enumerate(recalls):
            result[f"{col_name}_recall_{idx2label[i]}"] = float(recall)

        # Compute average recall (macro average)
        result[f"{col_name}_avg_recall"] = recall_score(
            ref, pred, average="macro", zero_division=0
        )

        return result


@dataclass
class QdataBenchPrecision(QdataBenchMetric):
    name: str = "precision"

    def compute(
        self,
        pred: list[int],
        ref: list[int],
        col_name: str,
        idx2label: list[str] | dict[int, str] | None = None,
    ) -> dict:
        assert isinstance(idx2label, list) or isinstance(idx2label, dict), (
            "You have to supply `idx2label for precision"
        )

        # Compute precision for each class
        precisions = precision_score(ref, pred, average=None, zero_division=0)

        # Create result dictionary
        result = {}
        for i, precision in enumerate(precisions):
            result[f"{col_name}_precision_{idx2label[i]}"] = float(precision)

        # Compute average precision (macro average)
        result[f"{col_name}_avg_precision"] = precision_score(
            ref, pred, average="macro", zero_division=0
        )

        return result


@dataclass
class QdataBenchF1(QdataBenchMetric):
    name: str = "f1"

    def compute(
        self,
        pred: list[int],
        ref: list[int],
        col_name: str,
        idx2label: list[str] | dict[int, str] | None = None,
    ) -> dict:
        assert isinstance(idx2label, list) or isinstance(idx2label, dict), (
            "You have to supply `idx2label for f1"
        )

        # Compute F1 for each class
        f1_scores = f1_score(ref, pred, average=None, zero_division=0)

        # Create result dictionary
        result = {}
        for i, f1 in enumerate(f1_scores):
            result[f"{col_name}_f1_{idx2label[i]}"] = float(f1)

        # Compute macro F1 score
        result[f"{col_name}_macro_f1"] = f1_score(
            ref, pred, average="macro", zero_division=0
        )

        return result


@dataclass
class QdataBenchAccuracy(QdataBenchMetric):
    name: str = "accuracy"

    def compute(
        self,
        pred: list[int],
        ref: list[int],
        col_name: str,
        idx2label: list[str] | dict[int, str] | None = None,
    ) -> dict:
        accuracy = accuracy_score(ref, pred)
        return {f"{col_name}_accuracy": float(accuracy)}


@dataclass
class QdataBenchR2(QdataBenchMetric):
    name: str = "r2"

    def compute(
        self,
        pred: list[float],
        ref: list[float],
        col_name: str,
        idx2label: list[str] | dict[int, str] | None = None,
    ) -> dict:
        r2 = r2_score(ref, pred)
        return {f"{col_name}_r2": float(r2)}


@dataclass
class QdataBenchRMSE(QdataBenchMetric):
    name: str = "rmse"

    def compute(
        self,
        pred: list[float],
        ref: list[float],
        col_name: str,
        idx2label: list[str] | dict[int, str] | None = None,
    ) -> dict:
        mse = mean_squared_error(ref, pred)
        rmse = np.sqrt(mse)
        return {f"{col_name}_rmse": float(rmse)}


@dataclass
class QdataColumAttributes:
    name: str
    extract_func: Callable
    metrics: list[QdataBenchMetric]
    idx2label: list[str] | dict[int, str] | None = None


EVAL_COLMS = [
    QdataColumAttributes(
        name="qalo_alif_len",
        extract_func=extract_qalo_alif_len,
        metrics=[
            QdataBenchRMSE(),
            # QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="qalo_waw_len",
        extract_func=extract_qalo_waw_len,
        metrics=[
            QdataBenchRMSE(),
            # QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="laa_alif_len",
        extract_func=extract_laa_alif_len,
        metrics=[
            QdataBenchRMSE(),
            # QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="separate_madd",
        extract_func=extract_separate_madd,
        metrics=[
            QdataBenchRMSE(),
            # QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="noon_moshaddadah_len",
        idx2label={member.value: member.name for member in NoonMoshaddahLen},
        extract_func=extract_noon_moshaddadah_len,
        metrics=[
            QdataBenchRecall(),
            QdataBenchPrecision(),
            QdataBenchF1(),
            QdataBenchAccuracy(),
        ],
    ),
    QdataColumAttributes(
        name="noon_mokhfah_len",
        idx2label={member.value: member.name for member in NoonMokhfahLen},
        extract_func=extract_noon_mokhfah_len,
        metrics=[
            QdataBenchRecall(),
            QdataBenchPrecision(),
            QdataBenchF1(),
            QdataBenchAccuracy(),
        ],
    ),
    QdataColumAttributes(
        name="allam_alif_len",
        extract_func=extract_allam_alif_len,
        metrics=[
            QdataBenchRMSE(),
            # QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="madd_aared_len",
        extract_func=extract_madd_aared_len,
        metrics=[
            QdataBenchRMSE(),
            # QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="qalqalah",
        idx2label={member.value: member.name for member in Qalqalah},
        extract_func=extract_qalqalah,
        metrics=[
            QdataBenchRecall(),
            QdataBenchPrecision(),
            QdataBenchF1(),
            QdataBenchAccuracy(),
        ],
    ),
]


# ATTR_TO_VAL_FUNC = {
#     "qalo_alif_len": extract_qalo_alif_len,
#     "qalo_waw_len": extract_qalo_waw_len,
#     "laa_alif_len": extract_laa_alif_len,
#     "separate_madd": extract_separate_madd,
#     "noon_moshaddadah_len": extract_noon_moshaddadah_len,
#     "noon_mokhfah_len": extract_noon_mokhfah_len,
#     "allam_alif_len": extract_allam_alif_len,
#     "madd_aared_len": extract_madd_aared_len,
#     "qalqalah": extract_qalqalah,
# }
#
#
# COL_TO_METRICS = {
#     "qalo_alif_len": ["accuracy", "f1", "recall", "precisoin"],
# }


def compute_qdat_bench_metrics(pred_trans_ds: Dataset, qdat_bench_ds: Dataset):
    pred_level_to_scripts_list = pred_trans_ds["level_to_scripts"]
    metrics_dict = {}
    for col_attr in EVAL_COLMS:
        pred_list = []
        for pred_level_to_scripts in pred_level_to_scripts_list:
            pred_list.append(col_attr.extract_func(pred_level_to_scripts["phonemes"]))

        ref_list = qdat_bench_ds[col_attr.name]
        for metric in col_attr.metrics:
            metrics_dict |= metric.compute(
                pred_list,
                ref_list,
                col_name=col_attr.name,
                idx2label=col_attr.idx2label,
            )

    return metrics_dict


def compute_qdat_bench_average_metrics(name_to_metric: dict[str, int | float]):
    def compute_avg_metric(columns: list[str] | str):
        if not isinstance(columns, list):
            columns = [columns]
        total_sum = 0
        for col in columns:
            total_sum += name_to_metric[col]
        return total_sum / len(columns)

    f1_columns = [
        "noon_moshaddadah_len_macro_f1",
        "noon_mokhfah_len_macro_f1",
        "qalqalah_macro_f1",
    ]
    acc_columns = [
        "noon_moshaddadah_len_accuracy",
        "noon_mokhfah_len_accuracy",
        "qalqalah_accuracy",
    ]
    rmse_normal_madd_columns = [
        "qalo_alif_len_rmse",
        "qalo_waw_len_rmse",
        "laa_alif_len_rmse",
        "allam_alif_len_rmse",
    ]
    rmse_columns = rmse_normal_madd_columns + [
        "separate_madd_rmse",
        "madd_aared_len_rmse",
    ]

    return {
        "per_phonemes": name_to_metric["per_phonemes"],
        "avg_per": name_to_metric["average_per"],
        "avg_tajweed_f1": compute_avg_metric(f1_columns),
        "avg_tajweed_acc": compute_avg_metric(acc_columns),
        "avg_madd_rmse": compute_avg_metric(rmse_columns),
        "avg_nromal_madd_rmse": compute_avg_metric(rmse_normal_madd_columns),
        "rmse_madd_aared": name_to_metric["madd_aared_len_rmse"],
        "rmse_separate_madd": name_to_metric["separate_madd_rmse"],
    }


def sequence_to_chars(labels) -> str:
    t = ""
    for label in labels:
        if label > len(string.ascii_letters):
            raise ValueError(
                f"We only support labels up to : `{len(string.ascii_letters)}` got {label}"
            )
        t += string.ascii_letters[label]
    return t


def compute_per_level(predictions: list[list[int]], references: list[list[int]]):
    """
    Compute Phoneme Error Rate using Levenshtein distance.
    """
    total_distance = 0
    total_length = 0

    pred_ids_list = predictions
    ref_ids_list = references

    for pred_seq, ref_seq in zip(pred_ids_list, ref_ids_list):
        pred_str = sequence_to_chars(pred_seq)
        ref_str = sequence_to_chars(ref_seq)
        # Compute Levenshtein distance
        distance = min(Levenshtein.distance(pred_str, ref_str), len(ref_str))
        total_distance += distance
        total_length += len(ref_str)

    return total_distance / total_length if total_length > 0 else 0.0


def compute_speech_metrics(
    level_to_pred_labels: dict[str, list[list[int]]],
    level_to_ref_labels: dict[str, list[list[int]]],
    pad_token_idx=0,
):
    """
    Compute PER metrics for multi-level predictions.

    Args:
        eval_pred: Tuple of (predictions, labels) where both are dictionaries
        multi_level_tokenizer: MultiLevelTokenizer instance for decoding

    Returns:
        Dictionary with PER metrics for each level and average
    """

    # remove pading tokens
    metrics = {}

    for level in level_to_ref_labels:
        metrics[f"per_{level}"] = compute_per_level(
            level_to_pred_labels[level],
            level_to_ref_labels[level],
        )

    # computing average per
    total_per = 0.0
    N = 0
    for key in metrics:
        total_per += metrics[key]
        N += 1

    metrics["average_per"] = total_per / N

    # Compute PER for each leve
    return metrics


def align_preditc_ds(pred_ds: Dataset, ref_ds_ids: list[str]) -> Dataset:
    pred_id_col_to_idx = {id_key: idx for idx, id_key in enumerate(pred_ds["id"])}
    aligend_pred_ids = [pred_id_col_to_idx[ref_id] for ref_id in ref_ds_ids]
    aligend_pred_ds = pred_ds.select(aligend_pred_ids)

    # None Necessary step but for confirmation
    assert aligend_pred_ds["id"] == ref_ds_ids
    return aligend_pred_ds


def compute_per_sample(pred_seq: list[int], ref_seq: list[int]) -> float:
    pred_str = sequence_to_chars(pred_seq)
    ref_str = sequence_to_chars(ref_seq)
    distance = min(Levenshtein.distance(pred_str, ref_str), len(ref_str))
    ref_len = len(ref_str)
    return distance / ref_len if ref_len > 0 else 0.0


def compute_sample_level_speech_metrics(
    level_to_pred_labels: dict[str, list[list[int]]],
    level_to_ref_labels: dict[str, list[list[int]]],
) -> dict[str, list[float]]:
    num_samples = len(next(iter(level_to_ref_labels.values())))
    sample_metrics = {}
    for level in level_to_ref_labels:
        per_values = []
        for i in range(num_samples):
            per = compute_per_sample(
                level_to_pred_labels[level][i],
                level_to_ref_labels[level][i],
            )
            per_values.append(per)
        sample_metrics[f"per_{level}"] = per_values
    return sample_metrics


def compute_sample_level_qdat_metrics(
    pred_trans_ds: Dataset,
    qdat_bench_ds: Dataset,
) -> dict[str, list]:
    pred_level_to_scripts_list = pred_trans_ds["level_to_scripts"]
    sample_metrics = {}
    for col_attr in EVAL_COLMS:
        pred_list = []
        for pred_level_to_scripts in pred_level_to_scripts_list:
            pred_list.append(col_attr.extract_func(pred_level_to_scripts["phonemes"]))
        ref_list = qdat_bench_ds[col_attr.name]
        sample_metrics[col_attr.name] = {
            "pred": pred_list,
            "ref": list(ref_list),
        }
    return sample_metrics


def compute_bootstrap_metrics(
    level_to_pred_labels: dict[str, list[list[int]]],
    level_to_ref_labels: dict[str, list[list[int]]],
    pred_trans_ds: Dataset,
    qdat_bench_ds: Dataset,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    np.random.seed(seed)
    num_samples = len(next(iter(level_to_ref_labels.values())))
    speech_sample_metrics = compute_sample_level_speech_metrics(
        level_to_pred_labels, level_to_ref_labels
    )
    qdat_sample_metrics = compute_sample_level_qdat_metrics(
        pred_trans_ds, qdat_bench_ds
    )
    classification_cols = ["noon_moshaddadah_len", "noon_mokhfah_len", "qalqalah"]
    bootstrap_samples = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(num_samples, size=num_samples, replace=True)
        sample = {}

        for level_name, per_values in speech_sample_metrics.items():
            sampled_per = [per_values[i] for i in indices]
            sample[level_name] = np.mean(sampled_per)

        per_values_list = [sample[k] for k in sample if k.startswith("per_")]
        sample["average_per"] = np.mean(per_values_list)

        for col_name, data in qdat_sample_metrics.items():
            pred_arr = np.array(data["pred"])
            ref_arr = np.array(data["ref"])
            pred_sampled = pred_arr[indices]
            ref_sampled = ref_arr[indices]

            if col_name in classification_cols:
                idx2label = None
                for col_attr in EVAL_COLMS:
                    if col_attr.name == col_name:
                        idx2label = col_attr.idx2label
                        break

                if idx2label is not None:
                    for class_idx, class_name in idx2label.items():
                        ref_binary = (ref_sampled == class_idx).astype(int)
                        pred_binary = (pred_sampled == class_idx).astype(int)
                        tp = np.sum((pred_binary == 1) & (ref_binary == 1))
                        fn = np.sum((pred_binary == 0) & (ref_binary == 1))
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        sample[f"{col_name}_recall_{class_name}"] = recall

                        fp = np.sum((pred_binary == 1) & (ref_binary == 0))
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        sample[f"{col_name}_precision_{class_name}"] = precision

                        f1 = (
                            2 * (precision * recall) / (precision + recall)
                            if (precision + recall) > 0
                            else 0.0
                        )
                        sample[f"{col_name}_f1_{class_name}"] = f1

                        correct = np.sum(pred_binary == ref_binary)
                        total = len(ref_binary)
                        sample[f"{col_name}_accuracy_{class_name}"] = (
                            correct / total if total > 0 else 0.0
                        )

                    recalls = [
                        sample[f"{col_name}_recall_{c}"] for c in idx2label.values()
                    ]
                    sample[f"{col_name}_avg_recall"] = np.mean(recalls)

                    precisions = [
                        sample[f"{col_name}_precision_{c}"] for c in idx2label.values()
                    ]
                    sample[f"{col_name}_avg_precision"] = np.mean(precisions)

                    f1s = [sample[f"{col_name}_f1_{c}"] for c in idx2label.values()]
                    sample[f"{col_name}_macro_f1"] = np.mean(f1s)

                    correct = np.sum(pred_sampled == ref_sampled)
                    sample[f"{col_name}_accuracy"] = correct / len(ref_sampled)
            else:
                mse = np.mean((pred_sampled - ref_sampled) ** 2)
                sample[f"{col_name}_rmse"] = np.sqrt(mse)

        bootstrap_samples.append(sample)

    all_metric_names = set()
    for s in bootstrap_samples:
        all_metric_names.update(s.keys())

    means = {}
    stds = {}
    for metric_name in sorted(all_metric_names):
        values = [s.get(metric_name, np.nan) for s in bootstrap_samples]
        means[metric_name] = float(np.nanmean(values))
        stds[metric_name] = float(np.nanstd(values))

    return {"bootstrap_samples": bootstrap_samples, "means": means, "stds": stds}


def group_metrics_for_violin_plots(metric_names: list[str]) -> dict[str, list[str]]:
    per_metrics = []
    rmse_metrics = []
    percentage_metrics = []
    for name in metric_names:
        name_lower = name.lower()
        if "per_" in name_lower or "average_per" in name_lower:
            per_metrics.append(name)
        elif "_rmse" in name_lower:
            rmse_metrics.append(name)
        elif any(x in name_lower for x in ["recall", "precision", "f1", "accuracy"]):
            percentage_metrics.append(name)
    return {
        "per_metrics": sorted(per_metrics),
        "rmse_metrics": sorted(rmse_metrics),
        "percentage_metrics": sorted(percentage_metrics),
    }


def run_bootstrap_analysis(
    level_to_pred_labels: dict[str, list[list[int]]],
    level_to_ref_labels: dict[str, list[list[int]]],
    pred_trans_ds: Dataset,
    qdat_bench_ds: Dataset,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    print(f"Running bootstrap analysis with {n_bootstrap} iterations...")
    bootstrap_results = compute_bootstrap_metrics(
        level_to_pred_labels=level_to_pred_labels,
        level_to_ref_labels=level_to_ref_labels,
        pred_trans_ds=pred_trans_ds,
        qdat_bench_ds=qdat_bench_ds,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    all_metric_names = list(bootstrap_results["means"].keys())
    metric_categories = group_metrics_for_violin_plots(all_metric_names)
    bootstrap_results["metric_categories"] = metric_categories

    print("\n=== Bootstrap Results Summary ===")
    print("\nPER Metrics (lower is better):")
    for metric in metric_categories["per_metrics"]:
        print(
            f"  {metric}: {bootstrap_results['means'][metric]:.4f} +/- {bootstrap_results['stds'][metric]:.4f}"
        )

    print("\nRMSE Metrics (lower is better):")
    for metric in metric_categories["rmse_metrics"]:
        print(
            f"  {metric}: {bootstrap_results['means'][metric]:.4f} +/- {bootstrap_results['stds'][metric]:.4f}"
        )

    print("\nPercentage Metrics (higher is better):")
    for metric in metric_categories["percentage_metrics"]:
        print(
            f"  {metric}: {bootstrap_results['means'][metric]:.4f} +/- {bootstrap_results['stds'][metric]:.4f}"
        )

    return bootstrap_results


def main(args):
    # Load datasets
    pred_trans_ds = Dataset.from_json(str(args.transcription_file))
    qdat_bench_ds = load_dataset("obadx/qdat_bench")["train"]
    pred_trans_ds = align_preditc_ds(pred_trans_ds, qdat_bench_ds["id"])

    multi_level_tokenizer = MultiLevelTokenizer("obadx/muaalem-model-v3_2")
    levels_ref = multi_level_tokenizer.tokenize(
        [re.sub(r"\s+", "", ph) for ph in qdat_bench_ds["phonetic_transcript"]],
        qdat_bench_ds["sifat"],
        to_dict=True,
    )
    levels = list(multi_level_tokenizer.vocab.keys())
    level_to_pred = {}
    for level in levels:
        level_to_pred[level] = []
        for item in pred_trans_ds["levels_labels"]:
            level_to_pred[level].append(item[level])

    args.save_dir.mkdir(exist_ok=True)
    save_path = args.save_dir / f"result_{args.transcription_file.stem}.json"

    # Always compute standard metrics
    speech_metrics = compute_speech_metrics(
        level_to_pred_labels=level_to_pred,
        level_to_ref_labels=levels_ref["input_ids"],
        pad_token_idx=0,
    )
    print(json.dumps(speech_metrics, indent=2))

    qdat_metrics = compute_qdat_bench_metrics(pred_trans_ds, qdat_bench_ds)
    print(json.dumps(qdat_metrics, indent=2))

    qdat_avg_metrics = compute_qdat_bench_average_metrics(qdat_metrics | speech_metrics)
    print(json.dumps(qdat_avg_metrics, indent=2))

    to_save_metrics = {
        "speech_metrics": speech_metrics,
        "qdat_metrics": qdat_metrics,
        "qdat_avg_metrics": qdat_avg_metrics,
    }
    with open(save_path, "w") as f:
        json.dump(to_save_metrics, f, indent=2)
    print(f"Result is saved in {save_path}")

    # Run bootstrap analysis if requested
    if args.bootstrap:
        bootstrap_results = run_bootstrap_analysis(
            level_to_pred_labels=level_to_pred,
            level_to_ref_labels=levels_ref["input_ids"],
            pred_trans_ds=pred_trans_ds,
            qdat_bench_ds=qdat_bench_ds,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

        # Compute bootstrap average metrics
        bootstrap_avg_metrics_samples = []
        for sample in bootstrap_results["bootstrap_samples"]:
            avg_sample = compute_qdat_bench_average_metrics(sample)
            bootstrap_avg_metrics_samples.append(avg_sample)

        # Compute mean and std for avg metrics
        avg_means = {}
        avg_stds = {}
        for key in bootstrap_avg_metrics_samples[0]:
            values = [s[key] for s in bootstrap_avg_metrics_samples]
            avg_means[f"{key}_mean"] = float(np.mean(values))
            avg_stds[f"{key}_std"] = float(np.std(values))

        # Organize bootstrapped metrics into categories
        speech_means = {
            k: v
            for k, v in bootstrap_results["means"].items()
            if k.startswith("per_") or k == "average_per"
        }
        speech_stds = {
            k: v
            for k, v in bootstrap_results["stds"].items()
            if k.startswith("per_") or k == "average_per"
        }

        qdat_means = {
            k: v for k, v in bootstrap_results["means"].items() if k not in speech_means
        }
        qdat_stds = {
            k: v for k, v in bootstrap_results["stds"].items() if k not in speech_stds
        }

        bootstrapped_speech_metrics = {}
        for k in speech_means:
            bootstrapped_speech_metrics[f"{k}_mean"] = speech_means[k]
            bootstrapped_speech_metrics[f"{k}_std"] = speech_stds[k]

        bootstrapped_qdat_metrics = {}
        for k in qdat_means:
            bootstrapped_qdat_metrics[f"{k}_mean"] = qdat_means[k]
            bootstrapped_qdat_metrics[f"{k}_std"] = qdat_stds[k]

        bootstrapped_avg_metrics = {}
        for k in avg_means:
            bootstrapped_avg_metrics[k] = avg_means[k]
            bootstrapped_avg_metrics[k.replace("_mean", "_std")] = avg_stds[
                k.replace("_mean", "_std")
            ]

        # Update to_save_metrics with bootstrapped results
        to_save_metrics["bootstrapped_speech_metrics"] = bootstrapped_speech_metrics
        to_save_metrics["bootstrapped_qdat_metrics"] = bootstrapped_qdat_metrics
        to_save_metrics["bootstrapped_avg_metrics"] = bootstrapped_avg_metrics

        # Save updated results
        with open(save_path, "w") as f:
            json.dump(to_save_metrics, f, indent=2)
        print(f"Updated results with bootstrap saved to {save_path}")

        # Save avg_metrics bootstrap samples separately for plotting
        bootstrap_samples_path = (
            args.save_dir
            / f"result_{args.transcription_file.stem}_bootstrap_avg_samples.json"
        )
        print(f"Saving bootstrap avg samples to {bootstrap_samples_path}...")
        with open(bootstrap_samples_path, "w") as f:
            json.dump(bootstrap_avg_metrics_samples, f)
        print(f"Bootstrap avg samples saved to {bootstrap_samples_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation Results of muaalem model")
    parser.add_argument(
        "--transcription-file",
        help="The path to the transcription file",
        type=Path,
        default=Path(
            "./assets/muaalem-transcripts/muaalem-model-v3_2_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--save-dir",
        help="The path to save results",
        type=Path,
        default=Path("./assets/results"),
    )
    parser.add_argument(
        "--bootstrap",
        help="Run bootstrap analysis (10,000 iterations by default)",
        action="store_true",
    )
    parser.add_argument(
        "--n-bootstrap",
        help="Number of bootstrap iterations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for bootstrap analysis",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    main(args)
