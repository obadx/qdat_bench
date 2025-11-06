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
        f"{ph.hamza}{ph.fatha}([{ph.noon}{ph.noon_mokhfah}]{{1,8}}){ph.taa}",
        ph_trans,
    )
    if match:
        if match.group(1)[0] == ph.noon:
            return NoonMokhfahLen.NOON
        elif len(match.group(1)) < 3:
            return NoonMokhfahLen.PARTIAL
        else:
            return NoonMokhfahLen.COMPLETE
    else:
        return NoonMokhfahLen.PARTIAL


def extract_allam_alif_len(
    phonetic_transcript: str, sifat: list[dict] | None = None
) -> int:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(f"{ph.lam}{{2}}{ph.fatha}({ph.alif}{{1,8}}){ph.meem}", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def extract_madd_aared_len(
    phonetic_transcript: str, sifat: list[dict] | None = None
) -> int:
    ph_trans = re.sub(r"\s+", "", phonetic_transcript)
    match = re.search(f"{ph.yaa}.({ph.waw_madd}{{1,8}})(?:{ph.baa}|$)", ph_trans)
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
        name="qalo_waw_len",
        extract_func=extract_qalo_waw_len,
        metrics=[
            QdataBenchRMSE(),
            QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="laa_alif_len",
        extract_func=extract_laa_alif_len,
        metrics=[
            QdataBenchRMSE(),
            QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="separate_madd",
        extract_func=extract_separate_madd,
        metrics=[
            QdataBenchRMSE(),
            QdataBenchR2(),
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
            QdataBenchR2(),
        ],
    ),
    QdataColumAttributes(
        name="madd_aared_len",
        extract_func=extract_madd_aared_len,
        metrics=[
            QdataBenchRMSE(),
            QdataBenchR2(),
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


def main(args):
    # TODO:
    # 1. Align ids
    # 2. compute the rest of the metrics

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

    speech_metrics = compute_speech_metrics(
        level_to_pred_labels=level_to_pred,
        level_to_ref_labels=levels_ref["input_ids"],
        pad_token_idx=0,
    )
    print(json.dumps(speech_metrics, indent=2))

    qdat_metrics = compute_qdat_bench_metrics(pred_trans_ds, qdat_bench_ds)
    print(json.dumps(qdat_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation Results of muaalem model")
    parser.add_argument(
        "--transcription-file",
        help="The parth to the transcription file",
        type=Path,
        default=Path(
            "./assets/muaalem-transcripts/muaalem-model-v3_2_predictions.jsonl"
        ),
    )
    args = parser.parse_args()
    main(args)
