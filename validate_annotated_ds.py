import json
import re

from qdat_bench.data_models import (
    QdataBenchItem,
    NoonMoshaddahLen,
    NoonMokhfahLen,
    Qalqalah,
)
from quran_transcript.alphabet import phonetics as ph


def validate_qalo_alif_len(item: QdataBenchItem) -> int:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
    match = re.search(f"{ph.qaf}{ph.fatha}({ph.alif}{{1,8}}){ph.lam}", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def validate_qalo_waw_len(item: QdataBenchItem) -> int:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
    match = re.search(f"{ph.lam}{ph.dama}({ph.waw_madd}{{1,8}})", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def validate_laa_alif_len(item: QdataBenchItem) -> int:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
    match = re.search(f"{ph.lam}{ph.fatha}({ph.alif}{{1,8}}){ph.ayn}", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def validate_separate_madd(item: QdataBenchItem) -> int:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
    match = re.search(
        f"{ph.lam}{ph.fatha}{ph.noon}{ph.fatha}({ph.alif}{{1,8}})", ph_trans
    )
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def validate_noon_moshaddadah_len(item: QdataBenchItem) -> NoonMoshaddahLen:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
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


def validate_noon_mokhfah_len(item: QdataBenchItem) -> NoonMokhfahLen:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
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


def validate_allam_alif_len(item: QdataBenchItem) -> int:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
    match = re.search(f"{ph.lam}{{2}}{ph.fatha}({ph.alif}{{1,8}}){ph.meem}", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def validate_madd_aared_len(item: QdataBenchItem) -> int:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
    match = re.search(f"{ph.yaa}.({ph.waw_madd}{{1,8}})(?:{ph.baa}|$)", ph_trans)
    if match:
        madd_len = len(match.group(1))
    else:
        madd_len = 0
    return madd_len


def validate_qalqalah(item: QdataBenchItem) -> Qalqalah:
    ph_trans = re.sub(r"\s+", "", item.phonetic_transcript)
    match = re.search(
        f"{ph.baa}{ph.qlqla}",
        ph_trans,
    )
    table_qalqalah = item.sifat[-1].qalqla
    if match and table_qalqalah == "moqalqal" and item.sifat[-1].phonemes[0] == ph.baa:
        return Qalqalah.HAS_QALQALAH
    else:
        return Qalqalah.NO_QALQALH


ATTR_TO_VAL_FUNC = {
    "qalo_alif_len": validate_qalo_alif_len,
    "qalo_waw_len": validate_qalo_waw_len,
    "laa_alif_len": validate_laa_alif_len,
    "separate_madd": validate_separate_madd,
    "noon_moshaddadah_len": validate_noon_moshaddadah_len,
    "noon_mokhfah_len": validate_noon_mokhfah_len,
    "allam_alif_len": validate_allam_alif_len,
    "madd_aared_len": validate_madd_aared_len,
    "qalqalah": validate_qalqalah,
}


def validate_item(item: QdataBenchItem) -> bool:
    has_error = False
    for attr, val_func in ATTR_TO_VAL_FUNC.items():
        exp_val = getattr(item, attr)
        script_val = val_func(item)
        if exp_val != script_val:
            has_error = True
            print(
                f"Error in `{item.original_id}` in attribute: `{attr}`. script_val: `{script_val}`, expecpted_val: `{exp_val}`"
            )
            print("-" * 30)

    return has_error


if __name__ == "__main__":
    with open("./qdat_bench_annotations.json", "r", encoding="utf-8") as f:
        id_to_item = json.load(f)

    error_counter = 0
    for item in id_to_item.values():
        item = QdataBenchItem(**item)
        has_error = validate_item(item)
        if has_error:
            print("*" * 50)
            print("\n" * 2)
        error_counter += 1 if has_error else 0
