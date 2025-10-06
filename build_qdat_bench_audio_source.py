import os

from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
import numpy as np


def fileter_ids(ids: list[str], seed=42, forced_ids: list[str] = []) -> list[str]:
    """Chose id randomly out every reciter

    Gived ids as f"{reciter_id}_{sample_id}" we chose only single
    sample for every reciter
    """
    forced_reciciter_to_smaple = {idx.split("_")[0]: idx for idx in forced_ids}
    reciter_to_sample = {}
    for idx in ids:
        reciter_id = idx.split("_")[0]
        sample_id = idx.split("_")[1]
        if reciter_id not in reciter_to_sample:
            reciter_to_sample[reciter_id] = []
        reciter_to_sample[reciter_id].append(idx)

    # select sample fro every reciter
    selected_ids = []
    rng = np.random.default_rng(seed=seed)
    for reciter_id, sample_ids in reciter_to_sample.items():
        choice = rng.choice(sample_ids)
        if reciter_id in forced_reciciter_to_smaple:
            choice = forced_reciciter_to_smaple[reciter_id]

        selected_ids.append(choice)

    return selected_ids


def chose_single_source(
    ds: Dataset,
    seed=42,
    selcted_columns=["audio", "id", "original_id", "gender", "age"],
    forced_ids=["s4_1", "s25_1", "s101_1", "s107_1", "s145_2"],
) -> Dataset:
    ds_original_ids = ds["original_id"]
    ds_filtered_ids = fileter_ids(ds_original_ids, seed=seed, forced_ids=forced_ids)
    ds = ds.filter(
        lambda ex: ex["original_id"] in ds_filtered_ids,
    )
    ds = ds.select_columns(selcted_columns)
    ds = ds.map(lambda ex: {"gender": "male" if ex["gender"] == 1 else "female"})

    return ds


def load_secrets():
    # Load environment variables from .env
    load_dotenv()

    # Retrieve tokens
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Log into HuggingFace Hub
    if hf_token:
        hf_login(token=hf_token)
    else:
        print("HuggingFace token not found!")


if __name__ == "__main__":
    load_dotenv()
    seed = 42
    qdata_path = "/home/abdullah/Downloads/qdat/"
    ds = load_dataset("audiofolder", data_dir=qdata_path)
    print("Before Filteration")
    print(ds)

    ds = chose_single_source(ds["train"], seed=seed)
    print("After Filteration")
    print(ds)

    print(ds[4]["original_id"])
    print(ds[25]["original_id"])
    print(ds[101]["original_id"])
    print(ds[107]["original_id"])
    print(ds[145]["original_id"])

    ds.push_to_hub("obadx/qdat_bench")
