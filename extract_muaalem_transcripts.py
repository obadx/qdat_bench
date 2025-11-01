import json
import argparse
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoFeatureExtractor, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
import os
from tqdm import tqdm


def ctc_decode(batch_arr, blank_id=0, collapse_consecutive=True) -> list:
    decoded_list = []
    for seq in batch_arr:
        if collapse_consecutive:
            tokens = []
            prev = blank_id
            for current in seq:
                if current == blank_id:
                    prev = blank_id
                    continue
                if current == prev:
                    continue
                tokens.append(current)
                prev = current
            decoded_list.append(tokens)
        else:
            tokens = seq[seq != blank_id]
            decoded_list.append(tokens)
    return decoded_list


def decode_phonetic_transcripts(
    level_to_labels: dict[str, list[list[int]]], vocab: dict
) -> dict[str, list[list[str]]]:
    """Multi level decoding (extraing string representation for every level of the `quran-transcript`

    Args:
        level_to_labels (dict[list[list[int]]): A dict of level list of shape batch, seq_len
        vocab (dict[str, dict]): dict of each label vocab

    Returns:
        level_to_batach_to_script:
            dict[str, list[list[str]]:
    """
    level_to_batch_to_script = {}
    for level in vocab:
        ids_to_ph = {idx: label for label, idx in vocab[level].items()}
        batch: list[list[str]] = []
        for seq in level_to_labels[level]:
            seq_list: list[str] = []
            for label in seq:
                seq_list.append(ids_to_ph[label])
            batch.append(seq_list)
        level_to_batch_to_script[level] = batch
    return level_to_batch_to_script


def decoed_phonemes_batch(batch: list[list[int]], vocab: dict) -> list[str]:
    ids_to_ph = list(vocab["phonemes"].keys())
    batch_str = []
    for seq in batch:
        seq_str = ""
        for label in seq:
            seq_str += ids_to_ph[label]
        batch_str.append(seq_str)
    return batch_str


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoFeatureExtractor

    def __call__(self, features):
        waves = [f["audio"]["array"] for f in features]
        ids = [f["id"] for f in features]
        original_ids = [f["original_id"] for f in features]

        batch = self.processor(
            waves,
            sampling_rate=16000,
            padding="longest",
            return_tensors="pt",
        )
        batch["id"] = ids
        batch["original_id"] = original_ids
        return batch


def main(args):
    # Load model and processor
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(args.model_id)
    processor = AutoFeatureExtractor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    vocab = tokenizer.vocab

    # Load dataset
    test_ds = load_dataset(args.dataset_name, split=args.split)

    # Filter dataset if specific IDs are provided
    if args.ids:
        test_ds = test_ds.filter(lambda x: x["id"] in args.ids)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Create DataLoader
    dataloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    dtype = torch.float32 if args.float32 else torch.bfloat16

    model.to(device, dtype=dtype)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            ids = batch.pop("id")
            original_ids = batch.pop("original_id")
            batch = {k: v.to(device, dtype=dtype) for k, v in batch.items()}

            outputs = model(**batch)
            level_to_logits = outputs[0]
            level_to_labels = {}
            # Greedy decoding + ctc decoding
            for level in level_to_logits:
                labels = level_to_logits[level].argmax(dim=-1).cpu().tolist()
                level_to_labels[level] = ctc_decode(labels)

            level_to_batch_to_script = decode_phonetic_transcripts(
                level_to_labels, vocab=vocab
            )

            # Convert logits to CPU and process
            for idx in range(len(ids)):
                results.append(
                    {
                        "id": ids[idx],
                        "original_id": original_ids[idx],
                        "levels_labels": {
                            level: level_to_labels[level][idx]
                            for level in level_to_labels
                        },
                        "level_to_scripts": {
                            level: level_to_batch_to_script[level][idx]
                            if level != "phonemes"
                            else "".join(level_to_batch_to_script[level][idx])
                            for level in level_to_batch_to_script
                        },
                    }
                )

    # Extract model name for filename
    args.output_dir.mkdir(exist_ok=True, parents=True)
    model_name = args.model_id.split("/")[-1] if "/" in args.model_id else args.model_id
    output_filename = args.output_dir / f"{model_name}_predictions.jsonl"

    # Save to JSONL file
    with open(output_filename, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on multiple model versions"
    )

    # Required arguments
    parser.add_argument(
        "--model-id",
        default="obadx/muaalem-model-v3_2",
        help="Model identifier from Hugging Face Hub or local path",
    )

    # Optional arguments
    parser.add_argument(
        "--dataset-name",
        default="obadx/qdat_bench",
        help="Dataset name to load from Hugging Face Hub",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (e.g., train, test, validation)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output Directory. If not provided, will Current directory",
        type=Path,
        default=Path(),
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use Float32 precision. Default is bfloat16",
    )
    parser.add_argument(
        "--ids", nargs="+", help="Specific IDs to process (space separated)"
    )

    args = parser.parse_args()
    main(args)
