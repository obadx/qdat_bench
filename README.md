# qdat_bench

Bench marking Some of Tajweed Rules to help creating New AI models to assist Muslim to learn reciting the Holy Quran using AI

## Usage

### Evaluation

Run standard evaluation:
```bash
uv run python eval_results.py
```

Run evaluation with bootstrap analysis (10,000 iterations):
```bash
uv run python eval_results.py --bootstrap
```

Options:
- `--transcription-file`: Path to predictions file (default: `./assets/muaalem-transcripts/muaalem-model-v3_2_predictions.jsonl`)
- `--save-dir`: Directory to save results (default: `./assets/results`)
- `--n-bootstrap`: Number of bootstrap iterations (default: 10000)
- `--seed`: Random seed for reproducibility (default: 42)

### Plotting

Generate violin plots from bootstrap samples:
```bash
uv run python plot_stats.py --bootstrap-samples assets/results/result_muaalem-model-v3_2_predictions_bootstrap_avg_samples.json --plot-type bootstrap_violin
```

Generate dataset statistics:
```bash
uv run python plot_stats.py --plot-type dataset_stats
```

Generate all plots:
```bash
uv run python plot_stats.py --plot-type all
```

Options:
- `--bootstrap-samples`: Path to bootstrap samples JSON file (required for violin plots)
- `--save-dir`: Directory to save plots (default: `assets`)
- `--plot-type`: Type of plot to generate: `bootstrap_violin`, `dataset_stats`, or `all` (default: `all`)

## Output Files

- `result_*.json`: Contains `speech_metrics`, `qdat_metrics`, `qdat_avg_metrics`, and their bootstrapped versions (`*_mean`, `*_std`)
- `result_*_bootstrap_avg_samples.json`: Bootstrap samples for violin plot generation
- `bootstrap_violin_plots.png`: Violin plots grouped by metric type (PER, RMSE, percentage)
