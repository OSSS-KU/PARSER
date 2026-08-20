# PARSER: Residual Sparsification via Output Importance for Compressing MoE LLMs

Reference implementation of **PARSER**, a residual-sparsification
method that compresses Mixture-of-Experts (MoE) experts by removing the hidden
dimensions that contribute least to the expert **output** (output importance),
pooled globally across all experts in each MoE layer.

The pipeline has three stages:

- `save_blocks.py` — extract MoE blocks from a Hugging Face model and collect calibration (activation) stats.
- `compress_blocks.py` — compress saved blocks with `parser-l-sp` (or `parser-e-sp`).
- `evaluate_blocks.py` — load original/compressed blocks back into the model and run lm-eval-harness.

`parser-l-sp` = layer-scope structured pruning (PARSER's **global pooling**).
`parser-e-sp` = expert-scope structured pruning (per-expert **local selection**,
used only by the Table 4 ablation).

## Requirements

Install the dependencies listed in `requirements.txt`. All original and
compressed models are loaded and evaluated in bfloat16.

## Driver scripts

Four driver scripts run the full **save → compress → evaluate** pipeline and
sweep ρ ∈ {0.9, 0.8, 0.7}. They share the same PARSER configuration (method
`parser-l-sp`, `--backbone barycenter`, `--importance formula`, `--min-units 4`,
routing score off, Dolly-15K calibration with 512 sequences × 2048 packed tokens,
seed 0, bfloat16, single GPU):

| Script | Model | Compressed layers |
|---|---|---|
| `run_residual_sparsification_qwen.sh`      | Qwen1.5-MoE-A2.7B  | `last:16` |
| `run_residual_sparsification_deepseek.sh`  | DeepSeek-V2-Lite   | `last:18` |
| `run_residual_sparsification_olmoe.sh`     | OLMoE-1B-7B-0125   | `last:11` |
| `run_residual_sparsification_moonlight.sh` | Moonlight-16B-A3B  | `last:18` |

Each script runs `METHODS="original,parser-l-sp"` and evaluates the 7 tasks
`arc_easy, arc_challenge, winogrande, hellaswag, openbookqa, piqa, mmlu`.
Edit the config block at the top of a script and run it, e.g.
`./run_residual_sparsification_qwen.sh`.

Results land in:
- `artifacts/<model_slug>/<tag>/eval/eval_results.json` — per-task accuracy (or perplexity) and peak GPU memory (`vram_*`).
- `artifacts/<model_slug>/<tag>/compressed/compression_manifest.json` — MoE parameter reduction.
- `artifacts/<model_slug>/<tag>/compressed/phase_timing.json` — compression time.

The per-ratio result dirs are tagged `s090` / `s080` / `s070` for ρ = 90 / 80 / 70%.

## How to reproduce (in paper order)

The items below follow the order in which figures and tables appear in the paper.

### Table 1 — Main results (Qwen & DeepSeek @ ρ=90%)

Accuracy and peak GPU memory.

```bash
./run_residual_sparsification_qwen.sh
./run_residual_sparsification_deepseek.sh
```

Read the `s090` result dir: per-task accuracy from `eval_results.json`, peak GPU
memory from its `vram_*` fields. The "No compression" row is the `original`
method's eval.

### Figure 3 — Accuracy vs compression ratio ρ (Qwen & DeepSeek)

Widen the ratio sweep in the Qwen/DeepSeek scripts, then plot average accuracy
against ρ from each per-ratio `eval_results.json`:

```bash
SPARSITIES="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
```

### Figure 4 — Accuracy vs peak GPU memory (Qwen & DeepSeek)

Same runs as Figure 3; plot average accuracy against the `vram_*` peak memory
instead of ρ.

### Table 2 — Calibration set 𝒟 sensitivity (Qwen & DeepSeek @ ρ=90%)

Summary (average accuracy + std dev) over three knobs. Vary one knob at a time
in the Qwen/DeepSeek scripts:

- seed: `CALIBRATION_SEED` ∈ {0, 1, 2}
- source: `CALIBRATION_DATASET` ∈ {dolly, c4, wikitext}
- size: `CALIBRATION_SAMPLES` ∈ {128, 256, 512}

Per-task breakdowns are Tables 12–14.

### Table 3 — Compression criterion ablation (Qwen & DeepSeek, ρ 90/80/70)

Set `PARSER_IMPORTANCE`:

- Wanda → `wanda`
- output importance (PARSER) → `formula`

### Table 4 — Selection process ablation (Qwen & DeepSeek @ ρ=90%)

Edit the Qwen/DeepSeek scripts:

- local selection → `METHODS="parser-e-sp"`
- global pooling (PARSER) → `METHODS="parser-l-sp"`, `PARSER_ROUTING_SCORE=0`
- routing-aware global pooling → `METHODS="parser-l-sp"`, `PARSER_ROUTING_SCORE=1`

### Table 5 — Overhead (Qwen & DeepSeek @ ρ=90%)

- Compression time — each compress step writes `phase_timing.json`; aggregate per model:
  ```bash
  python aggregate_timing.py artifacts/Qwen_Qwen1.5-MoE-A2.7B
  ```
- Serving throughput (decode tokens/sec) on a compressed (or original) blocks dir:
  ```bash
  python benchmark_tps.py \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --blocks-dir artifacts/Qwen_Qwen1.5-MoE-A2.7B/blocks \
    --compressed-dir artifacts/Qwen_Qwen1.5-MoE-A2.7B/parser-l-sp-s090-barycenter-rnone/compressed \
    --output-dir artifacts/Qwen_Qwen1.5-MoE-A2.7B/tps
  ```

### Table 8 — OLMoE & Moonlight @ ρ=90% (accuracy + peak GPU memory, Appendix F.1)

```bash
./run_residual_sparsification_olmoe.sh
./run_residual_sparsification_moonlight.sh
```

Read the `s090` dir as in Table 1.

### Table 9 — All 4 models @ ρ=80% (Appendix F.2)

The four driver scripts already sweep ρ=80%; read the `s080` result dirs.

### Table 10 — All 4 models @ ρ=70% (Appendix F.2)

Same runs; read the `s070` result dirs.

### Table 11 — WikiText perplexity (all 4 models, ρ 70/80/90, Appendix F.3)

Set `EVAL_TASKS="wikitext"` in each script and run. Perplexity is in
`eval_results.json`.

### Table 12 — Seed robustness, per-task (Appendix F.4)

Same runs as the Table 2 seed knob: `CALIBRATION_SEED` ∈ {0, 1, 2}; read the
per-task numbers in each `eval_results.json`.

### Table 13 — Calibration dataset robustness, per-task (Appendix F.4)

`CALIBRATION_DATASET` ∈ {dolly, c4, wikitext}; per-task results.

### Table 14 — Calibration sample-size robustness, per-task (Appendix F.4)

`CALIBRATION_SAMPLES` ∈ {128, 256, 512}; per-task results.

## Quick start (manual, single config)

1) Save blocks + calibration stats:

```bash
python save_blocks.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --device cuda \
  --calibrate \
  --calibration-dataset dolly \
  --calibration-samples 512 \
  --calibration-seq-len 2048 \
  --calibration-packed
```

2) Compress (PARSER, ρ=0.7), writing to an explicit dir:

```bash
python compress_blocks.py \
  --blocks-dir artifacts/Qwen_Qwen1.5-MoE-A2.7B/blocks \
  --method parser-l-sp \
  --sparsity 0.7 \
  --backbone barycenter \
  --layers last:16 \
  --output-dir artifacts/Qwen_Qwen1.5-MoE-A2.7B/demo/compressed
```

3) Evaluate:

```bash
python evaluate_blocks.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --blocks-dir artifacts/Qwen_Qwen1.5-MoE-A2.7B/blocks \
  --compressed-dir artifacts/Qwen_Qwen1.5-MoE-A2.7B/demo/compressed \
  --layers last:16 \
  --tasks wikitext arc_easy
```

## CLI details

### `save_blocks.py`

- `--model`: Hugging Face model ID or local path
- `--output-dir`: output root (default: auto under `artifacts/`)
- `--device`: `cpu` or `cuda`; `--device-map`: HF device map (e.g. `auto`)
- `--dtype`: `auto`, `fp32`, `fp16`, `bf16`
- `--calibrate`: collect activation stats for PARSER
- `--calibration-dataset`: dataset name(s), comma-separated (config/text-field inferred; `dolly`, `c4`, `wikitext`, ...)
- `--calibration-samples`: number of sequences (int) or `auto`
- `--calibration-seq-len`: token length (int), `auto`, or `auto:<cap>`
- `--calibration-packed`: concat-then-chunk to exact seq-len tokens (GPTQ/Wanda recipe)
- `--calibration-seed`, `--calibration-batch-size`, `--calibration-padding`, `--skip-existing`

Outputs: `blocks/original/layer_*.pt`, `manifest.json`, `activation_stats.pt` (when `--calibrate`).

### `compress_blocks.py`

- `--blocks-dir`: directory containing `manifest.json` and `original/`
- `--output-dir`: destination for compressed blocks (default: auto tag under the model dir)
- `--method`: `parser-l-sp` (layer scope / global pooling, default), `parser-e-sp` (expert scope / local selection), or `original` (no-op passthrough)
- `--sparsity`: float in [0, 1] (compression ratio ρ)
- `--backbone`: `mean` or `barycenter`
- `--importance`: `formula` (output importance, default) or `wanda`
- `--parser-routing-score`: `on`/`off` — weight importance by pre-top-k router softmax² stats
- `--min-units`, `--layers` (`all` | `last:k` | `start:end` | comma list)

Notes: `parser-*` requires `activation_stats.pt` (from `save_blocks.py --calibrate`).

Outputs: `layer_*.pt`, `compression_manifest.json` (MoE parameter metrics), `phase_timing.json`.

### `evaluate_blocks.py`

- `--model`, `--blocks-dir`, `--compressed-dir` (omit to evaluate the original blocks)
- `--device-map`: HF device map (e.g. `auto`) to shard across GPUs
- `--layers`, `--tasks` (lm-eval tasks, e.g. `wikitext arc_easy`)
- `--batch-size`, `--num-fewshot`, `--limit`

Outputs: `eval_results.json` (per-task metrics + `eval_time_sec`, VRAM totals/per-device, `vram_model_size_mb`).

## Output layout

```
artifacts/
  <model_slug>/
    blocks/
      original/
      manifest.json
      activation_stats.pt
    parser-l-sp-s090-barycenter-rnone/      # <method-tag> (ρ, backbone, routing)
      compressed/
        layer_*.pt
        compression_manifest.json
        phase_timing.json
      eval/
        eval_results.json
```

## Tips

- Use `--device cuda` and `--dtype bf16` for large models.
- To evaluate the uncompressed baseline, omit `--compressed-dir` (or use `--method original`).
