# nemo_dream Korean Curation Pipeline

English-to-Korean localization data curation pipeline for hackathon demos.

This repo now has three connected pieces:

- `scripts/run_from_seed.py`: full seed-to-report runner
- `scripts/build_curated_stages.py`: Stage1~3 NIM generation from seed text
- `pipeline/`: curator-style validation/evaluation stages from `feat/mockup-pipeline`
- `dashboard.py`: visualization dashboard for `data/curated/stage1` through `stage7`

The full flow starts from seed English text, builds Stage1~3 with NIM, then feeds Stage3 into the validation/evaluation stages. The important integration point is `pipeline/stages/s0_generate.py`: it normalizes Stage3 or Stage4-style JSONL into the schema expected by the validation stages.

## Pipeline Stages

| Stage | File | Purpose |
| --- | --- | --- |
| Stage 1 | `scripts/build_curated_stages.py` | NIM sociolinguistic decomposition from seed English text |
| Stage 2 | `scripts/build_curated_stages.py` | NIM cultural reference mapping |
| Stage 3 | `scripts/build_curated_stages.py` | NIM Korean rewrite generation |
| Stage 4 | `pipeline/stages/s1_schema.py`, `s2_rules.py`, `s3_safety.py` | Schema, rule, and safety validation after S0 normalization |
| Stage 5 | `pipeline/stages/s4_semantic.py`, `s5_filter.py` | Semantic/property evaluation and aggregate filtering |
| Stage 6 | `pipeline/stages/s5_filter.py`, `s6_reward.py` | Dedup and reward scoring |
| Stage 7 | `pipeline/stages/s7_report.py` | Report and chart output |

## Data Flow

Curated data lives under `data/curated/`:

```text
data/curated/prosocial_input.jsonl     seed English text
data/curated/stage1.jsonl              Stage1 decomposition output
data/curated/stage2.jsonl              Stage2 cultural mapping output
data/curated/stage3.jsonl              Stage3 Korean rewrite generation output
data/curated/stage4.jsonl              semantic judge style records
data/curated/stage5_postprocessed.jsonl
data/curated/stage6_evaluated.jsonl
data/curated/stage7_report.jsonl
```

`stage1` to `stage3` use the `nemo_dream_step1` shape, for example `decomposed`, `mapped_refs`, and `generation.rewritten_text`.

`stage4` is closer to the validation pipeline shape, with `en_text`, `ko_text`, `metadata`, `quality`, and `valid`.

S0 supports both forms and normalizes them into this shape before S1:

```json
{
  "id": "sample-id",
  "en_text": "original English text",
  "ko_text": "localized Korean text",
  "metadata": {
    "speech_act": "complaint",
    "register": "casual",
    "emotion_type": "anger",
    "emotion_intensity": 3,
    "estimated_age_group": "20s",
    "platform_fit": ["twitter"],
    "target_platform": "twitter",
    "cultural_refs": [],
    "internet_markers": {
      "laughter": "none",
      "emphasis": [],
      "sarcasm_marker": false
    }
  },
  "provenance": {
    "source": "curated_jsonl"
  }
}
```

## Quick Start

Start from seed data and run the full pipeline:

```bash
export NVIDIA_API_KEY=nvapi-...

python scripts/import_prosocial_dialog.py \
  --split train \
  --limit 10 \
  --safety-label __casual__ \
  --output data/curated/prosocial_input.jsonl

python scripts/run_from_seed.py \
  --input data/curated/prosocial_input.jsonl \
  --curated-dir data/curated \
  --limit 10 \
  --normalized-out raw/prosocial_normalized.jsonl \
  --validation-out-dir val_out_prosocial \
  --no-chart
```

The full runner executes:

```text
seed JSONL
-> Stage1 NIM decomposition
-> Stage2 NIM cultural mapping
-> Stage3 NIM Korean rewrite
-> S0 normalize Stage3 output
-> Stage4 schema/rules/safety
-> Stage5 semantic/filter
-> Stage6 dedup/reward
-> Stage7 report
```

Outputs are written to:

```text
data/curated/stage1.jsonl
data/curated/stage2.jsonl
data/curated/stage3.jsonl
raw/prosocial_normalized.jsonl
val_out_prosocial/accepted.jsonl
val_out_prosocial/rejected.jsonl
val_out_prosocial/report.json
```

You can also start from an existing generated stage file. Run Stage4+ validation/evaluation from `stage3` generation output:

```bash
python -m pipeline.run \
  generate.source_jsonl=data/curated/stage3.jsonl \
  generate.out_jsonl=/tmp/stage3_s0_normalized.jsonl \
  report.chart=false
```

Or start from existing `stage4` data:

```bash
python -m pipeline.run \
  generate.source_jsonl=data/curated/stage4.jsonl \
  generate.out_jsonl=/tmp/stage4_s0_normalized.jsonl \
  report.chart=false
```

## Config

Main config:

```text
conf/config.yaml
```

Useful overrides:

```bash
# Choose an input JSONL from data/curated for Stage4+ validation
python -m pipeline.run generate.source_jsonl=data/curated/stage4.jsonl

# Write normalized S0 output somewhere explicit
python -m pipeline.run generate.source_jsonl=data/curated/stage4.jsonl generate.out_jsonl=/tmp/normalized.jsonl

# Disable chart generation for faster local runs
python -m pipeline.run generate.source_jsonl=data/curated/stage4.jsonl report.chart=false

# Change output directory
python -m pipeline.run generate.source_jsonl=data/curated/stage4.jsonl paths.out_dir=val_out_stage4
```

Default judge config is mock mode:

```text
conf/judge/mock.yaml
```

NIM judge config exists here:

```text
conf/judge/nim.yaml
```

Use it with:

```bash
export NVIDIA_API_KEY=nvapi-...
python -m pipeline.run judge=nim generate.source_jsonl=data/curated/stage4.jsonl
```

## Data Designer Mode

S0 can still generate new data through NVIDIA Data Designer when `generate.enabled=true` and no `generate.source_jsonl` is provided.

```bash
export NVIDIA_API_KEY=nvapi-...
python -m pipeline.run generate.enabled=true generate.num_samples=20
```

Generated raw candidates go to:

```text
raw/generated.jsonl
```

## Prosocial Dialog Seed Data

You can seed the pipeline from AllenAI's Prosocial Dialog dataset:

```bash
python scripts/import_prosocial_dialog.py \
  --split train \
  --limit 100 \
  --text-field context \
  --output data/curated/prosocial_input.jsonl
```

Useful filters:

```bash
# Safer casual examples only
python scripts/import_prosocial_dialog.py \
  --split train \
  --limit 100 \
  --safety-label __casual__ \
  --output data/curated/prosocial_input.jsonl

# Use both user context and assistant response as the seed text
python scripts/import_prosocial_dialog.py \
  --split train \
  --limit 100 \
  --text-field context_response \
  --output data/curated/prosocial_input.jsonl
```

The importer writes records like `{"id": "...", "text": "..."}` plus metadata. Use this as the raw seed input for the Stage1 decomposition/generation workflow, then rebuild the dashboard from the resulting curated stage files.

## Build Stage1~3 Curated Files

Stage1~3 can be generated locally from seed text, following the flow in `jwee01/nemo_dream_step1`:

```text
seed text
-> Stage1 sociolinguistic decomposition
-> Stage2 cultural reference mapping
-> Stage3 Korean rewrite generation
```

This path always uses NIM for Stage1 decomposition, Stage2 cultural mapping, and Stage3 Korean rewriting:

```bash
export NVIDIA_API_KEY=nvapi-...
python scripts/build_curated_stages.py \
  --input data/curated/prosocial_input.jsonl \
  --out-dir data/curated \
  --limit 10
```

The command writes:

```text
data/curated/stage1.jsonl
data/curated/stage2.jsonl
data/curated/stage3.jsonl
```

Then run the existing Stage4+ validation/evaluation path from the generated Stage3 file:

```bash
python scripts/run_from_seed.py \
  --input data/curated/prosocial_input.jsonl \
  --curated-dir data/curated \
  --limit 10 \
  --normalized-out raw/prosocial_normalized.jsonl \
  --validation-out-dir val_out_prosocial \
  --no-chart
```

## Visualization Dashboard

Build a static dashboard from the curated stage files:

```bash
python dashboard.py \
  --curated-dir data/curated \
  --output artifacts/dashboard.html
```

Serve a live dashboard that refreshes when `data/curated` files change:

```bash
python dashboard.py --serve --curated-dir data/curated --host 0.0.0.0 --port 8770
```

Then open:

```text
http://127.0.0.1:8770
```


## Mockup Fixtures

The original mockup pipeline fixtures are in:

```text
mockup-data/candidates.jsonl
mockup-data/expected-outcomes.json
```

Run the default mockup pipeline:

```bash
python -m pipeline.run
```

Verify expected outcomes after running it:

```bash
python -m tests.verify_mockup
```

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Some NeMo Curator reference classes need heavier optional packages and compatible GPU/PyTorch/Transformers versions. The local runner has fallbacks for the basic demo path, so these commands should still work without the full NeMo stack:

```bash
python -m pipeline.run generate.source_jsonl=data/curated/stage4.jsonl report.chart=false
python dashboard.py --serve --curated-dir data/curated --host 0.0.0.0 --port 8770
```

## Repository Layout

```text
conf/                         Hydra configs
pipeline/                     S0-S7 validation/evaluation pipeline
pipeline/judges/              mock and NIM judge implementations
pipeline/stages/              individual stage implementations
mockup-data/                  fixture data from feat/mockup-pipeline
data/curated/                 external curated stage data + dashboard inputs
artifacts/                    generated dashboard and visualization artifacts
dashboard.py                  curated stage dashboard builder/server
```
