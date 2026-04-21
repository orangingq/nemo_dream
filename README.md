# nemo_dream Korean Curation Pipeline

English-to-Korean localization data curation pipeline for hackathon demos.

This repo now has two connected pieces:

- `pipeline/`: curator-style validation/evaluation stages from `feat/mockup-pipeline`
- `curated_pipeline_dashboard.py`: visualization dashboard for `data/curated/stage1` through `stage7`

The important integration point is `pipeline/stages/s0_generate.py`: it can normalize external curated JSONL files into the schema expected by the validation stages.

## Pipeline Stages

| Stage | File | Purpose |
| --- | --- | --- |
| S0 | `pipeline/stages/s0_generate.py` | Generate new data or normalize existing curated JSONL into pipeline `Sample` records |
| S1 | `pipeline/stages/s1_schema.py` | Validate schema with Pydantic |
| S2 | `pipeline/stages/s2_rules.py` | Rule checks for laughter, register, ASCII ratio, cultural refs, length, emoji |
| S3 | `pipeline/stages/s3_safety.py` | PII and content-safety checks |
| S4 | `pipeline/stages/s4_semantic.py` | Semantic/property/naturalness/cultural/register evaluation |
| S5 | `pipeline/stages/s5_filter.py` | Aggregate quality filtering and fuzzy dedup |
| S6 | `pipeline/stages/s6_reward.py` | Reward scoring |
| S7 | `pipeline/stages/s7_report.py` | Report and chart output |

## Data Flow

External curated data lives under `data/curated/`:

```text
data/curated/stage1.jsonl              decomposition output from another repo
data/curated/stage2.jsonl              cultural mapping output
data/curated/stage3.jsonl              Korean rewrite generation output
data/curated/stage4.jsonl              semantic judge style records
data/curated/stage5_postprocessed.jsonl
data/curated/stage6_evaluated.jsonl
data/curated/stage7_report.jsonl
```

`stage1` to `stage3` use the external repo's shape, for example `decomposed`, `mapped_refs`, and `generation.rewritten_text`.

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

Run the validation/evaluation pipeline from the external `stage4` data:

```bash
python -m pipeline.run \
  generate.source_jsonl=data/curated/stage4.jsonl \
  generate.out_jsonl=/tmp/stage4_s0_normalized.jsonl \
  report.chart=false
```

Run from `stage3` generation output instead:

```bash
python -m pipeline.run \
  generate.source_jsonl=data/curated/stage3.jsonl \
  generate.out_jsonl=/tmp/stage3_s0_normalized.jsonl \
  report.chart=false
```

Outputs are written to:

```text
val_out/accepted.jsonl
val_out/rejected.jsonl
val_out/report.json
```

The command runs this sequence:

```text
S0 normalize curated data
-> S1 schema
-> S2 rules
-> S3 safety
-> S4 semantic
-> S5 filter/dedup
-> S6 reward
-> S7 report
```

## Config

Main config:

```text
conf/config.yaml
```

Useful overrides:

```bash
# Choose an input JSONL from data/curated
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

## Visualization Dashboard

Build a static dashboard from the curated stage files:

```bash
python curated_pipeline_dashboard.py \
  --curated-dir data/curated \
  --output artifacts/curated_pipeline_dashboard.html
```

Serve a live dashboard that refreshes when `data/curated` files change:

```bash
python curated_pipeline_dashboard.py --serve --curated-dir data/curated --port 8767
```

Then open:

```text
http://127.0.0.1:8767
```

The older distribution dashboard is still available:

```bash
python evaluator.py --input data/example_generated_data.jsonl --output data/evaluated_data.jsonl --mode heuristic
python visualize_distribution.py --generated data/example_generated_data.jsonl --evaluated data/evaluated_data.jsonl
python dashboard_server.py --generated data/example_generated_data.jsonl --evaluated data/evaluated_data.jsonl --reference data/raw/sample_chat_en.jsonl
```

Default URL for `dashboard_server.py`:

```text
http://127.0.0.1:8765
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
python curated_pipeline_dashboard.py --serve --curated-dir data/curated --port 8767
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
curated_pipeline_dashboard.py curated stage dashboard builder/server
visualize_distribution.py     PCA distribution visualization
```
