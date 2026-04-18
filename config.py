import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

CURATOR_ROOT = Path(__file__).resolve().parents[1] / "Curator"
if str(CURATOR_ROOT) not in sys.path:
	sys.path.insert(0, str(CURATOR_ROOT))

DATA_DESIGNER_ROOT = Path(__file__).resolve().parents[1] / "DataDesigner"
DATA_DESIGNER_SRC = DATA_DESIGNER_ROOT / "packages" / "data-designer" / "src"
DATA_DESIGNER_CONFIG_SRC = DATA_DESIGNER_ROOT / "packages" / "data-designer-config" / "src"
for p in [DATA_DESIGNER_SRC, DATA_DESIGNER_CONFIG_SRC]:
	if p.exists() and str(p) not in sys.path:
		sys.path.insert(0, str(p))


@dataclass
class PipelineConfig:
	raw_dir: Path
	curated_dir: Path
	report_path: Path
	data_designer_endpoint: str | None
	data_designer_api_key_env: str
	data_designer_prompt: str
	designer_input_jsonl: Path | None
	source_language: str
	target_language: str
	chat_turns: int
	num_samples: int
	min_words: int
	max_words: int
	max_non_alpha_ratio: float
	max_url_ratio: float
	validation_samples: int
	seed: int
	skip_curation: bool
	skip_validation: bool
	dd_dataset_name: str
	dd_model_alias: str
	dd_model_id: str
	dd_model_provider: str
	dd_temperature: float
	dd_top_p: float
	dd_max_tokens: int
	dd_max_parallel_requests: int
	dd_batch_size: int
	key_json_path: Path


def parse_args() -> PipelineConfig:
	parser = argparse.ArgumentParser(
		description="NVIDIA-first chat data pipeline: Data Designer -> NeMo Curator -> NeMo Framework validation"
	)
	parser.add_argument("--raw-dir", type=Path, default=Path("project/data/raw"))
	parser.add_argument("--curated-dir", type=Path, default=Path("project/data/curated"))
	parser.add_argument("--report-path", type=Path, default=Path("project/data/validation_report.json"))
	parser.add_argument("--designer-input-jsonl", type=Path, default=None)
	parser.add_argument("--data-designer-endpoint", type=str, default=None)
	parser.add_argument("--data-designer-api-key-env", type=str, default="NVIDIA_API_KEY")
	parser.add_argument(
		"--data-designer-prompt",
		type=str,
		default="Create high-quality chat training samples for LLM training.",
	)
	parser.add_argument("--source-language", type=str, default="English")
	parser.add_argument("--target-language", type=str, default="Korean")
	parser.add_argument("--chat-turns", type=int, default=3)
	parser.add_argument("--num-samples", type=int, default=200)
	parser.add_argument("--min-words", type=int, default=20)
	parser.add_argument("--max-words", type=int, default=400)
	parser.add_argument("--max-non-alpha-ratio", type=float, default=0.35)
	parser.add_argument("--max-url-ratio", type=float, default=0.15)
	parser.add_argument("--validation-samples", type=int, default=300)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--skip-curation", action="store_true")
	parser.add_argument("--skip-validation", action="store_true")
	parser.add_argument("--dd-dataset-name", type=str, default="nvidia_designer_text_dataset")
	parser.add_argument("--dd-model-alias", type=str, default="nemotron-nano-v3")
	parser.add_argument("--dd-model-id", type=str, default="nvidia/nemotron-3-nano-30b-a3b")
	parser.add_argument("--dd-model-provider", type=str, default="nvidia")
	parser.add_argument("--dd-temperature", type=float, default=0.7)
	parser.add_argument("--dd-top-p", type=float, default=0.95)
	parser.add_argument("--dd-max-tokens", type=int, default=1024)
	parser.add_argument("--dd-max-parallel-requests", type=int, default=8)
	parser.add_argument("--dd-batch-size", type=int, default=200)
	parser.add_argument("--key-json-path", type=Path, default=Path("project/key.json"))

	args = parser.parse_args()
	return PipelineConfig(
		raw_dir=args.raw_dir,
		curated_dir=args.curated_dir,
		report_path=args.report_path,
		data_designer_endpoint=args.data_designer_endpoint,
		data_designer_api_key_env=args.data_designer_api_key_env,
		data_designer_prompt=args.data_designer_prompt,
		designer_input_jsonl=args.designer_input_jsonl,
		source_language=args.source_language,
		target_language=args.target_language,
		chat_turns=args.chat_turns,
		num_samples=args.num_samples,
		min_words=args.min_words,
		max_words=args.max_words,
		max_non_alpha_ratio=args.max_non_alpha_ratio,
		max_url_ratio=args.max_url_ratio,
		validation_samples=args.validation_samples,
		seed=args.seed,
		skip_curation=args.skip_curation,
		skip_validation=args.skip_validation,
		dd_dataset_name=args.dd_dataset_name,
		dd_model_alias=args.dd_model_alias,
		dd_model_id=args.dd_model_id,
		dd_model_provider=args.dd_model_provider,
		dd_temperature=args.dd_temperature,
		dd_top_p=args.dd_top_p,
		dd_max_tokens=args.dd_max_tokens,
		dd_max_parallel_requests=args.dd_max_parallel_requests,
		dd_batch_size=args.dd_batch_size,
		key_json_path=args.key_json_path,
	)


def ensure_dirs(config: PipelineConfig) -> None:
	config.raw_dir.mkdir(parents=True, exist_ok=True)
	config.curated_dir.mkdir(parents=True, exist_ok=True)
	config.report_path.parent.mkdir(parents=True, exist_ok=True)


def load_env_from_key_json(key_json_path: Path) -> None:
	if not key_json_path.exists():
		return

	with key_json_path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	if isinstance(payload, dict) and isinstance(payload.get("env"), dict):
		payload = payload["env"]

	if not isinstance(payload, dict):
		raise ValueError(f"{key_json_path} must contain a JSON object of environment keys and values.")

	for key, value in payload.items():
		if not isinstance(key, str):
			continue
		if value is None:
			continue
		os.environ[key] = str(value)
