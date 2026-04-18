import importlib
import json
import logging
import os
import random
from pathlib import Path
from pydantic import BaseModel
from statistics import mean
from typing import Any
from urllib import error, request

from config import PipelineConfig, ensure_dirs, load_env_from_key_json, parse_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChatMessage(BaseModel):
	role: str
	content: str


class ChatConversation(BaseModel):
	messages: list[ChatMessage]


def _normalize_role(role: Any) -> str | None:
	if not isinstance(role, str):
		return None
	role = role.strip().lower()
	role_map = {
		"human": "user",
		"user": "user",
		"assistant": "assistant",
		"gpt": "assistant",
		"bot": "assistant",
		"system": "system",
	}
	return role_map.get(role)


def _coerce_messages(value: Any) -> list[dict[str, str]]:
	if isinstance(value, str):
		try:
			value = json.loads(value)
		except json.JSONDecodeError:
			return []

	if isinstance(value, dict):
		value = value.get("messages") or value.get("conversation") or value.get("chat") or []

	if not isinstance(value, list):
		return []

	messages: list[dict[str, str]] = []
	for item in value:
		if not isinstance(item, dict):
			continue
		role = _normalize_role(item.get("role") or item.get("from") or item.get("speaker"))
		content = item.get("content") or item.get("value") or item.get("text")
		if role and isinstance(content, str) and content.strip():
			messages.append({"role": role, "content": content.strip()})
	return messages


def _messages_to_text(messages: list[dict[str, str]]) -> str:
	return "\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("content"))


def _normalize_chat_record(row: dict[str, Any]) -> dict[str, Any] | None:
	messages = _coerce_messages(row.get("messages"))
	if not messages:
		messages = _coerce_messages(row.get("conversations"))

	if not messages:
		instruction = row.get("instruction") or row.get("prompt") or row.get("question")
		input_text = row.get("input")
		output = row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
		user_parts = [part.strip() for part in [instruction, input_text] if isinstance(part, str) and part.strip()]
		if user_parts and isinstance(output, str) and output.strip():
			messages = [
				{"role": "user", "content": "\n".join(user_parts)},
				{"role": "assistant", "content": output.strip()},
			]

	if not messages:
		text = row.get("text") or row.get("content")
		if isinstance(text, str) and text.strip():
			messages = [{"role": "assistant", "content": text.strip()}]

	if not messages:
		return None

	record = {
		**row,
		"messages": messages,
		"text": _messages_to_text(messages),
	}
	return record


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rows.append(json.loads(line))
	return rows


def _extract_records(payload: Any) -> list[dict[str, Any]]:
	if isinstance(payload, list):
		rows = payload
	elif isinstance(payload, dict):
		rows = payload.get("records") or payload.get("samples") or payload.get("data") or []
	else:
		rows = []

	normalized = []
	for row in rows:
		if isinstance(row, dict):
			record = _normalize_chat_record(row)
			if record:
				normalized.append(record)
		elif isinstance(row, str):
			normalized.append(
				{
					"messages": [{"role": "assistant", "content": row}],
					"text": row,
				}
			)
	return normalized


def _import_data_designer_modules() -> tuple[Any, Any]:
	try:
		dd = importlib.import_module("data_designer.config")
		interface_mod = importlib.import_module("data_designer.interface")
		return dd, interface_mod
	except Exception as exc:  # noqa: BLE001
		raise RuntimeError(
			"Data Designer import failed. Install it with `make install` in DataDesigner and ensure NVIDIA_API_KEY is set."
		) from exc


def _build_data_designer_config(config: PipelineConfig, dd: Any) -> Any:
	model_configs = [
		dd.ModelConfig(
			alias=config.dd_model_alias,
			model=config.dd_model_id,
			provider=config.dd_model_provider,
			inference_parameters=dd.ChatCompletionInferenceParams(
				temperature=config.dd_temperature,
				top_p=config.dd_top_p,
				max_tokens=config.dd_max_tokens,
				max_parallel_requests=config.dd_max_parallel_requests,
				extra_body={"chat_template_kwargs": {"enable_thinking": False}},
			),
		)
	]

	builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
	builder.add_column(
		dd.SamplerColumnConfig(
			name="topic",
			sampler_type=dd.SamplerType.CATEGORY,
			params=dd.CategorySamplerParams(
				values=[
					"Korean AI education",
					"NVIDIA GPU optimization",
					"MLOps best practices",
					"NeMo ecosystem tutorials",
					"data engineering how-to",
				],
			),
		)
	)
	builder.add_column(
		dd.SamplerColumnConfig(
			name="style",
			sampler_type=dd.SamplerType.CATEGORY,
			params=dd.CategorySamplerParams(
				values=["tutorial", "faq", "step-by-step", "checklist", "troubleshooting memo"],
			),
		)
	)
	builder.add_column(
		dd.LLMStructuredColumnConfig(
			name="messages",
			model_alias=config.dd_model_alias,
			output_format=ChatConversation,
			prompt=(
				f"{config.data_designer_prompt}\n\n"
				f"Create one high-quality {config.target_language} chat training sample for supervised fine-tuning. "
				"Topic: {{ topic }}. Style: {{ style }}. "
				f"Write exactly {config.chat_turns} user-assistant turns. "
				"The conversation must feel natural, helpful, and diverse."
			),
		)
	)
	return builder


def _write_df_to_jsonl(
	df: Any,
	output_path: Path,
	prompt: str,
	messages_column: str = "messages",
) -> None:
	with output_path.open("w", encoding="utf-8") as f:
		for _, row in df.iterrows():
			messages = _coerce_messages(row.get(messages_column) if hasattr(row, "get") else None)
			if not messages:
				continue
			text = _messages_to_text(messages)
			record = {
				"messages": messages,
				"text": text,
				"source": "nvidia_data_designer",
				"prompt": prompt,
			}
			f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_seed_chat_dataframe(config: PipelineConfig) -> Any:
	if config.designer_input_jsonl is None:
		raise ValueError("designer_input_jsonl is required to build a seed chat dataset.")

	rows = _read_jsonl_rows(config.designer_input_jsonl)
	normalized_rows = []
	for row in rows:
		record = _normalize_chat_record(row)
		if not record:
			continue
		record["source_messages_json"] = json.dumps(record["messages"], ensure_ascii=False)
		record["source_text"] = record["text"]
		normalized_rows.append(record)

	if not normalized_rows:
		raise RuntimeError("No usable chat records were found in the input JSONL.")

	pd = importlib.import_module("pandas")
	return pd.DataFrame(normalized_rows[: config.num_samples])


def _build_translation_data_designer_config(config: PipelineConfig, dd: Any, seed_df: Any) -> Any:
	model_configs = [
		dd.ModelConfig(
			alias=config.dd_model_alias,
			model=config.dd_model_id,
			provider=config.dd_model_provider,
			inference_parameters=dd.ChatCompletionInferenceParams(
				temperature=config.dd_temperature,
				top_p=config.dd_top_p,
				max_tokens=config.dd_max_tokens,
				max_parallel_requests=config.dd_max_parallel_requests,
				extra_body={"chat_template_kwargs": {"enable_thinking": False}},
			),
		)
	]

	builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
	builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))
	builder.add_column(
		dd.LLMStructuredColumnConfig(
			name="translated_messages",
			model_alias=config.dd_model_alias,
			output_format=ChatConversation,
			prompt=(
				f"Convert the following {config.source_language} chat training example into a high-quality "
				f"{config.target_language} chat training example.\n\n"
				"Requirements:\n"
				"- Preserve the original user intent and assistant helpfulness.\n"
				"- Keep the turn structure aligned with the source conversation.\n"
				"- Rewrite naturally for native Korean usage instead of literal translation.\n"
				"- Keep system messages in the target language too.\n"
				"- Return only valid JSON.\n\n"
				"Source conversation JSON:\n{{ source_messages_json }}"
			),
		)
	)
	return builder


def _write_records_to_jsonl(records: list[dict[str, Any]], output_path: Path) -> Path:
	with output_path.open("w", encoding="utf-8") as f:
		for record in records:
			normalized = _normalize_chat_record(record)
			if not normalized:
				continue
			f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
	return output_path


def _normalize_input_chat_dataset(config: PipelineConfig, output_path: Path) -> Path:
	if config.designer_input_jsonl is None:
		raise ValueError("designer_input_jsonl is required for normalization.")
	rows = _read_jsonl_rows(config.designer_input_jsonl)
	return _write_records_to_jsonl(rows, output_path)


def _translate_with_data_designer_python_api(config: PipelineConfig, output_path: Path) -> Path:
	dd, interface_mod = _import_data_designer_modules()
	DataDesigner = interface_mod.DataDesigner

	if not os.environ.get(config.data_designer_api_key_env):
		raise ValueError(f"Missing API key. Set environment variable: {config.data_designer_api_key_env}")

	seed_df = _load_seed_chat_dataframe(config)
	data_designer = DataDesigner(artifact_path=config.raw_dir)
	config_builder = _build_translation_data_designer_config(config, dd, seed_df)
	results = data_designer.create(
		config_builder,
		num_records=min(config.num_samples, len(seed_df)),
		dataset_name=f"{config.dd_dataset_name}_translated_chat",
	)
	translated_df = results.load_dataset()
	_write_df_to_jsonl(
		translated_df,
		output_path,
		config.data_designer_prompt,
		messages_column="translated_messages",
	)

	if output_path.stat().st_size == 0:
		raise RuntimeError("Translation produced an empty dataset after filtering generation failures.")

	return output_path


def _generate_with_data_designer_python_api(config: PipelineConfig, output_path: Path) -> Path:
	dd, interface_mod = _import_data_designer_modules()
	DataDesigner = interface_mod.DataDesigner

	if not os.environ.get(config.data_designer_api_key_env):
		raise ValueError(f"Missing API key. Set environment variable: {config.data_designer_api_key_env}")

	data_designer = DataDesigner(artifact_path=config.raw_dir)
	config_builder = _build_data_designer_config(config, dd)

	total = config.num_samples
	batch_size = max(1, config.dd_batch_size)
	num_batches = (total + batch_size - 1) // batch_size

	all_frames = []
	remaining = total
	for batch_idx in range(num_batches):
		n_records = min(batch_size, remaining)
		results = data_designer.create(
			config_builder,
			num_records=n_records,
			dataset_name=f"{config.dd_dataset_name}_part_{batch_idx:03d}",
		)
		all_frames.append(results.load_dataset())
		remaining -= n_records

	pd = importlib.import_module("pandas")
	merged_df = pd.concat(all_frames, ignore_index=True)
	_write_df_to_jsonl(merged_df, output_path, config.data_designer_prompt)

	if output_path.stat().st_size == 0:
		raise RuntimeError("Data Designer created an empty dataset after filtering generation failures.")

	return output_path


def generate_with_data_designer(config: PipelineConfig) -> Path:
	output_path = config.raw_dir / "generated_from_data_designer.jsonl"

	if config.designer_input_jsonl is not None:
		if config.source_language.strip().lower() != config.target_language.strip().lower():
			if config.data_designer_endpoint:
				raise RuntimeError(
					"Seeded chat translation is only implemented for the local Data Designer Python API path in this script."
				)
			return _translate_with_data_designer_python_api(config, output_path)
		return _normalize_input_chat_dataset(config, output_path)

	if config.data_designer_endpoint:
		api_key = os.environ.get(config.data_designer_api_key_env)
		if not api_key:
			msg = f"Missing API key. Set environment variable: {config.data_designer_api_key_env}"
			raise ValueError(msg)

		body = {
			"prompt": config.data_designer_prompt,
			"num_samples": config.num_samples,
			"seed": config.seed,
		}
		req = request.Request(
			config.data_designer_endpoint,
			data=json.dumps(body).encode("utf-8"),
			headers={
				"Authorization": f"Bearer {api_key}",
				"Content-Type": "application/json",
			},
			method="POST",
		)

		try:
			with request.urlopen(req, timeout=120) as response:  # noqa: S310
				response_payload = json.loads(response.read().decode("utf-8"))
		except error.HTTPError as exc:
			details = exc.read().decode("utf-8", errors="ignore")
			raise RuntimeError(f"Data Designer request failed: {exc.code} {details}") from exc
		except error.URLError as exc:
			raise RuntimeError(f"Cannot reach Data Designer endpoint: {exc.reason}") from exc

		records = _extract_records(response_payload)
		if not records:
			raise RuntimeError("Data Designer response does not contain any usable text records.")

		with output_path.open("w", encoding="utf-8") as f:
			for row in records:
				row.setdefault("source", "nvidia_data_designer")
				row.setdefault("prompt", config.data_designer_prompt)
				f.write(json.dumps(row, ensure_ascii=False) + "\n")
		return output_path

	return _generate_with_data_designer_python_api(config, output_path)


def curate_with_nemo_curator(config: PipelineConfig, input_jsonl: Path) -> None:
	ray_client_mod = importlib.import_module("nemo_curator.core.client")
	pipeline_mod = importlib.import_module("nemo_curator.pipeline")
	filters_mod = importlib.import_module("nemo_curator.stages.text.filters")
	heuristic_mod = importlib.import_module("nemo_curator.stages.text.filters.heuristic")
	reader_mod = importlib.import_module("nemo_curator.stages.text.io.reader")
	writer_mod = importlib.import_module("nemo_curator.stages.text.io.writer")
	modifiers_mod = importlib.import_module("nemo_curator.stages.text.modifiers")

	RayClient = ray_client_mod.RayClient
	Pipeline = pipeline_mod.Pipeline
	ScoreFilter = filters_mod.ScoreFilter
	WordCountFilter = heuristic_mod.WordCountFilter
	NonAlphaNumericFilter = heuristic_mod.NonAlphaNumericFilter
	UrlsFilter = heuristic_mod.UrlsFilter
	JsonlReader = reader_mod.JsonlReader
	JsonlWriter = writer_mod.JsonlWriter
	Modify = modifiers_mod.Modify
	NewlineNormalizer = modifiers_mod.NewlineNormalizer

	ray_client = RayClient()
	ray_client.start()

	try:
		stages = [
			JsonlReader(file_paths=str(input_jsonl), files_per_partition=1),
			Modify(modifier_fn=NewlineNormalizer()),
			ScoreFilter(
				filter_obj=[
					WordCountFilter(min_words=config.min_words, max_words=config.max_words),
					NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=config.max_non_alpha_ratio),
					UrlsFilter(max_url_to_text_ratio=config.max_url_ratio),
				],
				text_field=["text", "text", "text"],
				score_field=["word_count", "non_alpha_ratio", "url_ratio"],
			),
			JsonlWriter(str(config.curated_dir)),
		]

		pipeline = Pipeline(
			name="nvidia_text_data_pipeline",
			description="Text post-processing pipeline using NeMo Curator",
			stages=stages,
		)
		pipeline.run()
	finally:
		ray_client.stop()


def _collect_texts(curated_dir: Path, max_samples: int) -> list[str]:
	texts: list[str] = []
	for jsonl_file in sorted(curated_dir.glob("**/*.jsonl")):
		with jsonl_file.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				row = json.loads(line)
				text = row.get("text")
				if isinstance(text, str) and text.strip():
					texts.append(text)
					if len(texts) >= max_samples:
						return texts
	return texts


def validate_with_nemo_framework(config: PipelineConfig) -> dict[str, Any]:
	texts = _collect_texts(config.curated_dir, config.validation_samples)
	if not texts:
		raise RuntimeError("No curated texts were found for validation.")

	random.seed(config.seed)
	sampled = random.sample(texts, k=min(len(texts), config.validation_samples))
	char_lengths = [len(t) for t in sampled]
	word_lengths = [len(t.split()) for t in sampled]
	message_counts = []
	for jsonl_file in sorted(config.curated_dir.glob("**/*.jsonl")):
		with jsonl_file.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				row = json.loads(line)
				messages = _coerce_messages(row.get("messages"))
				if messages:
					message_counts.append(len(messages))
				if len(message_counts) >= len(sampled):
					break
		if len(message_counts) >= len(sampled):
			break

	report: dict[str, Any] = {
		"dataset_size_checked": len(sampled),
		"dataset_format": "chat",
		"char_length": {
			"min": min(char_lengths),
			"max": max(char_lengths),
			"mean": round(mean(char_lengths), 2),
		},
		"word_length": {
			"min": min(word_lengths),
			"max": max(word_lengths),
			"mean": round(mean(word_lengths), 2),
		},
		"message_count": {
			"min": min(message_counts) if message_counts else None,
			"max": max(message_counts) if message_counts else None,
			"mean": round(mean(message_counts), 2) if message_counts else None,
		},
		"nemo_framework": {
			"available": False,
			"tokenizer": None,
			"token_length": None,
			"error": None,
		},
	}

	try:
		nemo = importlib.import_module("nemo")
		tokenizer_utils = importlib.import_module("nemo.collections.nlp.modules.common.tokenizer_utils")
		get_nmt_tokenizer = tokenizer_utils.get_nmt_tokenizer
		tokenizer = get_nmt_tokenizer(library="huggingface", model_name="bert-base-uncased")
		token_lengths = [len(tokenizer.text_to_ids(text)) for text in sampled]

		report["nemo_framework"] = {
			"available": True,
			"version": getattr(nemo, "__version__", "unknown"),
			"tokenizer": "nemo.collections.nlp.modules.common.tokenizer_utils.get_nmt_tokenizer",
			"token_length": {
				"min": min(token_lengths),
				"max": max(token_lengths),
				"mean": round(mean(token_lengths), 2),
			},
			"error": None,
		}
	except Exception as exc:  # noqa: BLE001
		report["nemo_framework"]["error"] = str(exc)

	return report


def write_report(path: Path, report: dict[str, Any]) -> None:
	with path.open("w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
	config = parse_args()
	ensure_dirs(config)
	load_env_from_key_json(config.key_json_path)

	generated_jsonl = generate_with_data_designer(config)
	report: dict[str, Any] | None = None

	if config.skip_curation:
		print("Skipping NeMo Curator stage.")
	else:
		curate_with_nemo_curator(config, generated_jsonl)

	if config.skip_validation:
		print("Skipping validation stage.")
	else:
		report = validate_with_nemo_framework(config)
		write_report(config.report_path, report)

	print("Pipeline complete.")
	print(f"Generated data: {generated_jsonl}")
	print(f"Curated data dir: {config.curated_dir}")
	if report is not None:
		print(f"Validation report: {config.report_path}")
	print(
		"Generation tuning: "
		f"batch_size={config.dd_batch_size}, max_parallel_requests={config.dd_max_parallel_requests}"
	)


if __name__ == "__main__":
	main()
