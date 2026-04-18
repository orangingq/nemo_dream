# Korean Data Pipeline Demo

Files:
- `example_generated_data.jsonl`: example generated dataset
- `evaluator.py`: scores each sample on semantic/attribute/cultural/fluency/style axes
- `visualize_distribution.py`: draws a histogram of final scores and average score bars

Dataset Metadata:
- Format: JSONL (`project/data/example_generated_data.jsonl`)
- Sample count: 6
- Translation direction: `en -> ko`
- Domain distribution: `social` 4, `qa` 2

Per-sample schema:
```json
{
  "sample_id": "sample_001",
  "domain": "social | qa",
  "source_language": "en",
  "target_language": "ko",
  "source_text": "original English text",
  "metadata": {
    "speech_act": "complaint | sarcasm | question",
    "register": "casual | public | intimate | formal",
    "emotion": {
      "type": "frustration | annoyance | neutral | shock | sadness",
      "intensity": "1-5"
    },
    "age_group": "20s | 30s",
    "platform": "twitter | reddit | community | linkedin | instagram_story | faq",
    "cultural_refs": ["source-side cultural expressions"],
    "internet_markers": {
      "laughter": "lol | none",
      "sarcasm_marker": "true | false"
    }
  },
  "cultural_mapping": {
    "source reference": "localized Korean reference"
  },
  "rewritten_text": "localized Korean output",
  "generation_notes": "notes about generation/localization choices"
}
```

Metadata summary:
- `speech_act`: complaint 3, question 2, sarcasm 1
- `register`: casual 2, public 2, intimate 1, formal 1
- `age_group`: 20s 4, 30s 2
- `platform`: twitter / reddit / community / linkedin / instagram_story / faq each 1
- `emotion.type`: neutral 2, frustration 1, annoyance 1, shock 1, sadness 1
- `emotion.intensity`: 1(2), 3(1), 4(2), 5(1)
- `internet_markers.sarcasm_marker`: false 5, true 1
- Average number of `cultural_refs` per sample: 1.5

Run:
```bash
python evaluator.py --input example_generated_data.jsonl --output evaluated_data.jsonl
python visualize_distribution.py --input evaluated_data.jsonl --output score_distribution.png
```
