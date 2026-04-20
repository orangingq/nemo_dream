# Korean Data Pipeline Demo

Files:
- `example_generated_data.jsonl`: example generated dataset
- `evaluator.py`: scores each sample on semantic/attribute/cultural/fluency/style axes
- `visualize_distribution.py`: builds PCA distribution artifacts for data quality analysis
- `dashboard_server.py`: serves a live dashboard that refreshes when pipeline JSONL files change

Dataset Metadata:
- Format: JSONL (`project/data/example_generated_data.jsonl`)
- Sample count: 20
- Translation direction: `en -> ko`
- Domain distribution: `sns` 8, `workplace` 2, `service_faq` 2, `customer_support` 2, `developer_qna` 2, `community_qna` 1, `education` 1, `travel` 1, `commerce` 1

Per-sample schema:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You rewrite English inputs into natural Korean text that matches Korean social and cultural context."
    },
    {
      "role": "user",
      "content": "original english text"
    },
    {
      "role": "assistant",
      "content": "final polished korean text"
    }
  ],
  "metadata": {
    "source_id": "sns-001",
    "domain": "sns",
    "target_platform": "twitter",
    "target_age_group": "20s",
    "target_community": "campus",
    "target_gender_style": "neutral",
    "quality_score": {
      "meaning_preservation": {"score": 4},
      "korean_naturalness": {"score": 5},
      "cultural_grounding": {"score": 4}
    }
  }
}
```

Metadata summary:
- `target_platform`: twitter 5, community 4, instagram_story 3, faq 2, threads/naver_cafe/linkedin/support_chat/work_chat/support_ticket each 1
- `target_age_group`: 20s 13, 30s 5, 10s 1, 40s 1
- Average quality score: 4.5167 / 5
- Dashboard projection: all keys under `metadata.quality_score` are treated as a quality vector, then reduced to 2D/3D with PCA.

Run:
```bash
python evaluator.py --input data/example_generated_data.jsonl --output data/evaluated_data.jsonl --mode heuristic
python visualize_distribution.py --generated data/example_generated_data.jsonl --evaluated data/evaluated_data.jsonl
python dashboard_server.py --generated data/example_generated_data.jsonl --evaluated data/evaluated_data.jsonl --reference data/raw/sample_chat_en.jsonl
```

Visualization outputs:
- `artifacts/distribution_viz/distribution_dashboard.html`: interactive 2D/3D PCA quality dashboard
- `artifacts/distribution_viz/distribution_pca_2d.png`: static 2D PCA map
- `artifacts/distribution_viz/distribution_pca_3d.png`: static 3D PCA map
- `artifacts/distribution_viz/distribution_summary.json`: distribution and score summary

Live dashboard:
- Default URL: `http://127.0.0.1:8765`
- The browser polls `/api/payload` and redraws the dashboard when `generated`, `evaluated`, or `reference` JSONL files change.
- The graph axes are PCA components from `metadata.quality_score`, so additional quality score dimensions can be added without changing the dashboard layout.
- If the default port is busy, run with `--port 8766`.
