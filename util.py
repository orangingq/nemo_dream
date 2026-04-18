
import json

def load_jsonl(path: str):
    '''Load a JSONL file and return a list of dictionaries'''
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
