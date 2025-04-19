#!/usr/bin/env python
"""Convert Excel/CSV → YAML and auto‑create folders."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys, pandas as pd, yaml
from pathlib import Path

def main(src, dst):
    src_p, dst_p = Path(src), Path(dst)
    if not src_p.exists():
        raise FileNotFoundError(src)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    df = (pd.read_excel if src_p.suffix in (".xlsx", ".xls") else pd.read_csv)(src_p)
    records = []
    for _, r in df.iterrows():
        node = next((n for n in records if n["id"] == r["id"]), None)
        if node is None:
            node = {
                "id": r["id"],
                "parent_id": None if pd.isna(r["parent_id"]) else r["parent_id"],
                "label": r["label"],
                "followups": []
            }
            records.append(node)
        if pd.notna(r["followup_question"]):
            node["followups"].append({
                "q": str(r["followup_question"]),
                "asks_for": [x.strip() for x in str(r["asks_for"]).split(',') if x.strip()]
            })
    dst_p.write_text(yaml.safe_dump(records, sort_keys=False))
    print(f"✅ ontology written → {dst_p.relative_to(Path.cwd())}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: build_ontology.py trees.xlsx ontology/v1.yaml")
    main(sys.argv[1], sys.argv[2])
