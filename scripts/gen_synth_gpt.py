#!/usr/bin/env python
"""
Generate long‑form dialogues with GPT‑4o‑mini.
Set OPENAI_API_KEY env var first.
"""

import sys
import os
import openai
import yaml
import json
import random
import uuid
import argparse
import pathlib
from tqdm import tqdm


from tqdm import tqdm
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT = """
You are simulating a patient in an online medical chat.
1. Write a SINGLE paragraph (250‑400 tokens) in first‑person describing their issue.
2. Fill the JSON template *completely*. Use null for unknown fields.

TEMPLATE:
{
  "text": <paragraph>,
  "extracted": {...},            // all variables in list below
  "label_leaf_id": <pick ONE from {leaf_ids}>,
  "risk": <0‑1 urgency>
}

Variables to fill inside "extracted":
{variables}
"""

VARIABLES = [
 "age","sex","ethnicity","socioeconomic_status","location","region",
 "past_conditions","surgeries","hospitalisations","chronic_illnesses",
 "medication_history","immunisation_status","allergies",
 "family_history","diet","physical_activity","sleep_pattern",
 "alcohol","tobacco","mental_health","work_stress",
 "environmental_exposure","housing","clean_water",
 "occupation",
 "symptom_duration_map","symptom_intensity_map"
]

def main(n_samples, ontology_path, out_dir):
    leaves=[n for n in yaml.safe_load(open(ontology_path))
            if not any(c["parent_id"]==n["id"] for c in yaml.safe_load(open(ontology_path)))]
    leaf_ids=[l["id"] for l in leaves]
    question=PROMPT.format(leaf_ids=leaf_ids, variables="\n- ".join(VARIABLES))
    out_dir=pathlib.Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    train=open(out_dir/"train.jsonl","w"); val=open(out_dir/"val.jsonl","w")
    for i in tqdm(range(n_samples)):
        resp=openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":question}],
                temperature=1)
        j=json.loads(resp.choices[0].message.content)
        j["uid"]=str(uuid.uuid4())[:8]
        (val if i%10==0 else train).write(json.dumps(j)+"\n")
    print("✅ synthetic saved to",out_dir)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=50000)
    p.add_argument("--onto",default="ontology/v1.yaml")
    p.add_argument("--out",default="data/synth")
    a=p.parse_args(); main(a.n, a.onto, a.out)
