## Folder Structure
ohealth/
├── data/                ← training & inference data
│   ├── synth/
│   ├── real/
│   └── new_batch/
├── models/              ← checkpoints (spaCy & SymptomNet)
├── ontology/
│   ├── trees.xlsx       ← editable spreadsheet
│   └── v1.yaml          ← generated YAML
├── scripts/             ← CLI helpers (build, train, etc.)
├── symptom_net/         ← runtime package (extractor, model, engine)
├── Makefile             ← one‑command workflow
└── requirements.txt

1. run requirements.txt
   pip install -r requirements.txt

2. Define the v1.yaml file or excel file
   Convert the tree.xlsx file to yaml format
   python scripts/build_ontology.py ontology/trees.xlsx ontology/v1.yaml

3. Validate the YAML File
   python scripts/validate_ontology.py ontology/v1.yaml

4. Generate Synthetic Data
   export OPENAI_API_KEY=<key>
   python scripts/gen_synth_gpt.py --n <iteration_samples> --onto ontology/v1.yaml --out data/synth

   Outputs files
   data/synth/train.jsonl
   data/synth/val.jsonl

5. NER Training
   python scripts/train_ner.py \
      --train data/synth/train.jsonl \
      --val   data/synth/val.jsonl   \
      --out   models/extractor_ner

6. Symptom_net training
   python scripts/train_symptom_net.py \
      --train data/synth/train.jsonl \
      --val   data/synth/val.jsonl   \
      --bs 4096 --epochs 5

7. Auto‑Label Raw Narratives and Drop new text lines at
   data/new_batch/unlabeled.jsonl
   {"uid":"u1","text":"Hi Doctor, …"}

   python scripts/auto_label_new_data.py \
      --infile  data/new_batch/unlabeled.jsonl \
      --outfile data/new_batch/with_predictions.jsonl

   python scripts/demo_predict_paths.py
      
Expected Output
INPUT #1: Hi Doctor, I've had a persistent fever …
Predicted Leaf: fever_cough_long_severe
Symptom Path  : fever → fever_cough → fever_cough_long_severe
Risk Score    : 0.823
Top 5 leaf predictions:
  #1 fever_cough_long_severe (0.823)
  #2 fever_cough             (0.710)
  …
   
