.PHONY: validate synth ner model auto

validate:
	python scripts/validate_ontology.py ontology/v1.yaml

synth:
	python scripts/gen_synth_gpt.py --n 50000 --onto ontology/v1.yaml --out data/synth

ner:
	python scripts/train_ner.py --train data/synth/train.jsonl --val data/synth/val.jsonl --out models/extractor_ner

model:
	python scripts/train_symptom_net.py --train data/synth/train.jsonl --val data/synth/val.jsonl

auto:
	python scripts/auto_label_new_data.py