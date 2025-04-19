#!/usr/bin/env python

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml, sys, json
from symptom_net.constants import CANON_KEYS

nodes=yaml.safe_load(open(sys.argv[1] if len(sys.argv)>1 else "ontology/v1.yaml"))

for n in nodes:
    for fu in n["followups"]:
        for k in fu["asks_for"]:
            assert k in CANON_KEYS, f"{n['id']} ask key {k} not in CANON_KEYS"

print("✅ YAML keys match canonical list")
