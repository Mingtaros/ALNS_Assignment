#!/bin/bash

SEED_TO_USE=606

echo "S0"
python3 alns_main.py psp_instances/sample_instances/S0.json $SEED_TO_USE
echo "S1"
python3 alns_main.py psp_instances/sample_instances/S1.json $SEED_TO_USE
echo "S2"
python3 alns_main.py psp_instances/sample_instances/S2.json $SEED_TO_USE
