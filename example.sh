#!/bin/bash
python src/main.py sample_data/set1 \
  --output output/set1 \
  --checkpoint shufflenetv2k16-apollo-24 \
  --instance-threshold 0.05 --seed-threshold 0.05 \
  --line-width 4 --font-size 0 --use_z