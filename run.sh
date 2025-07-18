#!/bin/bash
python src/main.py $1 \
  --output $2 \
  --checkpoint shufflenetv2k16-apollo-24 \
  --instance-threshold 0.05 --seed-threshold 0.05 \
  --line-width 4 --font-size 0