# Example for multiple stereo image pairs (stream processing)
# Assumes data_dir contains:
# - left/ subdirectory with 0000000000.png, 0000000001.png, ...
# - right/ subdirectory with 0000000000.png, 0000000001.png, ...
# - calib.txt file (or specify with --calib)
python src/main_stream.py sample_data/sample-stream \
  --output output_stream \
  --checkpoint shufflenetv2k16-apollo-24 \
  --instance-threshold 0.05 --seed-threshold 0.05 \
  --line-width 4 --font-size 0 --use_z \
  --bev_scale 5000