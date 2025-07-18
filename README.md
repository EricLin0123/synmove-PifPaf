# Openpifpaf for Stereo BEV

## installation
To install the environment, you can use the following steps.

### 0. UV
I use `uv` for faster installation. However it's optional, you can use `pip` directly if you prefer. Just ignore all `uv` prefixes in the commands below. To install `uv`, you can run:
```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
```
or (not recommended):
```bash
  pip install uv
```

### 1. Create and Activate a Virtual Environment
```bash
    uv venv --python 3.10
    source .venv/bin/activate
```

### 2. Install packages
For special need for certain torch versions, usually for RTX50 series GPUs, run the following command first:
```bash
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Others can skip this step.
Then install the required packages:
```bash
    uv pip install -r requirements.txt
    uv pip install -e .
```

## Usage
### Input Data
The structure of the input data should be as follows:
```
    <input_dir>/
    ├── calib.txt
    ├── left.png
    ├── right.png
```
- `calib.txt`: Calibration file containing camera parameters.
- `left.png`: Left image.
- `right.png`: Right image.

The file names are currently hardcoded, so please ensure they match.
Also, the data folder contains some example data, which can be used for testing. They should be removed from the repository in the future, but for now, they are useful for testing the code.

### Running the code
To run the code, you can use the following command:
```bash
  python src/main.py <input_dir> \
  --output <output_dir> \
  --checkpoint shufflenetv2k16-apollo-24 \
  --instance-threshold 0.05 --seed-threshold 0.05 \
  --line-width 4 --font-size 0
```
Or simply run:
```bash
  bash run.sh <input_dir> <output_dir>
```
Also, you can run the hardcoded version:
```bash
  bash example.sh
```

## Code Structure
The code is structured as follows:
```
    src/
    ├── main.py          # Main script to run the BEV prediction
    ├── utils/
    │   ├── matching.py  # Contains functions for matching vehicle centroids between left and right images
    │   └── visualize.py # Contains functions for visualizing the results
    └── openpifpaf/ # The original openpifpaf library
```