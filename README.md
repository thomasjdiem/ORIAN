# ORIAN: Optional Reference Inference Ancestry Network

**ORIAN** is a local ancestry inference method that can be run **with** or **without** reference panels.  
It leverages deep learning to predict ancestry across the genome, offering flexibility in scenarios where reference panels are limited or unavailable.

---

## Features
- Run inference **with** or **without** reference panels
- Pretrained model weights available via Google Drive
- Built on PyTorch and scikit-allel

---

## Installation

We recommend installing ORIAN in a clean environment to avoid dependency conflicts.
Both conda and venv work.

### Option 1: Using conda (recommended)

```bash
git clone https://github.com/thomasjdiem/ORIAN.git
cd ORIAN

# Create and activate a conda environment
conda create -n orian python=3.9
conda activate orian

# Install dependencies
pip install -r requirements.txt
python3 setup.py
```


### Option 2: Using venv

```bash
git clone https://github.com/thomasjdiem/ORIAN.git
cd ORIAN
python3 -m venv orian-env
source orian-env/bin/activate  # (on Windows: orian-env\Scripts\activate)
pip install -r requirements.txt
python3 setup.py
```

Note: PyTorch will install either the CPU or GPU build depending on your system.  
If you need a specific CUDA build, see: https://pytorch.org/get-started/locally/

## Use 

To see options, run 
```bash
python3 inference/with_panels/main.py -h
```
if reference panels are available or
```bash
python3 inference/without_panels/main.py -h
```
if not. Running this software will create an output directory containing inferred ancestry of each admixed sample, within the current working directory.

