# ORIAN: Optional Reference Inference Ancestry Network

**ORIAN** is a local ancestry inference method that can be run **with** or **without** reference panels.  
It leverages deep learning to predict ancestry across the genome, offering flexibility in scenarios where reference panels are limited or unavailable.

---

## Features
- Run inference **with** or **without** reference panels
- Pretrained model weights available via Google Drive
- Handles large genomic datasets
- Built on PyTorch and scikit-allel

---

## Installation

Clone this repository and set up:

```bash
git clone https://github.com/thomasjdiem/ORIAN.git
cd ORIAN
pip install -r requirements.txt
python3 setup.py
```

## Use 

To see options, run 
```bash
python3 inference/with_panels/main.py -h
```
if reference panels are available. Or
```bash
python3 inference/without_panels/main.py -h
```
if not. This will create an output inferred ancestries in the specified output directory.


