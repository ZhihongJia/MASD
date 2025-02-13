# Magnetoencephalography (MEG) Based Non-Invasive Chinese Speech Decoding

## Overview
This is the PyTorch implementation of the following paper: "Magnetoencephalography (MEG) Based Non-Invasive Chinese Speech Decoding".

## Usage

### Setup

Please see the `requirements.txt` for environment configuration.

```bash
pip install -r requirements.txt
```

### Processing

Please use the `data_processing.py` to process the MEG data.

```bash
cd ./src
python -u data_processing.py
```

### Feature extraction

Please use the `word_features.py` and `wav_features.py` to extract the text and speech features respectively.

```bash
cd ./src
python -u word_features.py
python -u wav_features.py
```

### Train and Test

Please use the following commands to train a model.

```bash
cd ./run_scripts
# Single-modal:
bash train_single.sh
bash train_cross_single.sh
## Our approach MASD: 
bash train.sh
bash train_phoneme.sh
bash train_cross.sh
```
