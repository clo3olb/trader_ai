# Deep Learning-based Algorithmic Trading: Design and Development of a Trading Bot for Stock Market Portfolio Optimization

Hyeonwoo KIM FYP Project
55326851

## Pre-requisites
- Python 3.9
- GPU (NVIDIA)
- CUDA 11.1

##  Basic setup

```python3
python3 -m venv venv

source venv/bin/activate
pip install -r requirements.txt

pip freeze
pip freeze > requirements.txt
```

## Removed files
to reduce the size of the repository, I removed the following files:
- `predictor/data/`
- `predictor/results/`
- `trader/tensorboard/`

If you need the original files, please clone the git repository from the following link:
```bash
git clone https://github.com/clo3olb/trader_ai.git
```


# Prediction model
You can find prediction model in the `predictor` directory.

## Trainning the predictor model
All the script that can run the prediction model is in the `scripts/` directory.

```bash
sh scripts/<MODEL_NAME>.sh <MODEL_NAME>_<DATASET_NAME> <DATASET_NAME>.csv <OUTPUT_COLUMN_SIZE>
sh scripts/PatchTST.sh PatchTST_AAPL AAPL.csv 19
```

You can find the results in the `predictor/results/` directory.


# Trading Bot
You can find trading bot in the `trader` directory.

# Run Script for trading RL model
```bash
python3 trader/trade.py
```
