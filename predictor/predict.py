import argparse
import os
from args import loadArgs
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import pandas as pd


def predict(model_id: str, data: pd.DataFrame):

    # load args
    args = loadArgs(model_id)

    data_path = "AAPL_pred_temp.csv"
    args.root_path = './dataset/'
    args.data_path = data_path
    args.result_path = './results/'

    data.to_csv(os.path.join(args.root_path, data_path), index=False)

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # # check GPU status and exit if not available
    # if not torch.cuda.is_available():
    #     print('GPU is not available')
    #     exit()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Main

    exp = Exp(args)  # set experiments
    print(
        '>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(model_id))
    preds = exp.predict(model_id, True)

    torch.cuda.empty_cache()

    return preds
