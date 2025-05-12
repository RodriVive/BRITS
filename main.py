import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import utils
import models
import argparse
import data_loader
import pandas as pd
import ujson as json
import os
import xarray as xr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

from sklearn import metrics
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str)
parser.add_argument('--hid_size', type=int)
parser.add_argument('--impute_weight', type=float)
args = parser.parse_args()


def evaluate(model, val_iter):
    model.eval()

    evals = []
    imputations = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    mae = np.abs(evals - imputations).mean()
    mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()

    print('MAE', mae)
    print('MRE', mre)

    return mae


def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_iter = data_loader.get_loader(batch_size=args.batch_size, split='train')
    val_iter = data_loader.get_loader(batch_size=args.batch_size, split='val', shuffle=False)

    train_maes = []
    val_maes = []

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        print(f"------------EPOCH: {epoch}------------")

        for idx, data in enumerate(train_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)
            run_loss += ret['loss'].item()

        print("\nEvaluating on training set...")
        train_mae = evaluate(model, train_iter)
        print("\nEvaluating on validation set...")
        val_mae = evaluate(model, val_iter)

        train_maes.append(train_mae)
        val_maes.append(val_mae)

        scheduler.step()

    # Plot and save training/validation MAE
    plt.plot(train_maes, label="Train MAE")
    plt.plot(val_maes, label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.title("Training and Validation MAE over Epochs")
    plt.grid(True)
    os.makedirs('./result', exist_ok=True)
    plt.savefig(f'./result/mae_plot_{args.model}.png')
    plt.close()


def save_full_imputations(model):
    print("\nRunning final model on full dataset...")

    all_iter = data_loader.get_loader(batch_size=183, split='final', shuffle=False)

    model.eval()
    imputations = None

    with torch.no_grad():
        for data in all_iter:
            data = utils.to_var(data)
            ret = model.run_on_batch(data, None)
            imputations = ret['imputations'].data.cpu().numpy()

    # imputations = np.concatenate(imputations, axis=0)
    imputations = np.vstack(imputations)
    print(imputations[0])
    print("Full imputations shape:", imputations.shape)
    data_path = "../clasp-src/data/IMPROVE_2010.nc"
    print(f"Loading data from: {os.path.abspath(data_path)}")
    try:
        ds = xr.open_dataset(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {os.path.abspath(data_path)}. Please ensure the file exists.")
        exit()
    df = ds.to_dataframe().reset_index()
    df['site'] = df['site'].str.decode('utf-8')
    sites_to_remove = ['MKGO1', 'ADPI1', 'LIVO1', 'CADI1', 'SIKE1', 'AREN1', 'DEVA1', 'CHER1', 'SOGP1']
    df = df[~df['site'].isin(sites_to_remove)]

    features = [
        "SO4", "Ca", "Ti", "Fe", "Si", "SS", "POM", "EC1", "EC2", "EC3", "EC",
        "EC1_mdl", "EC2_mdl", "EC3_mdl",
        "EC1_unc", "EC2_unc", "EC3_unc",
        "SO4_mdl", "Ca_mdl", "Ti_mdl", "Fe_mdl", "Si_mdl",
        "SO4_unc", "Ca_unc", "Ti_unc", "Fe_unc", "Si_unc",
        "lat", "lon", "elev", "timestamp", "site"
    ]
    df = df[features]

    print(df.head())
    fully_nan_cols = df.columns[df.isna().all()]
    print("Fully NaN columns:", fully_nan_cols.tolist())

    unique_sites = df['site'].unique()
    print("Number of unique sites:", len(unique_sites))

    data_numeric = df.select_dtypes(include=[np.number])
    print(imputations.shape)
    df_imputed = pd.DataFrame(imputations, columns=data_numeric.columns)
    print("df len:", len(df))
    print("df_imputed len:", len(df_imputed))

    print(df_imputed.head())

    df_result = pd.concat([df.select_dtypes(exclude=[np.number]), df_imputed], axis=1)
    print(df_result.head())

    ds_imputed = df_result.set_index(['time']).to_xarray()

    os.makedirs('./result', exist_ok=True)
    output_path = f'./result/imputed_IMPROVE_2010_{args.model}.nc'
    ds_imputed.to_netcdf(output_path)
    print(f"Imputed data saved to: {os.path.abspath(output_path)}")


def run():
    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)
    save_full_imputations(model)


if __name__ == '__main__':
    run()
