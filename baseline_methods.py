# coding: utf-8
import json
import fancyimpute
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning)

X = []
Y = []

# Load data without tqdm
for ctx in open('json/improve_brits_data.json'):
    ctx = json.loads(ctx)['forward']
    x = np.asarray(ctx['values'])
    y = np.asarray(ctx['evals'])

    x_mask = np.asarray(ctx['masks']).astype(bool)
    y_mask = np.asarray(ctx['eval_masks']).astype(bool)

    x[~x_mask] = np.nan
    y[(~x_mask) & (~y_mask)] = np.nan

    X.append(x)
    Y.append(y)

def get_loss(X, X_pred, Y):
    mask = np.logical_and(np.isnan(X), ~np.isnan(Y))  # Focus on imputed values only
    print(f"Mask sum: {np.sum(mask)}")
    
    X_pred = np.nan_to_num(X_pred)
    pred = X_pred[mask]
    label = Y[mask]

    mae = np.abs(pred - label).mean()
    mre = np.abs(pred - label).sum() / (1e-5 + np.sum(np.abs(label)))

    return {'mae': mae, 'mre': mre}

# Algo1: Mean imputation
X_mean = []
print(len(X))

# Mean imputation without tqdm
for x, y in zip(X, Y):
    X_mean.append(fancyimpute.SimpleFill().fit_transform(x))

print(f"Sample shape: {X[0].shape}")
X_c = np.concatenate(X, axis=0).reshape(-1, 1818, 133)
Y_c = np.concatenate(Y, axis=0).reshape(-1, 1818, 133)
X_mean = np.concatenate(X_mean, axis=0).reshape(-1, 1818, 133)

print('Mean imputation:')
print(get_loss(X_c, X_mean, Y_c))

print(X_c.shape, Y_c.shape)
np.save('./result/mean_data.npy', X_mean)

# Algo2: KNN imputation
X_knn = []

# KNN imputation without tqdm
for x, y in zip(X, Y):
    X_knn.append(fancyimpute.KNN(k=10, verbose=False).fit_transform(x))

X_c = np.concatenate(X, axis=0)
Y_c = np.concatenate(Y, axis=0)
X_knn = np.concatenate(X_knn, axis=0)

print('KNN imputation')
print(get_loss(X_c, X_knn, Y_c))

# MICE imputation (batch style)
X_mice = []

n = len(X)
batch_size = 128
nb_batch = (n + batch_size - 1) // batch_size

# MICE imputation without tqdm
for i in range(nb_batch):
    print('On batch {}'.format(i))
    x = np.concatenate(X[i * batch_size: (i + 1) * batch_size])
    y = np.concatenate(Y[i * batch_size: (i + 1) * batch_size])
    x_mice = fancyimpute.MICE(n_imputations=100, n_pmm_neighbors=20, verbose=False).fit_transform(x)
    X_mice.append(x_mice)

X_mice = np.concatenate(X_mice, axis=0)
X_c = np.concatenate(X, axis=0)
Y_c = np.concatenate(Y, axis=0)

print('MICE imputation')
print(get_loss(X_c, X_mice, Y_c))
