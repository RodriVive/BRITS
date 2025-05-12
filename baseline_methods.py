# coding: utf-8
import json
import fancyimpute
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
warnings.filterwarnings("ignore", category=FutureWarning)

X = []
Y = []
x_masks = []
y_masks = []

# Load data with tqdm
for ctx in tqdm(open('json/improve_brits_data.json'), desc="Loading data", total=183):
    ctx = json.loads(ctx)['forward']
    x = np.asarray(ctx['values'])
    y = np.asarray(ctx['evals'])

    x_mask = np.asarray(ctx['masks']).astype(bool)
    y_mask = np.asarray(ctx['eval_masks']).astype(bool)

    x[~x_mask] = np.nan
    y[(~x_mask) & (~y_mask)] = np.nan

    x_masks.append(x_mask)
    y_masks.append(y_mask)
    X.append(x)
    Y.append(y)

x_masks = np.array(x_masks)
y_masks = np.array(y_masks)

def get_loss(X, X_pred, Y, x_masks, y_masks):
    mask = np.logical_and(x_masks, ~y_masks)  # Focus on imputed values only
    # mask = np.concatenate(mask, axis=0)
    print(f"Mask sum: {np.sum(mask)}")
    
    X_pred = np.nan_to_num(X_pred)
    # if len(X_pred.shape) == 3:
    #     X_pred = np.concatenate(X_pred, axis=0)
    # if len(Y.shape) == 3:
    #     Y = np.concatenate(Y, axis=0)
    pred = X_pred[mask]
    label = Y[mask]

    mae = np.abs(pred - label).mean()
    mre = np.abs(pred - label).sum() / (1e-5 + np.sum(np.abs(label)))

    return {'mae': mae, 'mre': mre}

# Algo1: Mean imputation
X_mean = []
print(len(X))

# Mean imputation without tqdm
for x, y in tqdm(zip(X, Y), desc="Mean imputation", total=len(X)):
    X_mean.append(fancyimpute.SimpleFill().fit_transform(y))

print(f"Sample shape: {X[0].shape}")
X_c = np.concatenate(X, axis=0).reshape(-1, 1818, 30)
Y_c = np.concatenate(Y, axis=0).reshape(-1, 1818, 30)
X_mean = np.concatenate(X_mean, axis=0).reshape(-1, 1818, 30)
print("post imputation nan count:", np.isnan(X_mean).sum())

print('Mean imputation:')
print(get_loss(X_c, X_mean, Y_c, x_masks, y_masks))

# Forward fill imputation
X_ffill = []
for x in tqdm(X, desc="Forward filling", total=len(X)):
    # Forward fill along time dimension (axis=0)
    x_ffill = pd.DataFrame(x).fillna(method='ffill').fillna(0).values
    X_ffill.append(x_ffill)
X_ffill = np.concatenate(X_ffill, axis=0).reshape(-1, 1818, 30)
print("Forward fill imputation:")
print(get_loss(X_c, X_ffill, Y_c, x_masks, y_masks))

np.save('./result/mean_data.npy', X_mean)

# Algo2: KNN imputation
X_knn = []

# KNN imputation with tqdm
for x, y in tqdm(zip(X, Y), desc="KNN imputation", total=len(X)):
    X_knn.append(fancyimpute.KNN(k=10, verbose=False).fit_transform(y))

# X_c = np.concatenate(X, axis=0)
# Y_c = np.concatenate(Y, axis=0)
X_knn = np.array(X_knn)

print('KNN imputation')
print(X_c.shape, X_knn.shape, Y_c.shape)
print(get_loss(X_c, X_knn, Y_c, x_masks, y_masks))

# MICE imputation (batch style)
# X_mice = []

# n = len(X)
# batch_size = 128
# nb_batch = (n + batch_size - 1) // batch_size

# # MICE imputation without tqdm
# for i in range(nb_batch):
#     print('On batch {}'.format(i))
#     x = np.concatenate(X[i * batch_size: (i + 1) * batch_size])
#     y = np.concatenate(Y[i * batch_size: (i + 1) * batch_size])
#     x_mice = fancyimpute.MICE(n_imputations=100, n_pmm_neighbors=20, verbose=False).fit_transform(x)
#     X_mice.append(x_mice)

# X_mice = np.concatenate(X_mice, axis=0)
# X_c = np.concatenate(X, axis=0)
# Y_c = np.concatenate(Y, axis=0)

# print('MICE imputation')
# print(get_loss(X_c, X_mice, Y_c, x_masks, y_masks))
