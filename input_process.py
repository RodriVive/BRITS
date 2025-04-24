import os
import re
import numpy as np
import pandas as pd
import ujson as json
from tqdm import tqdm
import xarray as xr
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data_path = "../clasp-src/data/IMPROVE_2010.nc"
print(f"Loading data from: {os.path.abspath(data_path)}")
try:
    ds = xr.open_dataset(data_path)
except FileNotFoundError:
    print(f"Error: File not found at {os.path.abspath(data_path)}. Please ensure the file exists.")
    exit

df = ds.to_dataframe().reset_index()
print(df.head())
fully_nan_cols = df.columns[df.isna().all()]
print("Fully NaN columns:", fully_nan_cols.tolist())

unique_sites = df['site'].unique()

print(len(unique_sites))

data_numeric = df.select_dtypes(include=[np.number])
print(data_numeric.head())

min_max_scaler = MinMaxScaler()
data_normalized_min_max = min_max_scaler.fit_transform(data_numeric)

# print(data_normalized_min_max.head())

og_data = data_normalized_min_max
print("data loaded successfully")
print("reshaping data...")

num_sites = len(unique_sites)
sites = []
site_set = set()
for i in range(num_sites):
    sites.append([])
idx = 0
for i, row in enumerate(og_data):
    # print(i, i%num_sites, len(sites), len(data))
    sites[i%num_sites].append(row)
    site_set.add((row[-4], row[-3]))

data = np.array(sites)

print(data.shape)
def make_masks_and_eval(data, missing_ratio=0.1):
    """
    Create masks and evaluation masks for BRITS training.
    """
    masks = ~np.isnan(data)             # (sites, days, features)
    evals = np.copy(data)
    eval_masks = np.zeros_like(masks, dtype=bool)

    for i in tqdm(range(data.shape[0]), desc="making masks", unit="site"):
        known_indices = np.argwhere(masks[i])
        n_eval = int(len(known_indices) * missing_ratio)
        sampled = known_indices[np.random.choice(len(known_indices), n_eval, replace=False)]
        for d, f in sampled:
            data[i, d, f] = np.nan
            eval_masks[i, d, f] = True

    return data, masks.astype(int), evals, eval_masks.astype(int)

def compute_deltas(masks):
    """
    Compute time gaps between observations for each feature.
    """
    sites, days, features = masks.shape
    deltas = np.zeros_like(masks, dtype=np.float32)

    for s in tqdm(range(data.shape[0]), desc="computing deltas", unit="site"):
        for f in range(features):
            last_observed = 0
            for t in range(days):
                if masks[s, t, f]:
                    deltas[s, t, f] = 0
                    last_observed = 0
                else:
                    last_observed += 1
                    deltas[s, t, f] = last_observed
    return deltas

def prepare_brits_records(data):
    """
    Package everything into a list of dicts for BRITS.
    One dict per site.
    """
    data, masks, evals, eval_masks = make_masks_and_eval(np.copy(data))
    deltas = compute_deltas(masks)

    records = []
    print("Preparing BRITS records...", flush=True)
    for s in tqdm(range(data.shape[0]), desc="Processing sites", unit="site"):
        record = {
            "values": np.nan_to_num(data[s]).tolist(),
            "masks": masks[s].tolist(),
            "evals": np.nan_to_num(evals[s]).tolist(),
            "eval_masks": eval_masks[s].tolist(),
            "deltas": deltas[s].tolist(),
            "forwards": pd.DataFrame(data[s]).fillna(method="ffill").fillna(0.0).to_numpy().tolist()
        }
        records.append(record)
    return records

records = prepare_brits_records(data)

with open("./json/improve_brits_data.json", "w") as f:
    for rec in tqdm(records, desc="Processing records", unit="record"):
        f.write(json.dumps({"forward": rec, "backward": rec}) + "\n")
