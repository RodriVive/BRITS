import os
import numpy as np
import pandas as pd
import ujson as json
from tqdm import tqdm
import xarray as xr

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
print(data_numeric.head())

print("data loaded successfully")
print("reshaping data...")

site_to_idx = {site: idx for idx, site in enumerate(unique_sites)}
site_data = {site: [] for site in unique_sites}

for idx, row in df.iterrows():
    site = row['site']
    site_data[site].append(row[data_numeric.columns].values)

data = np.zeros((len(unique_sites), max(len(v) for v in site_data.values()), len(data_numeric.columns)))
for site, idx in site_to_idx.items():
    site_array = np.array(site_data[site])
    data[idx, :len(site_array)] = site_array

print(f"Data shape: {data.shape}")
print(f"Number of sites: {len(unique_sites)}")
print(f"Number of features: {len(data_numeric.columns)}")
print(f"Max entries per site: {data.shape[1]}")

temp = pd.DataFrame(data[0], columns=data_numeric.columns)
print(temp.head())
print(data.shape)

def make_masks_and_eval(data, missing_ratio=0.1):
    masks = ~np.isnan(data)
    evals = np.copy(data)
    eval_masks = np.zeros_like(masks, dtype=bool)

    for i in tqdm(range(data.shape[0]), desc="making masks", unit="site"):
        # Exclude last 3 features (lat, lon, elev) from being masked
        known_indices = np.argwhere(masks[i, :, :-3])
        n_eval = int(len(known_indices) * missing_ratio)
        sampled = known_indices[np.random.choice(len(known_indices), n_eval, replace=False)]
        for d, f in sampled:
            data[i, d, f] = np.nan
            eval_masks[i, d, f] = True

    return data, masks.astype(int), evals, eval_masks.astype(int)

def compute_deltas(masks):
    sites, days, features = masks.shape
    deltas = np.zeros_like(masks, dtype=np.float32)

    for s in tqdm(range(sites), desc="computing deltas", unit="site"):
        for f in range(features - 3):  # Exclude lat, lon, elev
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
    data, masks, evals, eval_masks = make_masks_and_eval(np.copy(data))
    deltas = compute_deltas(masks)

    records = []
    print("Preparing BRITS records...", flush=True)
    for s in tqdm(range(data.shape[0]), desc="Processing sites", unit="site"):
        df = pd.DataFrame(data[s])
        df.iloc[:, :-3] = df.iloc[:, :-3].fillna(method="ffill").fillna(0.0)
        df.iloc[:, -3:] = df.iloc[:, -3:].fillna(0.0)  # Or keep NaNs if preferred

        record = {
            "values": np.nan_to_num(data[s]).tolist(),
            "masks": masks[s].tolist(),
            "evals": np.nan_to_num(evals[s]).tolist(),
            "eval_masks": eval_masks[s].tolist(),
            "deltas": deltas[s].tolist(),
            "forwards": df.to_numpy().tolist()
        }
        records.append(record)
    return records

records = prepare_brits_records(data)

os.makedirs("./json", exist_ok=True)
with open("./json/improve_brits_data.json", "w") as f:
    for rec in tqdm(records, desc="Processing records", unit="record"):
        f.write(json.dumps({"forward": rec, "backward": rec}) + "\n")
