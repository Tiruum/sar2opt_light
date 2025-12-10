import glob
import numpy as np
import rasterio

sar_paths = sorted(glob.glob("data/sen12/agri/s1/ROIs1868_summer_s1_59_p2.png", recursive=True))  # подстрой под свою структуру
mins, maxs, means = [], [], []
for i, path in enumerate(sar_paths[:100]):  # достаточно 50–100 тайлов
    with rasterio.open(path) as src:
        sar = src.read().astype(np.float32)     # (C, H, W)
        sar = sar[0]                            # возьмём один канал VV, (H, W)
    mins.append(sar.min())
    maxs.append(sar.max())
    means.append(sar.mean())

print(f'Images quantity: {len(sar_paths)}')
print("GLOBAL min:", float(np.min(mins)))
print("GLOBAL max:", float(np.max(maxs)))
print("MEAN of means:", float(np.mean(means)))

all_vals = []
for path in sar_paths[:50]:
    with rasterio.open(path) as src:
        sar = src.read(1).astype(np.float32)  # один канал
    all_vals.append(sar.flatten())

all_vals = np.concatenate(all_vals)
for p in [1, 5, 50, 95, 99]:
    print(f"P{p}:", np.percentile(all_vals, p))
