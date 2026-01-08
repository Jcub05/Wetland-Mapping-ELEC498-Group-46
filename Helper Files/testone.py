import rasterio
import numpy as np

with rasterio.open('bow_river_embeddings_2020_CORRECTED-0000021504-0000009216.tif') as src:
    data = src.read(1, window=((0, 10), (0, 10)))
    print("Sample data:", data[:5, :5])
    print("All NaN?:", np.all(np.isnan(data)))