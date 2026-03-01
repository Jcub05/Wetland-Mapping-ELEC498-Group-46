import rasterio
import numpy as np
from collections import defaultdict
import os

labels_file = "C:/Users/jcube/OneDrive/Desktop/Jacob/School/Queens/Year 5/ELEC 490 Capstone/GITHUB REPO/Wetland-Mapping-ELEC498-Group-46/data_preprocessing/bow_river_wetlands_10m_final.tif"

# If the file isn't local, we'll need to run this in Colab instead
if not os.path.exists(labels_file):
    print("Labels file not found locally. This script is intended to be run in Colab where the drive is mounted.")
else:
    print(f"Scanning {labels_file}...")
    class_stats = {
        cls: {'count': 0, 'min_row': float('inf'), 'max_row': -1, 'min_col': float('inf'), 'max_col': -1}
        for cls in range(6)
    }

    with rasterio.open(labels_file) as src:
        windows = list(src.block_windows(1))
        for block_id, window in windows:
            data = src.read(1, window=window)
            row_off = window.row_off
            col_off = window.col_off
            
            for cls in range(6):
                mask = (data == cls)
                if not mask.any():
                    continue
                
                rows, cols = np.where(mask)
                global_rows = rows + row_off
                global_cols = cols + col_off
                
                class_stats[cls]['count'] += len(global_rows)
                class_stats[cls]['min_row'] = min(class_stats[cls]['min_row'], global_rows.min())
                class_stats[cls]['max_row'] = max(class_stats[cls]['max_row'], global_rows.max())
                class_stats[cls]['min_col'] = min(class_stats[cls]['min_col'], global_cols.min())
                class_stats[cls]['max_col'] = max(class_stats[cls]['max_col'], global_cols.max())

    print("\n" + "="*50)
    print("CLASS DISTRIBUTION & BOUNDING BOXES")
    print("="*50)
    for cls in range(6):
        stats = class_stats[cls]
        if stats['count'] == 0:
            print(f"Class {cls}: Not found")
            continue
            
        print(f"Class {cls}: {stats['count']:,} pixels")
        print(f"  Rows: {stats['min_row']} to {stats['max_row']} (Span: {stats['max_row'] - stats['min_row']})")
        print(f"  Cols: {stats['min_col']} to {stats['max_col']} (Span: {stats['max_col'] - stats['min_col']})")
        print("-" * 50)
