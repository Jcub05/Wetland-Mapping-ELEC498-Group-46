# CELL 4: Extract Embeddings - VECTORIZED with ERROR HANDLING
print("\n" + "="*70)
print("EXTRACTING EMBEDDINGS - OPTIMIZED")
print("="*70)

n_samples = len(y_indices)
X = np.zeros((n_samples, 64), dtype=np.float32)
found_samples = np.zeros(n_samples, dtype=bool)

skipped_tiles = []

with tqdm(total=len(tile_files), desc="Tiles", unit=" tiles") as pbar:
    for tile_file in tile_files:
        try:
            with rasterio.open(tile_file) as tile_src:
                # Parse tile position
                parts = tile_file.stem.split('-')
                if len(parts) >= 3:
                    try:
                        tile_row_offset = int(parts[-2])
                        tile_col_offset = int(parts[-1])
                    except ValueError:
                        pbar.update(1)
                        continue
                else:
                    pbar.update(1)
                    continue
                
                # CHECK: Verify tile has 64 bands
                if tile_src.count != 64:
                    print(f"\n⚠️  Skipping {tile_file.name}: has {tile_src.count} bands instead of 64")
                    skipped_tiles.append(tile_file.name)
                    pbar.update(1)
                    continue
                
                tile_height, tile_width = tile_src.height, tile_src.width
                
                # Find samples in this tile
                in_tile_y = (y_indices >= tile_row_offset) & (y_indices < tile_row_offset + tile_height)
                in_tile_x = (x_indices >= tile_col_offset) & (x_indices < tile_col_offset + tile_width)
                in_tile_mask = in_tile_y & in_tile_x
                
                if in_tile_mask.any():
                    # Read tile once
                    tile_data = tile_src.read()  # (64, H, W)
                    
                    # Double-check shape after read
                    if tile_data.shape[0] != 64:
                        print(f"\n⚠️  Skipping {tile_file.name}: data shape is {tile_data.shape}")
                        skipped_tiles.append(tile_file.name)
                        pbar.update(1)
                        continue
                    
                    # Get local coordinates
                    local_y = y_indices[in_tile_mask] - tile_row_offset
                    local_x = x_indices[in_tile_mask] - tile_col_offset
                    
                    # VECTORIZED EXTRACTION
                    pixel_values = tile_data[:, local_y, local_x].T  # Shape: (n_pixels, 64)
                    
                    # Find valid samples (no NaN)
                    valid_mask = ~np.isnan(pixel_values).any(axis=1)
                    
                    # Get global indices for valid samples
                    global_indices = np.where(in_tile_mask)[0]
                    valid_global_indices = global_indices[valid_mask]
                    
                    # Assign valid samples
                    X[valid_global_indices, :] = pixel_values[valid_mask]
                    found_samples[valid_global_indices] = True
        
        except Exception as e:
            print(f"\n❌ Error processing {tile_file.name}: {e}")
            skipped_tiles.append(tile_file.name)
        
        pbar.update(1)
        pbar.set_postfix({"found": f"{found_samples.sum():,}/{n_samples:,}"})

print(f"\n✓ Extracted {found_samples.sum():,} / {n_samples:,} samples")

if skipped_tiles:
    print(f"\n⚠️  Skipped {len(skipped_tiles)} corrupted tiles:")
    for tile_name in skipped_tiles:
        print(f"     {tile_name}")

if not found_samples.all():
    missing = (~found_samples).sum()
    print(f"\n   ⚠ {missing:,} samples had NaN or were in corrupted tiles")
    
    print("\n   Missing by class:")
    for cls in np.unique(y):
        cls_mask = (y == cls)
        missing_cls = (~found_samples[cls_mask]).sum()
        if missing_cls > 0:
            print(f"     Class {cls}: {missing_cls:,} / {cls_mask.sum():,}")

print("✅ Extraction complete!")
