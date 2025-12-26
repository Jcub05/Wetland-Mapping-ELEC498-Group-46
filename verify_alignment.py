"""
Quick verification script to check if the new embeddings download
matches the labels raster dimensions perfectly
"""
import rasterio
import os

# File to check (update the filename once download completes)
embeddings_file = "bow_river_embeddings_2020_matched.tif"  # Update this name
labels_file = "bow_river_wetlands_10m_final.tif"

if not os.path.exists(embeddings_file):
    print(f"❌ Embeddings file not found: {embeddings_file}")
    print("Update the filename in this script once the download completes.")
    exit(1)

print("=== Checking Alignment ===\n")

with rasterio.open(embeddings_file) as emb:
    with rasterio.open(labels_file) as lab:
        
        print("Embeddings:")
        print(f"  Dimensions: {emb.height} x {emb.width}")
        print(f"  Bands: {emb.count}")
        print(f"  CRS: {emb.crs}")
        print(f"  Bounds: {emb.bounds}")
        print(f"  Transform: {emb.transform}")
        
        print("\nLabels:")
        print(f"  Dimensions: {lab.height} x {lab.width}")
        print(f"  Bands: {lab.count}")
        print(f"  CRS: {lab.crs}")
        print(f"  Bounds: {lab.bounds}")
        print(f"  Transform: {lab.transform}")
        
        print("\n=== Verification ===")
        
        # Check dimensions
        if (emb.height, emb.width) == (lab.height, lab.width):
            print("✓ Dimensions MATCH!")
        else:
            print(f"❌ Dimensions mismatch: {emb.height}x{emb.width} vs {lab.height}x{lab.width}")
        
        # Check CRS
        if emb.crs == lab.crs:
            print("✓ CRS matches!")
        else:
            print(f"❌ CRS mismatch: {emb.crs} vs {lab.crs}")
        
        # Check transform
        if emb.transform == lab.transform:
            print("✓ Transform MATCHES (perfect pixel alignment)!")
        else:
            print(f"⚠ Transform differs slightly")
            print(f"  This might be okay if bounds are very close")
        
        # Check bounds
        bounds_close = all(abs(a - b) < 1.0 for a, b in zip(emb.bounds, lab.bounds))
        if bounds_close:
            print("✓ Bounds match (within 1m tolerance)!")
        else:
            print(f"❌ Bounds significantly different")
        
        print("\n" + "="*50)
        if all([
            (emb.height, emb.width) == (lab.height, lab.width),
            emb.crs == lab.crs,
            bounds_close
        ]):
            print("🎉 SUCCESS! Embeddings and labels are perfectly aligned!")
            print("You can now run your dataloader.py")
        else:
            print("⚠ Issues detected - may need reprojection")
