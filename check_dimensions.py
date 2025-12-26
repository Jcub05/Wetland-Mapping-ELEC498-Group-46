import rasterio

print("=== Checking Raster Dimensions ===\n")

# Check embeddings VRT
print("1. Embeddings (VRT):")
with rasterio.open("bow_river_embeddings_2020.vrt") as src:
    print(f"   Shape: {src.count} bands x {src.height} rows x {src.width} cols")
    print(f"   CRS: {src.crs}")
    print(f"   Bounds: {src.bounds}")
    print(f"   Transform: {src.transform}")
    print(f"   Resolution: {src.res}")

print("\n2. Labels:")
with rasterio.open("bow_river_wetlands_10m_final.tif") as src:
    print(f"   Shape: {src.count} bands x {src.height} rows x {src.width} cols")
    print(f"   CRS: {src.crs}")
    print(f"   Bounds: {src.bounds}")
    print(f"   Transform: {src.transform}")
    print(f"   Resolution: {src.res}")

print("\n=== Analysis ===")
with rasterio.open("bow_river_embeddings_2020.vrt") as emb_src:
    with rasterio.open("bow_river_wetlands_10m_final.tif") as lab_src:
        if emb_src.shape != lab_src.shape:
            print(f"❌ Dimensions DON'T match!")
            print(f"   Embeddings: {emb_src.height} x {emb_src.width}")
            print(f"   Labels: {lab_src.height} x {lab_src.width}")
            print(f"   Difference: {emb_src.height - lab_src.height} x {emb_src.width - lab_src.width}")
        
        if emb_src.crs != lab_src.crs:
            print(f"❌ CRS doesn't match!")
        
        if emb_src.res != lab_src.res:
            print(f"❌ Resolution doesn't match!")
            print(f"   Embeddings: {emb_src.res}")
            print(f"   Labels: {lab_src.res}")
