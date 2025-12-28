# Alignment Issue: 1-Row Mismatch

## Current Status
- **Embeddings VRT**: 20606 × 31428 pixels
- **Labels**: 20607 × 31428 pixels  
- **Difference**: Missing 1 row (31,428 pixels)
- **CRS**: ✓ Both EPSG:32612
- **Bounds**: ❌ Don't match exactly

## Options

### Option 1: Crop Labels to Match Embeddings (FASTEST - 2 min)
**Pros:**
- Very fast
- Data already downloaded
- Lose only 0.005% of data (31k out of 648M pixels)

**Cons:**
- Slight spatial mismatch
- Not "perfect" alignment

**Implementation:**
```python
labels_cropped = labels[:20606, :]  # Remove last row
```

### Option 2: Pad Embeddings with Zeros (FAST - 2 min)
**Pros:**
- Keeps all label data
- Fast to implement

**Cons:**
- Extra row will have no embeddings (zeros)
- Could confuse model if that row has labels

### Option 3: Re-download with Exact Parameters (SLOW - 2-3 hours)
**Pros:**
- Perfect alignment guaranteed

**Cons:**
- Another 2-3 hour wait
- May still have same issue (GEE export quirk)

### Option 4: Use As-Is with Spatial Matching (MEDIUM - 10 min)
**Pros:**
- Uses all data
- Handles alignment programmatically

**Cons:**
- More complex code
- Need to resample one raster

## Recommendation

**Option 1: Crop Labels**

The 1-row difference is likely due to Google Earth Engine's tiling/export quirks. Losing 0.005% of pixels (probably at the edge) won't affect your model training. This gets you running immediately.

If you need perfect alignment for production, use Option 3 later.
