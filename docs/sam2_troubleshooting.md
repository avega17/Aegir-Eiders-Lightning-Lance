# SAM2 Troubleshooting Guide

## Problem 1: AttributeError with SamGeo2

### Error Messages
```python
AttributeError: 'SamGeo2' object has no attribute 'image'
AttributeError: 'SamGeo2' object has no attribute 'source'
```

### Root Cause

The issue is a **bug in the `samgeo2.py` source code** (lines 484-512). When passing a **numpy array** to `set_image()`, the method does NOT set the `self.image` and `self.source` attributes:

```python
def set_image(self, image: Union[str, np.ndarray, Image]) -> None:
    if isinstance(image, str):
        # ... sets self.source and self.image
        self.source = image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
    elif isinstance(image, np.ndarray) or isinstance(image, Image):
        pass  # ← BUG: Does nothing! Doesn't set self.image or self.source
    
    self.predictor.set_image(image)  # Only sets predictor's image
```

Later, when `predict()` is called with `point_crs` parameter, it tries to access `self.source` for coordinate transformation (line 1032):

```python
if (point_crs is not None) and (point_coords is not None):
    point_coords, out_of_bounds = common.coords_to_xy(
        self.source,  # ← AttributeError! self.source was never set
        point_coords, 
        point_crs, 
        return_out_of_bounds=True
    )
```

Similarly, other methods like `show_anns()` try to access `self.image`, causing the same error.

### Why This Wasn't Caught

1. **Documentation examples use file paths**: All samgeo documentation examples pass file paths to `set_image()`, not numpy arrays
2. **No GitHub issues**: This suggests most users work with file-based workflows
3. **Environment-specific**: The bug only manifests when:
   - Using numpy arrays directly (common in notebook workflows)
   - Using `point_crs` parameter for coordinate transformation
   - Calling visualization methods that reference `self.image`

### Solution 1: Save to File First (Recommended)

The cleanest solution is to save the numpy array to a temporary file:

```python
from PIL import Image

# Save numpy array to temporary file
temp_image_path = "../data/temp_sam_input.png"
Image.fromarray(hr_rgb).save(temp_image_path)

# Initialize SAM2
sam = SamGeo2(
    model_id="sam2-hiera-small",
    automatic=False,
    device="cuda"
)

# Set image from file path (works correctly)
sam.set_image(temp_image_path)

# Predict with CRS transformation
sam.predict(
    boxes=boxes,
    point_crs="EPSG:4326",  # Now works because self.source is set
    output=output_path,
    dtype="uint8",
    multimask_output=False
)
```

**Key Points:**
- Saves image to disk once, can be reused across runs
- No manual attribute setting required
- CRS transformation works properly
- Cleaner and more maintainable code

### Solution 2: Use Original SAM or SAM-HQ

The original `SamGeo` and `SamGeoHQ` classes handle numpy arrays correctly:

```python
from samgeo import SamGeo

sam = SamGeo(
    model_type="vit_h",
    automatic=False,
    sam_kwargs=None,
)

sam.set_image(hr_rgb)  # Works correctly with numpy arrays

sam.predict(
    boxes=boxes,
    point_crs="EPSG:4326",  # CRS transformation works
    output=output_path,
    dtype="uint8",
    multimask_output=False
)
```

## Coordinate Transformation

When using the file-based approach (Solution 1), SAM2 handles CRS transformation automatically:

```python
# Just pass lat/lon coordinates directly
boxes = [[lon_min, lat_min, lon_max, lat_max]]

sam.predict(
    boxes=boxes,
    point_crs="EPSG:4326",  # SAM2 converts to pixel coords automatically
    output=output_path,
    dtype="uint8",
    multimask_output=False
)
```

**Note**: The `point_crs` parameter requires `self.source` to be set (which is why the numpy array bug causes issues).

## Comparison: SAM vs SAM2 vs SAM-HQ

| Feature | Original SAM | SAM2 | SAM-HQ |
|---------|-------------|------|--------|
| **Numpy array support** | ✅ Works | ❌ Bug | ✅ Works |
| **CRS transformation** | ✅ Works | ⚠️ Requires workaround | ✅ Works |
| **Speed** | Baseline | ~6x faster | Similar to SAM |
| **Quality** | Good | Better | Best (high-quality masks) |
| **Video support** | ❌ No | ✅ Yes | ❌ No |
| **Model size** | Large | Smaller options | Large |

## Recommendations

1. **For production pipelines**: Use original SAM or SAM-HQ until the SAM2 bug is fixed
2. **For experimentation**: Use the workaround in Solution 1
3. **For video**: Use SAM2 with file-based workflow (Solution 3)
4. **For highest quality masks**: Use SAM-HQ with point or box prompts

## Testing Checklist

After applying fixes, verify:
- [ ] `sam.image` is set (check with `hasattr(sam, 'image')`)
- [ ] `sam.source` is set (check with `hasattr(sam, 'source')`)
- [ ] Image dtype is `uint8` (check with `sam.image.dtype`)
- [ ] Image shape is `(height, width, 3)` (check with `sam.image.shape`)
- [ ] `sam.predict()` runs without AttributeError
- [ ] Output mask is generated successfully

## Related Issues

- **dtype mismatch**: Fixed by converting to `uint8` before passing to SAM
- **CRS transformation**: Requires `self.source` to be set (triggers the bug)
- **Visualization methods**: Require `self.image` to be set (triggers the bug)

## Problem 2: CRS Error with point_crs

### Error Message

```python
CRSError: Invalid projection: : (Internal Proj Error: proj_create: unrecognized format / unknown name)
```

### Root Cause

When using `point_crs="EPSG:4326"`, SAM2's `coords_to_xy()` function tries to read the CRS from the image file using `rasterio.open(src_fp)`. If the image is saved as PNG (or any format without geospatial metadata), `src.crs` is `None`, causing pyproj to fail.

### Solution

Save the image as a **GeoTIFF with proper CRS and geotransform metadata**:

```python
import rasterio
from rasterio.transform import from_bounds

temp_image_path = "../data/temp_sam_input.tif"

# Get image bounds and CRS from xarray tree
x_coords = tree["hr"].ds.x.values
y_coords = tree["hr"].ds.y.values
bounds = (x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())

# Get CRS from tree (core-five uses various projected CRS, not always EPSG:4326)
crs = tree.crs

# Create geotransform from bounds
transform = from_bounds(*bounds, hr_rgb.shape[1], hr_rgb.shape[0])

# Save as GeoTIFF with metadata
with rasterio.open(
    temp_image_path,
    'w',
    driver='GTiff',
    height=hr_rgb.shape[0],
    width=hr_rgb.shape[1],
    count=3,
    dtype=hr_rgb.dtype,
    crs=crs,
    transform=transform
) as dst:
    # Write RGB bands
    for i in range(3):
        dst.write(hr_rgb[:, :, i], i + 1)
```

**Important**: Use `tree.crs` (not hardcoded `"EPSG:4326"`) because core-five datasets use various projected coordinate systems.

## Problem 3: IndexError with tensor_to_numpy

### Error Message

```python
IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed
```

### Root Cause

When passing `output=` parameter to `predict()` with a **single box**, SAM2 returns masks with shape `(1, H, W)` (3D), but the `tensor_to_numpy()` function expects `(N, 3, H, W)` (4D) for multiple boxes. This is a bug in samgeo's handling of single-box predictions.

### Solution

Don't pass `output=` to `predict()`. Instead, get the masks with `return_results=True` and save manually:

```python
# Don't pass output= to avoid tensor_to_numpy bug with single boxes
masks, scores, logits = sam.predict(
    boxes=boxes,
    point_crs="EPSG:4326",
    multimask_output=False,
    return_results=True
)

# Get the mask (for single box with multimask_output=False, shape is (1, H, W))
mask = masks[0]  # Get first (and only) mask

# Save with same geospatial metadata as input
output_path = "../data/sam2_box_mask.tif"

with rasterio.open(temp_image_path) as src:
    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype='uint8',
        compress='deflate'
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write((mask * 255).astype('uint8'), 1)
```

**Benefits**:
- Preserves geospatial metadata from input image
- Works with single or multiple boxes
- More control over output format

## Future Work

Consider submitting pull requests to `segment-geospatial` to fix these bugs:

1. **Fix `set_image()` for numpy arrays** (Problem 1)
2. **Fix `tensor_to_numpy()` for single boxes** (Problem 3)

