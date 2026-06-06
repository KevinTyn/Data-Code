# Data-Code
Data and code related to ACEVI

## Project Structure

```
├── InSAR_Mosaic/         # Core InSAR image mosaic algorithms (Pareto + ALS + IGG-III)
│   ├── 1224_new2.py      # Main script: joint constrained modeling with Pareto optimization
│   ├── without_coh.py    # Variant without coherence weighting
│   ├── 0.75.py           # Variant with 0.75 coherence threshold
│   ├── evaluate022.py    # Interactive validation/evaluation tool
│   ├── statistic_points.py # Tie point statistics
│   └── history/          # Archived previous versions
│
├── Analysis/             # Analysis, visualization & method comparison
│   ├── rmse_mae_std.py   # RMSE/MAE/STD comparison plots
│   ├── scatter4.py       # GPU-accelerated scatter density comparison
│   ├── precision.py      # Error profile re-plotting
│   ├── ia_diff.py        # Interactive TIF global comparison
│   └── efficiency_comparison/  # Method comparison study (Const vs Linear vs AINA)
│
├── Transform/            # GeoTIFF preprocessing & format conversion utilities
│   ├── reprojection.py   # Batch reprojection
│   ├── annual_mean.py    # Annual mean computation
│   ├── los2vertical.py   # LOS to vertical conversion
│   ├── nc2tif.py         # NetCDF to GeoTIFF
│   ├── tif2shp.py        # TIFF boundary to shapefile
│   └── ...
│
├── formulas_explanation.txt  # Mathematical documentation of core formulas
├── others/               # Previous repository content
└── README.md
```

## Core Method: Pareto_ALS InSAR Mosaic

The InSAR mosaic pipeline uses **Pareto-front optimization** with **IGG-III robust weights** in an **Alternating Least Squares (ALS)** framework for aligning multiple InSAR velocity maps. The workflow includes:

1. **Joint constrained modeling** (Constant / Linear) with Pareto-optimal coupling parameter selection
2. **IGG-III iterative robust weighting** to handle outliers
3. **RANSAC spatial ramp correction** with smart axis locking
4. **Global network adjustment** via sparse least squares

Supports both same-track (high stability) and cross-track (linear model) mosaic modes with GPU (CuPy) acceleration and CPU fallback.
